from torch import nn
import math
import torch
import torch.nn.functional as F
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY


# Top-T Sparse Cross Attention used by the anatomy-guided Gaussian parametrizer.
class TopTSparseCrossAttention(nn.Module):
    '''
    top-m cross attention module using Linear layers
    adapted from Topm_CrossAttention_Restormer for sequence data
    '''

    def __init__(self, dim, num_heads, bias):
        super(TopTSparseCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Linear layers for q, k, v
        self.kv = nn.Linear(dim, dim * 2, bias=bias)
        self.q = nn.Linear(dim, dim, bias=bias)

        self.project_out = nn.Linear(dim, dim, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        # Learnable weights for different sparsity levels
        self.attn1 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x_q, x_kv):
        '''
        Args:
            x_q: query tensor, shape (b, n, c)
            x_kv: key-value tensor, shape (b, n, c)
        Returns:
            out: output tensor, shape (b, n, c)
        '''
        b, n, c = x_q.shape

        # Generate q, k, v
        q = self.q(x_q)  # (b, n, c)
        kv = self.kv(x_kv)  # (b, n, 2*c)
        k, v = kv.chunk(2, dim=-1)  # each (b, n, c)

        # Reshape for multi-head attention
        q = rearrange(q, 'b n (head c) -> b head c n', head=self.num_heads)
        k = rearrange(k, 'b n (head c) -> b head c n', head=self.num_heads)
        v = rearrange(v, 'b n (head c) -> b head c n', head=self.num_heads)

        # Normalize q and k
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        _, _, C, N = q.shape

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (b, head, C, C)

        # Create masks for different sparsity levels
        mask1 = torch.zeros(b, self.num_heads, C, C, device=x_q.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x_q.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x_q.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x_q.device, requires_grad=False)

        # Top-k selection with different k values
        # Level 1: top 1/2
        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        # Level 2: top 2/3
        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        # Level 3: top 3/4
        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        # Level 4: top 4/5
        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        # Apply softmax
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        # Compute outputs for each sparsity level
        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        # Weighted fusion of different sparsity levels
        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        # Reshape back
        out = rearrange(out, 'b head c n -> b n (head c)')

        # Project output
        out = self.project_out(out)
        return out


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.ReLU):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

@ARCH_REGISTRY.register()
# Anatomy-Guided Gaussian Parametrizer (AGGP).
class AnatomyGuidedGaussianParametrizer(nn.Module):
    def __init__(self, channel=180, num_colors=3, gs_up_factor=1.0, window_size=8, num_gs_seed=64, channel_mean=20, num_heads=4):
        super(AnatomyGuidedGaussianParametrizer, self).__init__()
        self.num_colors = num_colors
        self.window_size = window_size
        self.num_gs_seed_sqrt = int(math.sqrt(num_gs_seed))
        self.norm = nn.LayerNorm(channel)
        self.channel_mean = channel_mean
        self.num_heads = num_heads
        # GS sigma_x, sigma_y
        self.mlp_block_sigma = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(2 * gs_up_factor))
        )

        # GS rho
        self.mlp_block_rho = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(1 * gs_up_factor))
        )

        # GS alpha
        self.mlp_block_alpha = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(1 * gs_up_factor))
        )

        # GS RGB values
        self.mlp_block_rgb = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(self.num_colors * gs_up_factor))
        )

        # GS mean_x, mean_y
        self.mlp_block_mean = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(2 * gs_up_factor))
        )

        self.mlp_crossattn = MLP(channel, channel * 2, channel)

        # Top-T sparse attention injects anatomy-gradient information into Gaussian means.
        self.topm_cross_attn = TopTSparseCrossAttention(channel, num_heads=num_heads, bias=True)
        # Map gradient to channel_mean dimension
        self.mlp_ref_gra = nn.Linear(1, channel)

    @staticmethod
    def compute_ref_gradient(ref, normalize=True, eps=1e-6):
        """
        计算参考图像的梯度图（单通道或多通道均可），并归一化到[0,1]
        输入 ref 形状可以是 (B, C, H, W)、(B, H, W) 或 (H, W)。
        输出与输入形状一致（若输入不含 batch/channel，会相应返回相同维度）。
        使用 Sobel 滤波器计算 x,y 方向梯度，合成梯度幅值作为结果。
        """
        # 保存原始维度，以便返回相同形状
        orig_dim = ref.dim()
        # 将 (H,W) -> (1,1,H,W), (B,H,W) -> (B,1,H,W)
        if orig_dim == 2:
            ref = ref.unsqueeze(0).unsqueeze(0)
        elif orig_dim == 3:
            # (B,H,W) -> (B,1,H,W)
            ref = ref.unsqueeze(1)
        # 现在 ref 形状为 (B, C, H, W)
        b, c, h, w = ref.shape
        # 若多通道，先转为单通道（均值）
        if c != 1:
            ref = ref.mean(dim=1, keepdim=True)
        device = ref.device
        dtype = ref.dtype
        # Sobel 核
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3)
        gx = F.conv2d(ref, sobel_x, padding=1)
        gy = F.conv2d(ref, sobel_y, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + eps)
        if normalize:
            # 按每个样本归一化
            grad_view = grad.view(b, -1)
            min_val = grad_view.min(dim=1)[0].view(b, 1, 1, 1)
            max_val = grad_view.max(dim=1)[0].view(b, 1, 1, 1)
            grad = (grad - min_val) / (max_val - min_val + eps)
        # 根据原始维度返回相同形状
        if orig_dim == 2:
            return grad.squeeze(0).squeeze(0)
        if orig_dim == 3:
            return grad.squeeze(1)
        return grad

    @staticmethod
    def get_N_reference_points(h, w, device='cuda'):
        # step_y = 1/(h+1)
        # step_x = 1/(w+1)
        step_y = 1 / h
        step_x = 1 / w
        ref_y, ref_x = torch.meshgrid(torch.linspace(step_y / 2, 1 - step_y / 2, h, dtype=torch.float32, device=device),
                                      torch.linspace(step_x / 2, 1 - step_x / 2, w, dtype=torch.float32, device=device))
        reference_points = torch.stack((ref_x.reshape(-1), ref_y.reshape(-1)), -1)
        reference_points = reference_points[None, :, None]
        return reference_points

    def forward(self, query, ref):
        '''
        using deformable detr decoder for cross attention
        Args:
            query: (batch_size, num_query, dim)
            query_pos: (batch_size, num_query, dim)
            srcs: (batch_size, dim, h1, w1)
        '''

        ref_gra = self.compute_ref_gradient(ref)

        b, h, w, c = query.shape

        query_sigma = self.mlp_block_sigma(query).reshape(b, -1, 2)

        query_rho = self.mlp_block_rho(query).reshape(b, -1, 1)
        query_alpha = self.mlp_block_alpha(query).reshape(b, -1, 1)
        query_rgb = self.mlp_block_rgb(query).reshape(b, -1, self.num_colors)

        # MDTA  TOP_m Cross Attention
        # Prepare ref_gra as query: (b, 1, h, w) -> (b, h*w, 1)
        ref_gra_flat = ref_gra.reshape(b, 1, -1).transpose(1, 2)  # (b, h*w, 1)
        # Map gradient to channel_mean dimension
        ref_gra_embed = self.mlp_ref_gra(ref_gra_flat)  # (b, h*w, channel_mean)

        # Prepare query_mean as key and value
        query_mean = query.reshape(b, -1, c)  # (b, h*w, channel_mean)

        # Apply TOP-m cross attention: q is ref_gra_embed, kv is query_mean
        query_mean_attn = self.topm_cross_attn(ref_gra_embed, query_mean)  # (b, h*w, channel_mean)

        # Residual connection
        query_mean = query_mean + query_mean_attn

        # FFN
        query_mean = query_mean.reshape(b, -1, c)
        resi = query_mean
        query_mean = self.norm(query_mean)
        query_mean = self.mlp_crossattn(query_mean)
        query_mean = query_mean + resi


        query_mean = self.mlp_block_mean(query_mean).reshape(b, -1, 2)

        query_mean = query_mean / torch.tensor(
            [self.num_gs_seed_sqrt * (w // self.window_size),
             self.num_gs_seed_sqrt * (h // self.window_size)])[
            None, None].to(query_mean.device)  # b, h_count*w_count*num_gs_seed, 2

        reference_offset = self.get_N_reference_points(
            self.num_gs_seed_sqrt * (h // self.window_size),
            self.num_gs_seed_sqrt * (w // self.window_size), query.device)
        query_mean = query_mean + reference_offset.reshape(1, -1, 2)

        query = torch.cat([query_sigma, query_rho, query_alpha, query_rgb, query_mean],
                          dim=-1)  # b, h_count*w_count*num_gs_seed, 9
        return query

if __name__ == '__main__':

    net = AnatomyGuidedGaussianParametrizer(channel=180, num_colors=3, gs_up_factor=1.0, window_size=8, num_gs_seed=64, channel_mean=20, num_heads=4).cuda()
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")
    ref = torch.rand((1, 1, 256, 256)).cuda()
    # 计算参考图像梯度（归一化到 [0,1]），返回形状与 ref 相同
    ref_gra = net.compute_ref_gradient(ref)
    print('ref shape:', ref.shape, 'ref_gra shape:', ref_gra.shape, 'min/max:', ref_gra.min().item(), ref_gra.max().item())
    x = torch.randn((1, 256, 256, 180)).cuda()
    y = net(x, ref)
    print('Output shape:', y.shape)


