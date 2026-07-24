from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
import copy
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_, kaiming_normal_
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from basicsr.archs.arch_util import trunc_normal_
import numpy as np


def fft_mag_log1p(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, C, H, W]
    Returns: log1p(|FFT2(x)|) with shape [B, C, H, W].
    """
    X = torch.fft.fft2(x, norm="ortho")
    mag = torch.abs(X)
    mag = torch.log1p(mag)
    return mag


def window_partition(x, window_size):
    # x is the feature from net_g
    b, c, h, w = x.shape
    windows = rearrange(x, 'b c (h_count dh) (w_count dw) -> (b h_count w_count) (dh dw) c', dh=window_size,
                        dw=window_size)
    # h_count = h // window_size
    # w_count = w // window_size
    # windows = x.reshape(b,c,h_count, window_size, w_count, window_size)
    # windows = windows.permute(0,1,2,4,3,5) #b,c,h_count,w_count,window_size,window_size
    # windows = windows.reshape(b,c,h_count*w_count, window_size * window_size)
    # windows = windows.permute(0,2,3,1) #b,h_count*w_count, window_size*window_size,c
    # windows = windows.reshape(-1, window_size*window_size, c)

    return windows


def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos


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


# Anatomy-Guided Dual-Domain Cross Attention (AG-DDCA): dual spatial-frequency branch.
class WindowCrossAttn(nn.Module):
    def __init__(self, dim=180, num_heads=6, window_size=12, num_gs_seed=2304):
        super(WindowCrossAttn, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_gs_seed = num_gs_seed
        self.num_gs_seed_sqrt = int(math.sqrt(num_gs_seed))

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias shared by the two branches.
        coords_source = (np.indices((self.num_gs_seed_sqrt, self.num_gs_seed_sqrt)) + 0.5) * self.window_size
        coords_tgt = (np.indices((self.window_size, self.window_size)) + 0.5) * self.num_gs_seed_sqrt
        coords_delta = coords_source.reshape(2,-1)[:, :, np.newaxis] - coords_tgt.reshape(2, -1)[:, np.newaxis, :]
        shape = coords_delta.shape
        indexed_coords_delta = coords_delta.flatten()
        indexed_coords_delta = list(map(list(set(sorted(indexed_coords_delta))).index, indexed_coords_delta))
        indexed_coords_delta = np.array(indexed_coords_delta).reshape(shape)

        indexed_coords_delta[0] *= indexed_coords_delta.max()
        indexed_coords_delta = torch.from_numpy(indexed_coords_delta.sum(0))
        self.register_buffer('relative_position_index', indexed_coords_delta)

        relative_position_bias_table = torch.zeros(
            (2 * max(self.num_gs_seed_sqrt, window_size) - 1) * (2 * max(self.num_gs_seed_sqrt, window_size) - 1),
            num_heads
        )
        for source_pix, closest_tgt_pix in enumerate((coords_delta**2).sum(0).argmin(1)):
            rel_pos_idx = indexed_coords_delta[source_pix][closest_tgt_pix]
            relative_position_bias_table[rel_pos_idx] = 2
        self.relative_position_bias_table = nn.Parameter(relative_position_bias_table)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        # Shared Q (from gs) with branch-specific K, V, and projection layers.
        self.qhead = nn.Linear(dim, dim, bias=True)

        # Spatial branch.
        self.khead_s = nn.Linear(dim, dim, bias=True)
        self.vhead_s = nn.Linear(dim, dim, bias=True)
        self.proj_s  = nn.Linear(dim, dim)

        # Frequency branch.
        self.khead_f = nn.Linear(dim, dim, bias=True)
        self.vhead_f = nn.Linear(dim, dim, bias=True)
        self.proj_f  = nn.Linear(dim, dim)

        # Dynamic gate that fuses the two branches using the current window's gs.
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 2)
        )

        self.softmax = nn.Softmax(dim=-1)

    def _relative_bias(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.num_gs_seed_sqrt * self.num_gs_seed_sqrt,
            self.window_size * self.window_size,
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias

    def _attend(self, q, k, v, rel_bias):
        # q: [B, H, Ng, Dh], k/v: [B, H, Nw, Dh]
        attn = (q @ k.transpose(-2, -1))  # [B,H,Ng,Nw]
        attn = attn + rel_bias.unsqueeze(0)
        attn = self.softmax(attn)
        out = (attn @ v).transpose(1, 2)  # [B,Ng,H,Dh]
        out = out.reshape(out.shape[0], out.shape[1], -1)  # [B,Ng,C]
        return out

    def forward(self, gs, feat_all):
        """
        gs:   [B*, Ng, C]，B* = b*h_count*w_count
        feat_all: [feat_lq_win, feat_guide_win, feat_lq_freq_win, feat_guide_freq_win]
                  each with shape [B*, Nw, C], where Nw = window_size * window_size.
        """
        feat_lq_win, feat_guide_win, feat_lq_freq_win, feat_guide_freq_win = feat_all
        B_, Ng, C = gs.shape
        Nw = feat_lq_win.shape[1]
        H = self.num_heads
        Dh = C // H

        # Shared Q.
        q = self.qhead(gs).reshape(B_, Ng, H, Dh).permute(0, 2, 1, 3)  # [B*,H,Ng,Dh]

        # Relative position bias.
        rel_bias = self._relative_bias()  # [H, Ng, Nw]

        # ----- Spatial branch -----
        k_s = self.khead_s(feat_guide_win).reshape(B_, Nw, H, Dh).permute(0, 2, 1, 3)
        v_s = self.vhead_s(feat_lq_win).reshape(B_, Nw, H, Dh).permute(0, 2, 1, 3)
        q_s = q * (Dh ** -0.5)
        out_s = self._attend(q_s, k_s, v_s, rel_bias)  # [B*,Ng,C]
        out_s = self.proj_s(out_s)

        # ----- Frequency branch -----
        k_f = self.khead_f(feat_guide_freq_win).reshape(B_, Nw, H, Dh).permute(0, 2, 1, 3)
        v_f = self.vhead_f(feat_lq_freq_win).reshape(B_, Nw, H, Dh).permute(0, 2, 1, 3)
        q_f = q * (Dh ** -0.5)
        out_f = self._attend(q_f, k_f, v_f, rel_bias)  # [B*,Ng,C]
        out_f = self.proj_f(out_f)

        # ----- Dynamic gated fusion -----
        # Use the mean gs token in the current window to compute the two branch weights.
        gate_logits = self.gate(gs.mean(dim=1))           # [B*, 2]
        gate = torch.softmax(gate_logits, dim=-1)         # [B*, 2]
        w_s = gate[:, 0].unsqueeze(1).unsqueeze(2)        # [B*,1,1]
        w_f = gate[:, 1].unsqueeze(1).unsqueeze(2)        # [B*,1,1]

        x = w_s * out_s + w_f * out_f                     # [B*,Ng,C]
        return x



class WindowCrossAttnLayer(nn.Module):
    def __init__(self, dim=180, num_heads=6, window_size=12, shift_size=0, num_gs_seed=2308):
        super(WindowCrossAttnLayer, self).__init__()

        self.gs_cross_attn_scale = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.shift_size = shift_size
        self.window_size = window_size

        self.window_cross_attn = WindowCrossAttn(dim=dim, num_heads=num_heads, window_size=window_size,
                                                 num_gs_seed=num_gs_seed)
        self.mlp_crossattn_scale = MLP(in_features=dim, hidden_features=dim, out_features=dim)
        self.mlp_crossattn_feature = MLP(in_features=dim, hidden_features=dim, out_features=dim)

    def forward(self, x, query_pos, feat, scale_embedding):
        # x: [B*, Ng, C]; query_pos: [B*, Ng, C]
        # feat: [feat_spatial, feat_guide, feat_freq, feat_guide_freq], each [B, C, H, W].
        # scale_embedding: [B*, Ng, C]
        resi = x
        x = self.norm1(x)
        x, _ = self.gs_cross_attn_scale(with_pos_embed(x, query_pos), scale_embedding, scale_embedding)
        x = resi + x

        resi = x
        x = self.norm2(x)
        x = self.mlp_crossattn_scale(x)
        x = resi + x

        # ---- Assemble window features, including shift. ----
        resi = x
        x = self.norm3(x)

        feat_spatial, feat_guide, feat_freq, feat_guide_freq = feat

        if self.shift_size > 0:
            shift_feat_lq    = torch.roll(feat_spatial,      shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            shift_feat_guide = torch.roll(feat_guide,         shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            shift_freq_lq    = torch.roll(feat_freq,          shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            shift_freq_guide = torch.roll(feat_guide_freq,    shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shift_feat_lq, shift_feat_guide = feat_spatial, feat_guide
            shift_freq_lq, shift_freq_guide = feat_freq,    feat_guide_freq

        # Partition each feature map into window tokens.
        win_lq        = window_partition(shift_feat_lq,    self.window_size)  # [B*, Nw, C]
        win_guide     = window_partition(shift_feat_guide, self.window_size)  # [B*, Nw, C]
        win_freq_lq   = window_partition(shift_freq_lq,    self.window_size)  # [B*, Nw, C]
        win_freq_guid = window_partition(shift_freq_guide, self.window_size)  # [B*, Nw, C]

        # Dual-branch cross attention.
        x = self.window_cross_attn(with_pos_embed(x, query_pos),
                                   [win_lq, win_guide, win_freq_lq, win_freq_guid])

        x = resi + x

        resi = x
        x = self.norm4(x)
        x = self.mlp_crossattn_feature(x)
        x = resi + x

        return x



class WindowCrossAttnBlock(nn.Module):
    def __init__(self, dim=180, window_size=12, num_heads=6, num_layers=4, num_gs_seed=2308):
        super(WindowCrossAttnBlock, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            WindowCrossAttnLayer(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                num_gs_seed=num_gs_seed) for i in range(num_layers)
        ])

    def forward(self, x, query_pos, feat, scale_embedding):
        resi = x
        x = self.norm(x)
        for block in self.blocks:
            x = block(x, query_pos, feat, scale_embedding)
        x = self.mlp(x)
        x = resi + x
        return x


# Gaussian Transformer: self-attention over the learned Gaussian tokens.
class GSSelfAttn(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_gs_seed_sqrt=12):
        super(GSSelfAttn, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_gs_seed_sqrt = num_gs_seed_sqrt

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.proj = nn.Linear(dim, dim)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.num_gs_seed_sqrt - 1) * (2 * self.num_gs_seed_sqrt - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.num_gs_seed_sqrt)
        coords_w = torch.arange(self.num_gs_seed_sqrt)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.num_gs_seed_sqrt - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.num_gs_seed_sqrt - 1
        relative_coords[:, :, 0] *= 2 * self.num_gs_seed_sqrt - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer('relative_position_index', relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)

        self.qhead = nn.Linear(dim, dim, bias=True)
        self.khead = nn.Linear(dim, dim, bias=True)
        self.vhead = nn.Linear(dim, dim, bias=True)

    def forward(self, gs):
        # gs shape: b*h_count*w_count, num_gs, c
        # pos shape: b*h_count*w_count, num_gs, c
        b_, num_gs, c = gs.shape

        q = self.qhead(gs)
        q = q.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        k = self.khead(gs)
        k = k.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        k = k.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        v = self.vhead(gs)
        v = v.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        v = v.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # b_, num_heads, num_gs, n

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.num_gs_seed_sqrt * self.num_gs_seed_sqrt, self.num_gs_seed_sqrt * self.num_gs_seed_sqrt,
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)

        attn = (attn @ v).transpose(1, 2).reshape(b_, num_gs, c)
        attn = self.proj(attn)

        return attn


class GSSelfAttnLayer(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_gs_seed_sqrt=12, shift_size=0):
        super(GSSelfAttnLayer, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.gs_self_attn = GSSelfAttn(dim=dim, num_heads=num_heads, num_gs_seed_sqrt=num_gs_seed_sqrt)

        self.mlp_selfattn = MLP(in_features=dim, hidden_features=dim, out_features=dim)

        self.num_gs_seed_sqrt = num_gs_seed_sqrt
        self.shift_size = shift_size

        self.gs_cross_attn_scale = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.mlp_crossattn = MLP(in_features=dim, hidden_features=dim, out_features=dim)

    def forward(self, gs, pos, h_count, w_count, scale_embedding):
        # gs shape:b*h_count*w_count, num_gs_seed, channel
        # pos shape: b*h_count*w_count, num_gs_seed, channel
        # scale_embedding shape: b*h_count*w_count, 1, channel

        # gs cross attn with scale_embedding
        resi = gs
        gs = self.norm3(gs)
        gs, _ = self.gs_cross_attn_scale(with_pos_embed(gs, pos), scale_embedding, scale_embedding)
        gs = gs + resi

        # FFN
        resi = gs
        gs = self.norm4(gs)
        gs = self.mlp_crossattn(gs)
        gs = gs + resi

        resi = gs
        gs = self.norm1(gs)

        #### shift gs
        if self.shift_size > 0:
            shift_gs = rearrange(gs, '(b m n) (h w) c -> b (m h) (n w) c', m=h_count, n=w_count,
                                 h=self.num_gs_seed_sqrt, w=self.num_gs_seed_sqrt)
            shift_gs = torch.roll(shift_gs, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shift_gs = rearrange(shift_gs, 'b (m h) (n w) c -> (b m n) (h w) c', m=h_count, n=w_count,
                                 h=self.num_gs_seed_sqrt, w=self.num_gs_seed_sqrt)
        else:
            shift_gs = gs

        #### gs self attention
        gs = self.gs_self_attn(shift_gs)

        #### shift gs back
        if self.shift_size > 0:
            shift_gs = rearrange(gs, '(b m n) (h w) c -> b (m h) (n w) c', m=h_count, n=w_count,
                                 h=self.num_gs_seed_sqrt, w=self.num_gs_seed_sqrt)
            shift_gs = torch.roll(shift_gs, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            shift_gs = rearrange(shift_gs, 'b (m h) (n w) c -> (b m n) (h w) c', m=h_count, n=w_count,
                                 h=self.num_gs_seed_sqrt, w=self.num_gs_seed_sqrt)
        else:
            shift_gs = gs

        gs = shift_gs + resi

        # FFN
        resi = gs
        gs = self.norm2(gs)
        gs = self.mlp_selfattn(gs)
        gs = gs + resi
        return gs


class GSSelfAttnBlock(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_selfattn_layers=4, num_gs_seed_sqrt=12):
        super(GSSelfAttnBlock, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            GSSelfAttnLayer(
                dim=dim,
                num_heads=num_heads,
                num_gs_seed_sqrt=num_gs_seed_sqrt,
                shift_size=0 if i % 2 == 0 else num_gs_seed_sqrt // 2
            ) for i in range(num_selfattn_layers)
        ])

    def forward(self, gs, pos, h_count, w_count, scale_embedding):
        resi = gs
        gs = self.norm(gs)
        for block in self.blocks:
            gs = block(gs, pos, h_count, w_count, scale_embedding)
        gs = self.mlp(gs)
        gs = gs + resi
        return gs

# Structure Prior Modulation Fusion (SPMF).
class StructurePriorModulationFusion(nn.Module):
    """Composite fusion: FiLM channel modulation, spatial gating, and residual connection."""
    def __init__(self, ch, mid=None):
        super().__init__()
        mid = mid or max(64, ch//2)
        # channel (FiLM) from guide
        self.chan = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch*2, 1)  # gamma and beta
        )
        # spatial gate
        self.spa = nn.Sequential(
            nn.Conv2d(ch*2, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
            nn.Sigmoid()
        )
        # refinement
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1)
        )
        # init gamma ~1, beta ~0
        # we'll fix by adding init in forward if desired
    def forward(self, x, guide):
        B,C,H,W = x.shape
        # channel modulators
        gb = self.chan(guide)  # (B,2C,1,1)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        gamma = gamma.view(B,C,1,1) + 1.0  # bias to near 1
        beta = beta.view(B,C,1,1)
        x_f = gamma * x + beta
        # spatial gate
        spa_map = self.spa(torch.cat([x_f, guide], dim=1))  # (B,1,H,W)
        fused = spa_map * x_f + (1-spa_map) * guide
        out = self.refine(fused) + x  # residual
        return out, guide

@ARCH_REGISTRY.register()
class AnatomyGuidedFeatureExtractor(nn.Module):
    def __init__(self, inchannel=64, channel=180, num_heads=6, num_crossattn_blocks=1, num_crossattn_layers=2,
                 num_selfattn_blocks=6, num_selfattn_layers=6,
                 num_gs_seed=144, num_colors=1, gs_up_factor=1.0, window_size=12, img_range=1.0, shuffle_scale1=2,
                 shuffle_scale2=2, use_checkpoint=False):
        super(AnatomyGuidedFeatureExtractor, self).__init__()
        self.channel = channel
        self.nhead = num_heads
        self.gs_up_factor = gs_up_factor
        self.num_gs_seed = num_gs_seed
        self.num_colors = num_colors
        self.window_size = window_size
        self.img_range = img_range
        self.use_checkpoint = use_checkpoint

        self.num_gs_seed_sqrt = int(math.sqrt(num_gs_seed))
        self.gs_up_factor_sqrt = int(math.sqrt(gs_up_factor))
        # self.mri_align = AnatomicalConstraintAlignment(channel)
        self.mri_align = StructurePriorModulationFusion(channel)

        # Align frequency-magnitude statistics with the spatial features.
        self.freq_align = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, 1, 0)
        )


        self.shuffle_scale1 = shuffle_scale1
        self.shuffle_scale2 = shuffle_scale2

        # shared gaussian embedding and its pos embedding
        self.gs_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel), requires_grad=True)
        self.pos_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel), requires_grad=True)

        self.img_feat_proj = nn.Sequential(
            nn.Conv2d(inchannel, channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )

        self.img_feat_proj_guide = nn.Sequential(
            nn.Conv2d(inchannel, channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )
        self.guide_out = nn.Sequential(
            nn.Conv2d(channel, channel//2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel//2, 1, 3, 1, 1)
        )

        self.window_crossattn_blocks = nn.ModuleList([
            WindowCrossAttnBlock(dim=channel,
                                 window_size=window_size,
                                 num_heads=num_heads,
                                 num_layers=num_crossattn_layers,
                                 num_gs_seed=num_gs_seed) for i in range(num_crossattn_blocks)
        ])

        self.gs_selfattn_blocks = nn.ModuleList([
            GSSelfAttnBlock(dim=channel,
                            num_heads=num_heads,
                            num_selfattn_layers=num_selfattn_layers,
                            num_gs_seed_sqrt=self.num_gs_seed_sqrt
                            ) for i in range(num_selfattn_blocks)
        ])

        self.scale_mlp = nn.Sequential(
            nn.Linear(inchannel * 2, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, channel)
        )

        # self.UPNet = nn.Sequential(
        #     nn.Conv2d(channel, channel * self.shuffle_scale1 * self.shuffle_scale1, 3, 1, 1),
        #     nn.PixelShuffle(self.shuffle_scale1),
        #     nn.Conv2d(channel, channel * self.shuffle_scale2 * self.shuffle_scale2, 3, 1, 1),
        #     nn.PixelShuffle(self.shuffle_scale2)
        # )

    def forward(self, x):
        '''
        using deformable detr decoder for cross attention
        Args:
            query: (batch_size, num_query, dim)
            query_pos: (batch_size, num_query, dim)
            srcs: (batch_size, dim, h1, w1)
        '''
        srcs = x[0]
        guide = x[1]
        scale = x[2]

        b, c, h, w = srcs.shape  ###srcs is pad to the size that could be divided by window_size
        query = self.gs_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (w // self.window_size),
                                                                   1, 1)  # b, h_count*w_count, num_gs_seed, channel
        query = query.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                              self.channel)  # b*h_count*w_count, num_gs_seed, channel

        # scale = 1 / scale
        # scale = scale.unsqueeze(1)  # b*1
        scale_embedding = self.scale_mlp(scale)  # b*channel
        scale_embedding = scale_embedding.unsqueeze(1).unsqueeze(2).repeat(1, (h // self.window_size) * (
                w // self.window_size), self.num_gs_seed, 1)  # b, h_count*w_count, num_gs_seed, channel
        scale_embedding = scale_embedding.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                                                  self.channel)  # b*h_count*w_count, num_gs_seed, channel

        query_pos = self.pos_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (
                w // self.window_size), 1, 1)  # b, h_count*w_count, num_gs_seed, channel

        feat = self.img_feat_proj(srcs)  # b*channel*h*w
        feat_guide = self.img_feat_proj_guide(guide)
        # SPMF enhances target features using anatomy priors from the reference image.
        feat, feat_guide = self.mri_align(feat, feat_guide)
        # === Frequency branch: log1p(|FFT|) followed by lightweight projection alignment. ===
        feat_freq       = self.freq_align(fft_mag_log1p(feat))          # [B,C,H,W]
        feat_guide_freq = self.freq_align(fft_mag_log1p(feat_guide))    # [B,C,H,W]


        query_pos = query_pos.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                                      self.channel)  # b*h_count*w_count, num_gs_seed, channel
        # AG-DDCA jointly refines spatial and frequency-domain features.
        for block in self.window_crossattn_blocks:
            query = block(query, query_pos,
                            [feat, feat_guide, feat_freq, feat_guide_freq],
                            scale_embedding)

        resi = query
        # Gaussian Transformer refines the Gaussian-token representation.
        for block in self.gs_selfattn_blocks:
            query = block(query, query_pos, h // self.window_size, w // self.window_size, scale_embedding)
        query = query + resi

        query = rearrange(query, '(b m n) (h w) c -> b c (m h) (n w)', m=h // self.window_size, n=w // self.window_size,
                          h=self.num_gs_seed_sqrt)
        ### We make the number of GS = 16*LR_size in the following, through an simple upsample operation
        # query = self.UPNet(query)
        query = query.permute(0, 2, 3, 1)

        return query, self.guide_out(feat_guide)