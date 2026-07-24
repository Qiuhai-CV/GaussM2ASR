"""GaussM2ASR feature network: CSMF Encoder -> SPMF -> AG-DDCA -> Gaussian Transformer."""
from basicsr.archs.csmf_encoder_arch import CSMFEncoder
from basicsr.archs.anatomy_guided_feature_extractor_arch import AnatomyGuidedFeatureExtractor
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn
import torch

@ARCH_REGISTRY.register()
class GaussM2ASRNet(nn.Module):
    def __init__(self,num_in_ch=1,num_feat=64,num_block=16,res_scale=1.0,
                 channel=180, num_heads=6, num_crossattn_blocks=2, num_crossattn_layers=2,
                 num_selfattn_blocks=5, num_selfattn_layers=5,
                 num_gs_seed=64, gs_up_factor=1.0, window_size=8, img_range=1.0, shuffle_scale1=2,
                 shuffle_scale2=2):
        super(GaussM2ASRNet, self).__init__()

        # Keep the historical attribute name for checkpoint key compatibility; it implements the CSMF encoder.
        self.EDSR = CSMFEncoder(num_in_ch=num_in_ch,num_feat=num_feat,num_block=num_block,res_scale=res_scale)
        self.fea2gs = AnatomyGuidedFeatureExtractor(inchannel=num_feat,channel=channel,num_heads=num_heads,num_crossattn_blocks=num_crossattn_blocks,
                             num_crossattn_layers=num_crossattn_layers,num_selfattn_blocks=num_selfattn_blocks,
                            num_selfattn_layers=num_selfattn_layers,num_gs_seed=num_gs_seed,gs_up_factor=gs_up_factor,
                            window_size=window_size,img_range=img_range,shuffle_scale1=shuffle_scale1,
                            shuffle_scale2=shuffle_scale2)


    def forward(self, x, guide, scale):
    # def forward(self, x):
    #     guide=x
    #     scale=torch.tensor([float(4), float(4)]).to('cuda:1')
        fea = self.EDSR(x, guide, scale)
        fea, guide = self.fea2gs(fea)
        return fea, guide

# if __name__ == '__main__':
#     model = GaussM2ASRNet()
#     from calflops import calculate_flops
#     import time
#     model.eval().to('cuda:1')  # 设置为评估模式
#
#     # 定义输入尺寸
#     input_shape = (1, 1, 256, 256)
#     a=time.time()
#     for i in range(100):
#         out = model(torch.randn(1, 1, 256, 256).to('cuda:1'))
#     print((time.time()-a)/100)
#
#     # # 计算 FLOPs 和参数量
#     # flops, macs, params = calculate_flops(model=model,
#     #                                       input_shape=input_shape,
#     #                                       output_as_string=True,
#     #                                       output_precision=4)
#
#     # print("FLOPs:", flops)
#     # print("MACs:", macs)
#     # print("Parameters:", params)