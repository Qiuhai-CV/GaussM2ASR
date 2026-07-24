"""Cross-Scale Multi-contrast Fusion (CSMF) encoder."""

import torch
from torch import nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2),
                                  nn.Conv2d(n_feat * 2, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2),
                                  nn.Conv2d(n_feat * 2, n_feat, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)

@ARCH_REGISTRY.register()
class CSMFEncoder(nn.Module):
    def __init__(self, num_in_ch, num_feat=64, num_block=16, res_scale=1.0):
        super(CSMFEncoder, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv_first_guide = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.scale_mlp = nn.Sequential(
            nn.Linear(1, num_feat * 4),
            nn.ReLU(),
            nn.Linear(num_feat * 4, num_feat * 2)
        )

        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_after_body_guide = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x, guide, scale):
        x = self.conv_first(x)
        guide = self.conv_first_guide(guide)

        scale = 1 / scale
        scale = scale.unsqueeze(1)  # b*1
        scale = self.scale_mlp(scale)  # [B, num_feat*2]
        gamma, beta = scale.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, num_feat, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)  # [B, num_feat, 1, 1]
        x = x * (1 + gamma) + beta  # 调制x特征

        out = self.body([x, guide])
        x, guide = out[0], out[1]

        res_guide = self.conv_after_body_guide(guide)
        guide = guide + res_guide
        res = self.conv_after_body(x)
        x = res + x

        return [x, guide, scale]