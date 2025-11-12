# -*- coding: utf-8 -*-
# @Time    : 2025/4/8 17:10
# @Author  : CFuShn
# @Comments: 
# @Software: PyCharm


import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DitBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super(DitBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                       act_layer=approx_gelu)
        self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(6, dim=1)
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa)) * gate_msa.unsqueeze(1)
        x = x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)) * gate_mlp.unsqueeze(1)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super(FinalLayer, self).__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.linear(modulate(self.norm_final(x), shift, scale))
        return x


class SimpleDiT(nn.Module):
    def __init__(self):
        super(SimpleDiT, self).__init__()
        self.block = DitBlock(hidden_size=hidden_size, num_heads=num_heads)
        self.final = FinalLayer(hidden_size=hidden_size, patch_size=patch_size,
                                out_channels=out_channels)

    def forward(self, x, c):
        x = self.block(x, c)
        x = self.final(x, c)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    hidden_size = 128
    num_heads = 4
    patch_size = 2
    out_channels = 3
    seq_len = 16  # 假设有 16 个 patch（例如 4x4 的 patch grid）
    batch_size = 2

    # 创建模型
    model = SimpleDiT()

    # 打印模型结构
    summary(model, (
        (batch_size, seq_len, hidden_size),  # x
        (batch_size, hidden_size)))  # c
