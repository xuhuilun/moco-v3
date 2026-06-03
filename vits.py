# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入数学库，用于初始化中的 sqrt 等计算
import math
# 导入 PyTorch 核心库
import torch
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 从 functools 导入 partial（用于固定函数参数）和 reduce（累积运算）
from functools import partial, reduce
# 从 operator 导入 mul 乘法操作符，配合 reduce 计算乘积
from operator import mul

# 从 timm 库导入 VisionTransformer 基类和 _cfg 默认配置函数
from timm.models.vision_transformer import VisionTransformer, _cfg
# 从 timm 导入将标量或元组转为 2-tuple 的辅助函数
from timm.models.layers.helpers import to_2tuple
# 从 timm 导入 PatchEmbed 类（将图像切片为 patch 并做线性投影）
from timm.models.layers import PatchEmbed

# 定义模块对外暴露的接口：四种模型变体
__all__ = [
    "vit_small",       # ViT-Small
    "vit_base",        # ViT-Base
    "vit_conv_small",  # ViT-Small + ConvStem（用卷积堆叠替代 patch embedding）
    "vit_conv_base",   # ViT-Base + ConvStem
]


class VisionTransformerMoCo(VisionTransformer):
    """MoCo v3 专用的 VisionTransformer，继承自 timm 的 VisionTransformer。

    主要改动：
    1. 使用固定的 2D sin-cos 位置编码（不可学习）
    2. 自定义权重初始化（QKV 独立处理 + Xavier uniform）
    3. 支持 stop_grad_conv1：冻结 patch embedding 层，防止 ViT 自监督训练早期坍塌
    """

    def __init__(self, stop_grad_conv1=False, **kwargs):
        # 调用父类 VisionTransformer 的初始化，完成标准 ViT 的搭建
        super().__init__(**kwargs)

        # ⭐ MoCo v3 实践：用固定的 2D sin-cos 位置编码替代可学习位置编码
        # 这样做的好处：(1) 外推能力强，可处理不同分辨率 (2) 减少可学习参数量
        self.build_2d_sincos_position_embedding()

        # 遍历所有子模块，进行自定义权重初始化
        for name, m in self.named_modules():
            # 只对 Linear 层做初始化
            if isinstance(m, nn.Linear):
                # QKV 是 attention 中融合的 query/key/value 投影矩阵
                if "qkv" in name:
                    # QKV 权重形状为 [3*d_model, d_model]，每 d_model 对应 Q、K、V 一组
                    # 用均匀分布 U(-val, val) 初始化，val 使用 Xavier 风格计算
                    val = math.sqrt(
                        6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1])
                    )
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    # 非 QKV 的 Linear 层使用标准 Xavier uniform 初始化
                    nn.init.xavier_uniform_(m.weight)
                # 所有 Linear 层的 bias 初始化为 0
                nn.init.zeros_(m.bias)
        # cls token 用很小的标准差初始化（1e-6），使其初始接近零向量
        nn.init.normal_(self.cls_token, std=1e-6)

        # 检查 patch_embed 是否为标准的 PatchEmbed 类型（而非 ConvStem 等自定义类型）
        if isinstance(self.patch_embed, PatchEmbed):
            # 对 patch embedding 卷积层做 Xavier uniform 初始化
            # 输入通道数=3（RGB），卷积核大小=patch_size（如 16x16）
            # 使用 reduce(mul, patch_size, 1) 计算卷积核的元素总数
            val = math.sqrt(
                6.0
                / float(
                    3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim
                )
            )
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            # patch embedding 卷积的 bias 初始化为 0
            nn.init.zeros_(self.patch_embed.proj.bias)

            # ⭐⭐ MoCo v3 核心创新之一：冻结 patch embedding 层（stop_grad_conv1）
            # 论文 §4.3 发现 ViT 在自监督训练早期会出现训练坍塌，
            # 信息瓶颈出现在 patch projection 层。冻结它可以提供稳定的初始特征提取。
            if stop_grad_conv1:
                # 设置 requires_grad = False，该层参数不参与反向传播
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.0):
        """构建固定的 2D sin-cos 位置编码。

        将 2D 位置分解为行和列两个维度，分别用 sin/cos 编码后拼接。
        temperature 控制频率的衰减速度（默认 10000 来自 Transformer 原文）。

        输出形状: [1, 1+num_patches, embed_dim]
          - 第一个 token 位置是 [cls] 的编码（全零）
          - 后续每个 patch 位置是 2D sin-cos 编码
        """
        # 获取 patch 网格尺寸，例如 224/16=14，即 14x14 的网格
        h, w = self.patch_embed.grid_size
        # 创建 0 到 w-1 的行位置序列
        grid_w = torch.arange(w, dtype=torch.float32)
        # 创建 0 到 h-1 的列位置序列
        grid_h = torch.arange(h, dtype=torch.float32)
        # 生成网格坐标 (grid_w, grid_h)，形状均为 [h, w]
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        # embed_dim 必须能被 4 整除（因为要编码为 4 部分：sin_w, cos_w, sin_h, cos_h）
        assert self.embed_dim % 4 == 0, (
            "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        )
        # 每个方向（行或列）的编码维度 = embed_dim / 4
        pos_dim = self.embed_dim // 4
        # 构造频率序列 omega，范围 [0, 1)，shape: [pos_dim]
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        # 按 temperature 衰减: 1 / (10000^(i/pos_dim))
        omega = 1.0 / (temperature**omega)
        # 计算行方向的位置编码: out_w[m, d] = grid_w[m] * omega[d]
        # out_w shape: [num_patches, pos_dim]
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        # 计算列方向的位置编码: out_h[m, d] = grid_h[m] * omega[d]
        # out_h shape: [num_patches, pos_dim]
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])

        # 拼接四个部分: [sin(out_w), cos(out_w), sin(out_h), cos(out_h)]
        # 最终 shape: [1, num_patches, embed_dim] （加上了 batch 维度）
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        # MoCo v3 中只用了一个 [cls] token，没有 distillation token
        assert self.num_tokens == 1, "Assuming one and only one token, [cls]"
        # [cls] token 的位置编码初始化为零向量
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        # 最终位置编码 = [cls] 编码 + patch 编码，shape: [1, 1+num_patches, embed_dim]
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        # ⭐ 固定位置编码：不参与训练，requires_grad = False
        self.pos_embed.requires_grad = False


class ConvStem(nn.Module):
    """ConvStem：用多层卷积堆叠替代 ViT 的 patch embedding。

    来自论文 "Early Convolutions Help Transformers See Better" (Tete et al., 2021)。
    使用 4 层 stride=2 的 3x3 卷积 + 1 层 1x1 卷积，将 224x224x3 的图像下采样为 14x14xembed_dim。

    设计动机：CNN 的局部归纳偏置让 Transformer 在训练初期能学到更好的低级特征（边缘、纹理），
    在自监督学习（如 MoCo v3）中比直接 Patchify 更稳定。
    ConvStem 版本会减少一层 ViT block（depth=11 而非 12）来保持计算量相当。
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()

        # ConvStem 目前只支持 patch_size=16（输出下采样 16 倍）
        assert patch_size == 16, "ConvStem only supports patch size of 16"
        # embed_dim 必须能被 8 整除，因为 4 层卷积每层输出通道翻倍
        assert embed_dim % 8 == 0, "Embed dimension must be divisible by 8 for ConvStem"

        # 将输入输出尺寸统一转为 2-tuple 形式
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # 下采样后的网格尺寸
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # patch 总数
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # 构建卷积 stem：4 层 stride=2 卷积，每层输出通道翻倍
        stem = []
        # 输入为 RGB 3 通道，第一层输出通道 = embed_dim/8
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            # 每一层都是 Conv2d(k=3, s=2, p=1) -> BN -> ReLU
            stem.append(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,  # 后面接 BN，不需要 bias
                )
            )
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            # 下一层的输入通道 = 当前输出通道
            input_dim = output_dim
            # 输出通道翻倍：embed_dim/8 -> embed_dim/4 -> embed_dim/2 -> embed_dim
            output_dim *= 2
        # 最后接一个 1x1 卷积将通道数映射到 embed_dim（这个 conv 没有 BN+ReLU）
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        # 整个 stem 作为 Sequential
        self.proj = nn.Sequential(*stem)

        # 可选的 LayerNorm（通常不用）
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """ConvStem 前向传播。

        Args:
            x: 输入图像，shape [B, 3, 224, 224]

        Returns:
            shape [B, num_patches, embed_dim] 或 [B, embed_dim, H', W']（未 flatten 时）
        """
        B, C, H, W = x.shape
        # 检查输入尺寸是否匹配
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        # 通过 5 层卷积：4 层 stride=2 + 1 层 1x1，最终 shape [B, embed_dim, 14, 14]
        x = self.proj(x)
        if self.flatten:
            # BCHW -> BNC: flatten spatial dims 再转置
            # shape 变为 [B, 14*14=196, embed_dim]
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # 可选的正则化（通常用 Identity）
        x = self.norm(x)
        return x


def vit_small(**kwargs):
    """创建 ViT-Small 模型。

    架构参数:
        patch_size=16: 每个 patch 16x16 像素
        embed_dim=384: token 特征维度
        depth=12: Transformer block 层数
        num_heads=12: 多头注意力头数（384/12=32 dim per head）
        mlp_ratio=4: MLP 隐藏层维度 = 384*4=1536
        qkv_bias=True: QKV 投影使用 bias
        norm_layer=LayerNorm(eps=1e-6): 使用 LayerNorm
    """
    model = VisionTransformerMoCo(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    # 设置 timm 的默认配置（用于下游任务）
    model.default_cfg = _cfg()
    return model


def vit_base(**kwargs):
    """创建 ViT-Base 模型。

    架构参数:
        patch_size=16: 每个 patch 16x16 像素
        embed_dim=768: token 特征维度
        depth=12: Transformer block 层数
        num_heads=12: 多头注意力头数（768/12=64 dim per head）
        mlp_ratio=4: MLP 隐藏层维度 = 768*4=3072
        qkv_bias=True: QKV 投影使用 bias
        norm_layer=LayerNorm(eps=1e-6): 使用 LayerNorm

    注：MoCo v3 论文中使用 vit_base 在 ImageNet-1K 上达到了 76.7% 的线性评估准确率。
    """
    model = VisionTransformerMoCo(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def vit_conv_small(**kwargs):
    """创建 ViT-Small + ConvStem 模型。

    与 vit_small 的区别：
    - depth=11（减少一层 Transformer block，因为 ConvStem 增加了参数量）
    - embed_layer=ConvStem（用 4 层卷积替代 patch embedding）
    """
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16,
        embed_dim=384,
        depth=11,  # 比标准 vit_small 少 1 层
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        embed_layer=ConvStem,  # 使用 ConvStem 替代标准 PatchEmbed
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def vit_conv_base(**kwargs):
    """创建 ViT-Base + ConvStem 模型。

    与 vit_base 的区别：
    - depth=11（减少一层 Transformer block，因为 ConvStem 增加了参数量）
    - embed_layer=ConvStem（用 4 层卷积替代 patch embedding）
    """
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16,
        embed_dim=768,
        depth=11,  # 比标准 vit_base 少 1 层
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        embed_layer=ConvStem,  # 使用 ConvStem 替代标准 PatchEmbed
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model