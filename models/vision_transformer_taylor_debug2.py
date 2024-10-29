import copy
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['VisionTransformer']  # model_registry will add each entrypoint fn to this

_logger = logging.getLogger(__name__)


def convert_attention_to_taylor(attention_module: nn.Module, order: int = 1) -> nn.Module:
    """
    Convert an OriginalAttention module to a TaylorAttention module with the specified order.

    Args:
        attention_module (nn.Module): The original attention module to convert.
        order (int): The order of the Taylor series expansion.

    Returns:
        nn.Module: The converted TaylorAttention module with copied weights.
    """
    if not isinstance(attention_module, OriginalAttention):
        raise TypeError("Input module must be an instance of OriginalAttention")

    taylor_attention = TaylorAttention(
        dim=attention_module.qkv.in_features,
        num_heads=attention_module.num_heads,
        order=order,
        qkv_bias=attention_module.qkv.bias is not None,
        qk_norm=isinstance(attention_module.q_norm, nn.LayerNorm),
        attn_drop=attention_module.attn_drop.p,
        proj_drop=attention_module.proj_drop.p,
        norm_layer=type(attention_module.q_norm) if isinstance(attention_module.q_norm, nn.LayerNorm) else nn.Identity
    )

    # Copy weights from the original attention module to the Taylor attention module
    taylor_attention.qkv.weight.data = attention_module.qkv.weight.data.clone()
    if attention_module.qkv.bias is not None:
        taylor_attention.qkv.bias.data = attention_module.qkv.bias.data.clone()
    taylor_attention.proj.weight.data = attention_module.proj.weight.data.clone()
    taylor_attention.proj.bias.data = attention_module.proj.bias.data.clone()

    return taylor_attention


class TaylorAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            order: int = 1,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.order = order
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        qk_matmul = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.ones_like(qk_matmul)  # 泰勒展开初始值为 1
        x_power = qk_matmul.clone()

        for i in range(1, self.order + 1):
            attn = attn + x_power / math.factorial(i)
            if i < self.order:  # 避免多计算一次下一个次幂
                x_power = x_power * qk_matmul  # 下一个次幂

        attn = F.relu(attn)  # ReLU 确保非负性
        attn = attn / attn.sum(dim=-1, keepdim=True)  # 归一化
        # attn = self.attn_drop(attn)
        x = torch.matmul(attn, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


class OriginalAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


# 测试运行
if __name__ == "__main__":
    torch.manual_seed(0)
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    device="cuda"
    x = torch.randn(64, 128, 512).to(device)  # [batch_size, seq_len, dim]

    # 原始注意力机制
    original_attn = OriginalAttention(dim=512, num_heads=8).to(device)
    start_time = time.time()
    output_original = original_attn(x)
    original_time = time.time() - start_time
    print(f"Original Attention Output Shape: {output_original.shape}, Time: {original_time:.6f} seconds")

    # 将原始注意力机制转换为泰勒注意力机制
    taylor_attn_converted = convert_attention_to_taylor(original_attn, order=5).to(device)
    start_time = time.time()
    output_taylor_converted = taylor_attn_converted(x)
    taylor_converted_time = time.time() - start_time
    print(
        f"Converted Taylor Attention Output Shape: {output_taylor_converted.shape}, Time: {taylor_converted_time:.6f} seconds")
