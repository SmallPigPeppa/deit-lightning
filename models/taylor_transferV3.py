import math

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from timm.layers import use_fused_attn
from models.vision_transformerV3 import vit_small_patch16_224
from models.vision_transformerV3 import Attention
import copy


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
        # qk_matmul = torch.matmul(q, k.transpose(-2, -1))
        # attn = torch.ones_like(qk_matmul)  # 泰勒展开初始值为 1
        attn = q @ k.transpose(-2, -1)
        attn = torch.exp(attn)
        # attn = 1 + attn * 0.5
        # x_power = qk_matmul.clone()

        # for i in range(1, self.order + 1):
        #     # attn = attn + x_power / math.factorial(i)
        #     attn = attn + x_power * 0.5
        #     if i < self.order:  # 避免多计算一次下一个次幂
        #         x_power = x_power * qk_matmul  # 下一个次幂

        # attn = F.relu(attn)  # ReLU 确保非负性
        attn = attn / attn.sum(dim=-1, keepdim=True)  # 归一化
        # attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


def convert_attention_to_taylor(attention_module: nn.Module, order: int = 1) -> nn.Module:
    """
    Convert an OriginalAttention module to a TaylorAttention module with the specified order.

    Args:
        attention_module (nn.Module): The original attention module to convert.
        order (int): The order of the Taylor series expansion.

    Returns:
        nn.Module: The converted TaylorAttention module with copied weights.
    """
    if not isinstance(attention_module, Attention):
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


def replace_attention_with_taylor_by_index(model, target_layer_idx, order: int = 1) -> nn.Module:
    """
    Replace the Attention layer at the specified index or indices in the Vision Transformer model with a TaylorAttention layer.

    Args:
        model (nn.Module): The Vision Transformer model.
        target_layer_idx (int or list of int): The index or indices of the layer(s) to replace.
        order (int): The order of the Taylor series expansion for the TaylorAttention.

    Returns:
        nn.Module: The modified Vision Transformer model.
    """
    model = copy.deepcopy(model)

    if isinstance(target_layer_idx, int):
        target_layer_idx = [target_layer_idx]

    for i, block in enumerate(model.blocks):
        if hasattr(block, 'attn') and isinstance(block.attn, Attention):
            if i in target_layer_idx:
                # Replace the Attention layer with TaylorAttention
                block.attn = convert_attention_to_taylor(block.attn, order=order)

    return model


def compare_model_forward_speed(model1: nn.Module, model2: nn.Module, input_tensor: torch.Tensor,
                                num_iterations: int = 500) -> None:
    """
    Compare the forward computation speed of two models.

    Args:
        model1 (nn.Module): The first model to compare.
        model2 (nn.Module): The second model to compare.
        input_tensor (torch.Tensor): The input tensor for the models.
        num_iterations (int): The number of iterations to run for comparison.
    """
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = model1.to(device)
    model2 = model2.to(device)
    input_tensor = input_tensor.to(device)

    # Measure model1 forward speed
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model1(input_tensor)
    model1_time = time.time() - start_time

    # Measure model2 forward speed
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model2(input_tensor)
    model2_time = time.time() - start_time

    print(f"Model 1 average forward time: {model1_time / num_iterations:.6f} seconds")
    print(f"Model 2 average forward time: {model2_time / num_iterations:.6f} seconds")


if __name__ == "__main__":
    # Example usage:
    vit_model = vit_small_patch16_224(pretrained=True)  # Assuming you have the Vision Transformer model instance
    # num_attention_layers = count_attention_layers(vit_model)
    # print(f"Number of Attention layers: {num_attention_layers}")
    indices_to_replace = list(range(0, 12))  # Example indices
    order = 1  # Taylor expansion order
    vit_model_taylor = replace_attention_with_taylor_by_index(vit_model, indices_to_replace, order)
    # print(vit_model)
    # print(vit_model_taylor)

    # Compare the forward speed of the original and modified models
    input_tensor = torch.randn(256, 3, 224, 224)  # Example input tensor
    compare_model_forward_speed(vit_model_taylor, vit_model, input_tensor)

# if __name__ == "__main__":
#     torch.manual_seed(0)
#     import time
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = "cpu"
#     # device="cuda"
#     x = torch.randn(64, 128, 512).to(device)  # [batch_size, seq_len, dim]
#
#     # 原始注意力机制
#     original_attn = Attention(dim=512, num_heads=8).to(device)
#     start_time = time.time()
#     output_original = original_attn(x)
#     original_time = time.time() - start_time
#     print(f"Original Attention Output Shape: {output_original.shape}, Time: {original_time:.6f} seconds")
#
#     # 将原始注意力机制转换为泰勒注意力机制
#     taylor_attn_converted = convert_attention_to_taylor(original_attn, order=1).to(device)
#     start_time = time.time()
#     output_taylor_converted = taylor_attn_converted(x)
#     taylor_converted_time = time.time() - start_time
#     print(
#         f"Converted Taylor Attention Output Shape: {output_taylor_converted.shape}, Time: {taylor_converted_time:.6f} seconds")
