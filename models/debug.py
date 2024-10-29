import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class TaylorAttention(nn.Module):
    def __init__(self, dim, num_heads=8, order=1):
        super(TaylorAttention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.order = order
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # 线性变换并拆分 Q、K、V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, num_heads, N, N]

        # 泰勒展开近似
        taylor_expansion = torch.ones_like(attn)
        x_power = attn.clone()
        factorial = 1
        for i in range(1, self.order + 1):
            if i > 1:
                factorial *= i
            taylor_expansion += x_power / factorial
            x_power = x_power * attn  # 下一个次幂

        # ReLU 确保非负性
        taylor_expansion = F.relu(taylor_expansion)

        # 归一化
        attn_weights = taylor_expansion / taylor_expansion.sum(dim=-1, keepdim=True)
        x = torch.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class OriginalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(OriginalAttention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # 线性变换并拆分 Q、K、V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, num_heads, N, N]
        attn_weights = F.softmax(attn, dim=-1)
        x = torch.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

# 测试运行
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 128, 512)  # [batch_size, seq_len, dim]

    # 原始注意力机制
    original_attn = OriginalAttention(dim=512, num_heads=8)
    start_time = time.time()
    output_original = original_attn(x)
    original_time = time.time() - start_time
    print(f"Original Attention Output Shape: {output_original.shape}, Time: {original_time:.6f} seconds")

    # 1～3 阶泰勒展开的注意力机制
    for order in range(1, 4):
        taylor_attn = TaylorAttention(dim=512, num_heads=8, order=order)
        start_time = time.time()
        output_taylor = taylor_attn(x)
        taylor_time = time.time() - start_time
        print(f"Order {order} Taylor Attention Output Shape: {output_taylor.shape}, Time: {taylor_time:.6f} seconds")
