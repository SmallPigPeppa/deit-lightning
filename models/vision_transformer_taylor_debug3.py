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

from models.vision_transformerV3 import Attention as OriginalAttention
from models.taylor_transferV2 import TaylorAttention

# 测试运行
if __name__ == "__main__":
    torch.manual_seed(0)
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="cpu"
    device="cuda"
    x = torch.randn(64, 128, 512).to(device)  # [batch_size, seq_len, dim]

    # 原始注意力机制
    original_attn = OriginalAttention(dim=512, num_heads=8).to(device)
    start_time = time.time()
    output_original = original_attn(x)
    original_time = time.time() - start_time
    print(f"Original Attention Output Shape: {output_original.shape}, Time: {original_time:.6f} seconds")

    # 1～3 阶泰勒展开的注意力机制
    # for order in range(1, 5):
    #     taylor_attn = TaylorAttention(dim=512, num_heads=8, order=order).to(device)
    #     start_time = time.time()
    #     output_taylor = taylor_attn(x)
    #     taylor_time = time.time() - start_time
    #     print(f"Order {order} Taylor Attention Output Shape: {output_taylor.shape}, Time: {taylor_time:.6f} seconds")

    order=5
    taylor_attn = TaylorAttention(dim=512, num_heads=8, order=order).to(device)
    taylor_attn = OriginalAttention(dim=512, num_heads=8).to(device)
    start_time = time.time()
    output_taylor = taylor_attn(x)
    taylor_time = time.time() - start_time
    print(f"Order {order} Taylor Attention Output Shape: {output_taylor.shape}, Time: {taylor_time:.6f} seconds")