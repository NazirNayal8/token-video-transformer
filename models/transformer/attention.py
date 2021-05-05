""""
Adapted from: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import torch.nn as nn


class MultiheadAttention(nn.Module):
    """
    Multihead Attention module computes attention for num_heads number of parallel independent heads.
    Params:
    - dim: dimension of input tokens
    - num_heads: number of parallel heads
    - spatial: if True, computes attention along the spatial dimension, else, along the temporal dimension
    - qkv_bias: if true, include bias term in query-key-value projection
    - qk_scale: custom scaling factor for scaled dot product. If None, sqrt(head_dim) is used. head_dim is
        dim divided by num_heads
    - attn_drop: dropout probability for dropout layer after scaled dot product
    - proj_drop: dropout probability for dropout layer after final projection
    """
    def __init__(self, token_dim: int, num_heads: int = 8, spatial: bool = True, qkv_bias: bool = False, qk_scale=None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.spatial = spatial
        self.head_dim = token_dim // num_heads

        assert token_dim % num_heads == 0, "token_dim must be divisible by num_heads"

        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(token_dim, token_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(token_dim, token_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Expected Input Dimension: (B, F, L, D), where:
        - B: batch size
        - F: temporal dimension of token
        - N: spatial dimension of tokens
        - D: size of token_dimension

        Important Note: If attention is computed along the temporal dimension, then F and L will be swapped. That is,
        F will be considered as spatial dimension, and L will be considered as temporal dimension.

        Expected Output Dimensions: (B, F, L, D)
        """

        x_in = x  # -> (B, F, L, D)
        if not self.spatial:
            x_in = x_in.permute(0, 2, 1, 3)  # -> (B, L, F, D)

        B, F, L, D = x_in.shape

        qkv = self.qkv(x_in)  # -> (B, F, L, 3 * D) 3 for query, key, value
        qkv = qkv.reshape(B, F, L, 3, self.num_heads, self.head_dim)  # -> (B, F, L, 3, num_heads, head_dim)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # -> (3, B, F, num_heads, L, head_dim)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]  # -> each is (B, F, num_heads, L, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # -> (B, F, num_heads, L, L)
        attn = attn.softmax(dim=-1)  # -> (B, F, num_heads, L, L)
        attn = self.attn_drop(attn)  # -> (B, F, num_heads, L, L)

        x_out = (attn @ v)  # -> (B, F, num_heads, L, head_dim)
        x_out = x_out.permute(0, 1, 3, 2, 4)  # -> (B, F, L, num_heads, head_dim)
        x_out = x_out.reshape(B, F, L, D)  # -> (B, F, L, D)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        if not self.spatial:
            x_out = x_out.permute(0, 2, 1, 3)

        return x_out
