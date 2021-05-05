"""
Adapted from : https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
"""

import torch.nn as nn
from torch import Tensor
from .attention import MultiheadAttention


def get_activation(activation):

    if activation == 'relu':
        return nn.ReLU()
    if activation == 'gelu':
        return nn.GELU()
    if activation == 'tanh':
        return nn.Tanh()
    if activation == 'sigmoid':
        return nn.Sigmoid()
    if activation == 'leaky_relu':
        return nn.LeakyReLU()

    raise ValueError(f"Activation type {activation} does not exist.")


class DividedEncoder(nn.Module):
    """
    TransformerEncoder module represents a single spatio-temporal Transformer Encoder Layer. In this version, a single
    transformer encoder layer is either spatial or temporal. That is, it either computes attention on the spatial
    dimension of the input tokens, or the temporal dimension.
    Params:
    - spatial_dim: the spatial dimension of input tokens
    - temporal_dim: the temporal dimension of input tokens
    - spatial: if true, the transformer encoder layer considers spatial dimension as the token dimension, otherwise,
        the temporal dimension is considered as the token dimension.
    - num_heads: number of heads for multi-head attention
    - feedforward_dim: the hidden size of the feed forward layer in the transformer encoder.
    - dropout: probability of dropout in dropout layers.
    - activation: type of non-linear activation function to be used.
    """

    def __init__(
            self,
            token_dim: int,
            spatial: bool = True,
            num_heads: int = 8,
            feedforward_dim: int = 2048,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super().__init__()

        self.spatial = spatial
        self.num_heads = num_heads
        self.token_dim = token_dim

        self.self_attn = MultiheadAttention(token_dim, num_heads, spatial)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(token_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, token_dim)

        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        """
        Expected Input Dimensions: (B, F, L, D), where:
        - B: batch size
        - F: temporal token dimension
        - L: spatial token dimension
        - D: token dimension size

        Expected output Dimensions: (B, F, L, D)
        """

        x_1 = self.self_attn(x)  # -> (B, F, L, D)

        x_1 = x_1 + self.dropout1(x_1)  # -> (B, F, L, D)
        x_1 = self.norm1(x_1)  # -> (B, F, L, D)

        x_2 = self.linear1(x_1)  # -> (B, F, L, feedforward_dim)
        x_2 = self.activation(x_2)  # -> (B, F, L, feedforward_dim)
        x_2 = self.dropout(x_2)  # -> (B, F, L, feedforward_dim)
        x_2 = self.linear2(x_2)  # -> (B, F, L, D)

        x_2 = x_1 + self.dropout2(x_2)  # -> (B, F, L, D)
        x_out = self.norm2(x_2)  # -> (B, F, L, D)
        return x_out
