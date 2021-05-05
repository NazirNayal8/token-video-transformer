import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LateTemporalTokenizer(nn.Module):
    """
    Late Temporal Tokenizer is a 3D feature map tokenizer modules. It creates spatio-temporal tokens that summarize
    information along the spatial and temporal dimensions of the input feature map.
    Params:
    - in_channels: channel size of input feature map
    - spatial_dim: spatial dimension size of tokens
    - temporal_dim: temporal dimension size of tokens
    - token_dim: dimension of each spatio-temporal token
    """

    def __init__(self, in_channels: int, spatial_dim: int, temporal_dim: int, token_dim: int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.token_dim = token_dim

        self.spatial_proj = nn.Linear(in_channels, token_dim)
        self.temporal_proj = nn.Linear(token_dim, token_dim)

        self.spatial_linear = nn.Linear(in_channels, spatial_dim)
        self.temporal_linear = nn.Linear(token_dim, temporal_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Expected Input Dimensions: (B, T, HW, C), where:
        - B: batch size
        - T: number of video frames
        - HW: number of pixels
        - C: size of input feature map channels

        Expected Output Dimensions: (B, F, L, D), where:
        - L: spatial dimension of tokens
        - F: temporal dimension of tokens
        - D: size of token dimension
        """

        # spatial tokenization
        x_proj = self.spatial_proj(x)  # -> (B, T, HW, D)
        a = self.spatial_linear(x)  # -> (B, T, HW, L)
        a = F.softmax(a, dim=2)  # -> (B, T, HW, L)
        a = torch.einsum('btij, btik -> btjk', a, x_proj)  # -> (B, T, L, D)

        # temporal tokenization
        a_proj = self.temporal_proj(a)  # -> (B, T, L, D)
        t = self.temporal_linear(a)  # -> (B, T, L, F)
        t = F.softmax(t, dim=1)  # -> (B, T, L, F)
        t = torch.einsum('btlf, btld -> bfld', t, a_proj)  # -> (B, F, L, D)

        return t
