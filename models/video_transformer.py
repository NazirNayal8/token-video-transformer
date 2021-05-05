
import torch.nn as nn
from torch import Tensor
from .transformer.layer import DividedTransformer
from .tokenizer.late_temporal_tokenizer import LateTemporalTokenizer
from .backbone.resnet import *


def get_backbone(backbone_type: str, pretrained: bool):

    if backbone_type == 'resnet18':
        return resnet18(pretrained=pretrained, custom_class_num=False)
    if backbone_type == 'resnet34':
        return resnet34(pretrained=pretrained, custom_class_num=False)
    if backbone_type == 'resnet50':
        return resnet50(pretrained=pretrained, custom_class_num=False)
    if backbone_type == 'resnet101':
        return resnet101(pretrained=pretrained, custom_class_num=False)
    if backbone_type == 'resnet52':
        return resnet152(pretrained=pretrained, custom_class_num=False)

    raise ValueError(f"Unregistered Backbone type: {backbone_type}")


def get_tokenizer(tokenizer_type: str, in_channels: int, spatial_dim: int, temporal_dim: int, token_dim: int):

    if tokenizer_type == 'late_temporal':
        return LateTemporalTokenizer(in_channels, spatial_dim, temporal_dim, token_dim)

    raise ValueError(f"Unregistered Tokenizer type: {tokenizer_type}")


class DividedVideoTransformer(nn.Module):

    def __init__(
            self,
            spatial_dim: int,
            temporal_dim: int,
            token_dim: int,
            tokenizer_type: str = 'late_temporal',
            backbone_type: str = 'resnet18',
            pretrained_backbone: bool = False,
            num_classes: int = 10,
            transformer_layers=None,
            num_heads: int = 8,
            feedforward_dim: int = 2048,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super().__init__()

        if transformer_layers is None:
            transformer_layers = [0, 1]

        assert isinstance(transformer_layers, list)

        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.token_dim = token_dim
        self.backbone_type = backbone_type
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        self.activation = activation

        self.backbone = get_backbone(backbone_type, pretrained_backbone)
        self.in_channels = self.backbone.backbone_channel_output
        self.tokenizer = get_tokenizer(tokenizer_type, self.in_channels, spatial_dim, temporal_dim, token_dim)

        self.transformer = DividedTransformer(token_dim, transformer_layers, num_heads, feedforward_dim, dropout,
                                              activation)

        self.classifier = nn.Linear(spatial_dim * temporal_dim * token_dim, num_classes)

    def forward(self, x: Tensor):
        """
        Expected Input Dimensions: (B, T, 3, H_in, W_in), where:
        - B: batch size
        - T: number of frames
        - H_in: image height
        - W_in: image width
        - 3: image channel size
        """
        B, T, C, H_in, W_in = x.shape
        x = x.reshape(B * T, C, H_in, W_in)  # -> (B * T, 3, H_in, W_in)
        x = self.backbone(x)  # -> (B * T, C, H, W)

        x = x.permute(0, 2, 3, 1)  # -> (B * T, H, W, C)
        x = x.reshape(B, T, -1, self.in_channels)  # -> (B, T, HW, C)

        t = self.tokenizer(x)  # -> (B, F, L, D)
        t = self.transformer(t)  # -> (B, F, L, D)

        x_out = t.flatten(start_dim=1)  # -> (B, F * L * D)
        x_out = self.classifier(x_out)  # -> (B, num_classes)

        return x_out
