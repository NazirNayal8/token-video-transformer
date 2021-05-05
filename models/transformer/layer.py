import torch.nn as nn
from torch import Tensor
from .encoder import DividedEncoder


class DividedTransformer(nn.Module):
    """
    Divided Transformer module is created to process spatio-temporal tokens. Spatio-temporal tokens are of dimension
    (F, L, D) each, where F represents the temporal dimension, L represents the spatial dimension, and D represents
    the dimension size of every token. The Divided Transformer consists of several encoder layers. Each layer can
    be either temporal or spatial, that is, the attention computations within each layer are either along the spatial
    dimension or the temporal dimension. The number of layers, their type, and their order is determined by the layers
    parameters, which is an ordered list of zeros and ones. A zero value at the ith index of this list means that
    the ith encoder layer must be spatial. Otherwise, the ith layer should be temporal.

    Params:
    - token_dim: size of input token dimension
    - layers: a list of zeros and ones determining the types and order of encoder layers.
    - num_heads: number of attention heads for multi-head attention
    - feedforward_dim: hidden size of the linear layer within every transformer encoder layer.
    - dropout: dropout probability for every transformer encoder layer.
    - activation: type of activation function to be used in every encoder layer.

    """
    def __init__(
            self,
            token_dim: int,
            layers=None,
            num_heads: int = 8,
            feedforward_dim: int = 2048,
            dropout: float = 0.1,
            activation: str = 'relu'
    ):
        super().__init__()

        if layers is None:
            layers = [0, 1]

        assert isinstance(layers, list)

        self.token_dim = token_dim
        self.layers = layers
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        self.activation = activation

        layers_list = []

        for t in layers:
            spatial = (t == 0)

            if t not in [0, 1]:
                raise ValueError(f"Type of Transformer encoder layers should be either 0 or 1, not {t}.")

            encoder = DividedEncoder(
                token_dim=token_dim,
                spatial=spatial,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                activation=activation
            )
            layers_list.append(encoder)

        self.net = nn.Sequential(*layers_list)

    def forward(self, x: Tensor):
        """
        Expected Input Dimensions (B, F, L, D), where:
        - B: batch size
        - F: temporal token dimension
        - L: spatial token dimension
        - D: token dimension size

        Expected Output Dimensions (B, F, L, D)
        """
        return self.net(x)
