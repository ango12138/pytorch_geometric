import math

import torch
from torch import Tensor

from torch_geometric.utils import scatter


class PositionalEncoding(torch.nn.Module):
    r"""The positional encoding scheme from the `"Attention Is All You Need"
    <https://arxiv.org/pdf/1706.03762.pdf>`_ paper

    .. math::

        PE(x)_{2 \cdot i} &= \sin(x / 10000^{2 \cdot i / d})

        PE(x)_{2 \cdot i + 1} &= \cos(x / 10000^{2 \cdot i / d})

    where :math:`x` is the position and :math:`i` is the dimension.

    Args:
        out_channels (int): Size :math:`d` of each output sample.
        base_freq (float, optional): The base frequency of sinusoidal
            functions. (default: :obj:`1e-4`)
        granularity (float, optional): The granularity of the positions. If
            set to smaller value, the encoder will capture more fine-grained
            changes in positions. (default: :obj:`1.0`)
    """
    def __init__(
        self,
        out_channels: int,
        base_freq: float = 1e-4,
        granularity: float = 1.0,
    ):
        super().__init__()

        if out_channels % 2 != 0:
            raise ValueError(f"Cannot use sinusoidal positional encoding with "
                             f"odd 'out_channels' (got {out_channels}).")

        self.out_channels = out_channels
        self.base_freq = base_freq
        self.granularity = granularity

        frequency = torch.logspace(0, 1, out_channels // 2, base_freq)
        self.register_buffer('frequency', frequency)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = x / self.granularity if self.granularity != 1.0 else x
        out = x.view(-1, 1) * self.frequency.view(1, -1)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'


class TemporalEncoding(torch.nn.Module):
    r"""The time-encoding function from the `"Do We Really Need Complicated
    Model Architectures for Temporal Networks?"
    <https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper.
    :class:`TemporalEncoding` first maps each entry to a vector with
    monotonically exponentially decreasing values, and then uses the cosine
    function to project all values to range :math:`[-1, 1]`

    .. math::
        y_{i} = \cos \left(x \cdot \sqrt{d}^{-(i - 1)/\sqrt{d}} \right)

    where :math:`d` defines the output feature dimension, and
    :math:`1 \leq i \leq d`.

    Args:
        out_channels (int): Size :math:`d` of each output sample.
    """
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        sqrt = math.sqrt(out_channels)
        weight = 1.0 / sqrt**torch.linspace(0, sqrt, out_channels).view(1, -1)
        self.register_buffer('weight', weight)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        """"""
        return torch.cos(x.view(-1, 1) @ self.weight)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'


class NodeEncoding(torch.nn.Module):
    r"""The node-encoding function from the `"Do We Really Need Complicated
    Model Architectures for Temporal Networks?"
    <https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper.
    :class:`NodeEncoding` captures the node identity and node feature
        information via neighbor mean-pooling

    """
    def __init__(
        self,
        structure_time_gap: int,
        structure_hops: int,
    ):
        super().__init__()
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, data):
        """

        Args:
            data: batch of nodes from NeighborLoader

        Returns:

        """
        x, edge_index, batch_size = data.x, data.edge_index, data.batch_size
        root_nodes = x[:batch_size]

        output_feats = torch.zeros((len(root_nodes), x.size(1)))

        row, col = edge_index

        embeddings = scatter(root_nodes[row], col, dim=0,
                             dim_size=data.num_nodes, reduce='mean')
        output_feats[:batch_size] = embeddings
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
