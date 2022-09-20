from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.typing import OptTensor

from .num_nodes import maybe_num_nodes
from .subgraph import subgraph


def filter_adj(row: Tensor, col: Tensor, edge_attr: OptTensor,
               mask: Tensor) -> Tuple[Tensor, Tensor, OptTensor]:
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dropout_adj(
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    p: float = 0.5,
    force_undirected: bool = False,
    num_nodes: Optional[int] = None,
    training: bool = True,
) -> Tuple[Tensor, OptTensor]:
    r"""Randomly drops edges from the adjacency matrix
    :obj:`(edge_index, edge_attr)` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6])
        >>> dropout_adj(edge_index, edge_attr)
        (tensor([[0, 1, 2, 3],
                [1, 2, 3, 2]]),
        tensor([1, 3, 5, 6]))

        >>> # The returned graph is kept undirected
        >>> dropout_adj(edge_index, edge_attr, force_undirected=True)
        (tensor([[0, 1, 2, 1, 2, 3],
                [1, 2, 3, 0, 1, 2]]),
        tensor([1, 3, 5, 1, 3, 5]))
    """

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        return edge_index, edge_attr

    row, col = edge_index

    mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        mask[row > col] = False

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


def dropout_node(edge_index: Tensor, edge_attr: OptTensor = None,
                 p: float = 0.5, num_nodes: Optional[int] = None,
                 training: bool = True,
                 relabel_nodes: bool = False) -> Tuple[Tensor, Tensor]:
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`(edge_index, edge_attr)` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6])
        >>> dropout_node(edge_index, edge_attr)
        (tensor([[2, 3],
                [3, 2]]),
        tensor([5, 6]))

        >>> # The nodes of returned graph is relabeled
        >>> dropout_node(edge_index, edge_attr, relabel_nodes=True)
        (tensor([[0, 1],
                [1, 0]]),
        tensor([5, 6]))
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        return edge_index, edge_attr

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    nodes = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(nodes, 1 - p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    subset = nodes[mask]
    return subgraph(subset, edge_index, edge_attr, num_nodes=num_nodes,
                    relabel_nodes=relabel_nodes)
