from typing import Dict, Optional
from torch_geometric.typing import NodeType, EdgeType, Adj

from collections import defaultdict

from torch import Tensor
from torch.nn import Module, ModuleDict
from torch_geometric.nn.conv.hgt_conv import group


class HeteroConv(Module):
    r"""A generic wrapper for computing graph convolution on heterogeneous
    graphs.
    This layer will pass messages from source nodes to target nodes based on
    the bipartite GNN layer given for a specific edge type.
    If multiple relations point to the same destination, their results will be
    aggregated according to :attr:`aggr`.
    In comparison to :meth:`torch_geometric.nn.to_hetero`, this layer is
    especially useful if you want to apply different message passing modules
    for different edge types.

    .. code-block:: python

        hetero_conv = HeteroConv({
            ('paper', 'cites', 'paper'): GCNConv(-1, 64),
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), 64),
            ('paper', 'written_by', 'author'): GATConv((-1, -1), 64),
        }, aggr='sum')

        out_dict = hetero_conv(x_dict, edge_index_dict)

        print(list(out_dict.keys()))
        >>> ['paper', 'author']

    Args:
        convs (Dict[Tuple[str, str, str], Module]): A dictionary
            holding a bipartite
            :class:`~torch_geometric.nn.conv.MessagePassing` layer for each
            individual edge type.
        aggr (string, optional): The aggregation scheme to use for grouping
            node embeddings generated by different relations.
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            obj:`None`). (default: :obj:`"sum"`)
    """
    def __init__(self, convs: Dict[EdgeType, Module],
                 aggr: Optional[str] = "sum"):
        super().__init__()
        self.convs = ModuleDict({'__'.join(k): v for k, v in convs.items()})
        self.aggr = aggr

    def reset_parameters(self):
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        **kwargs_dict,
    ) -> Dict[NodeType, Tensor]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
                holding graph connectivity information for each individual
                edge type.
            **kwargs_dict (optional): Additional forward arguments
                of individual :class:`torch_geometric.nn.conv.MessagePassing`
                layers.
                For example, if a specific GNN layer at edge type
                :obj:`edge_type` expects edge attributes :obj:`edge_attr` as a
                forward argument, then you can pass them to
                :meth:`~torch_geometric.nn.conv.HeteroConv.forward` via
                :obj:`edge_attr_dict = { edge_type: edge_attr }`.
        """
        out_dict = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type

            str_edge_type = '__'.join(edge_type)
            if str_edge_type not in self.convs:
                continue

            kwargs = {
                arg[:-5]: value[edge_type]  # `{*}_dict`
                for arg, value in kwargs_dict.items() if edge_type in value
            }

            conv = self.convs[str_edge_type]

            if src == dst:
                out = conv(x=x_dict[src], edge_index=edge_index, **kwargs)
            else:
                out = conv(x=(x_dict[src], x_dict[dst]), edge_index=edge_index,
                           **kwargs)

            out_dict[dst].append(out)

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        return out_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'
