from typing import Dict, Union, Optional
from torch_geometric.typing import NodeType, EdgeType

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
    aggregated according to :obj:`aggr`.
    In comparison to :meth:`torch_geometric.nn.to_hetero`, this layer is
    especially useful if you want to apply different message passing modules
    for different edge types.

    ..code-block:: python

        hetero_conv = HeteroConv({
            ('paper', 'cites', 'paper'): GCNConv(-1, 64),
            ('author', 'writes', 'paper'): SAGEConv(-1, 64),
            ('paper', 'written_by', 'author'): GATConv(-1, 64),
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
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)

    """
    def __init__(self, convs: Dict[EdgeType, Module], aggr: str = "sum"):
        super().__init__()
        self.keys = list(convs.keys())
        self.convs = ModuleDict({'__'.join(k): v for k, v in convs.items()})
        self.aggr = aggr

    def reset_parameters(self):
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Union[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor]],
        edge_weight_dict: Optional[Dict[EdgeType, Tensor]] = None,
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
    ) -> Dict[NodeType, Tensor]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
                holding graph connectivity information for each individual
                edge type.
            edge_weight_dict (Dict[Tuple[str, str, str], Tensor], optional): A
                dictionary holding one-dimensional edge weight information
                for individual edge types. (default: :obj:`None`)
            edge_attr_dict (Dict[Tuple[str, str, str], Tensor], optional): A
                dictionary holding multi-dimensional edge feature information
                for individual edge types. (default: :obj:`None`)
        """

        out_dict = defaultdict(list)
        for key in self.keys:
            src, _, dst = key
            conv = self.convs['__'.join(key)]

            kwargs = {}
            if edge_weight_dict is not None and key in edge_weight_dict:
                kwargs['edge_weight'] = edge_weight_dict[key]
            if edge_weight_dict is not None and key in edge_attr_dict:
                kwargs['edge_attr'] = edge_attr_dict[key]

            if src == dst:
                out = conv(x=x_dict[src], edge_index=edge_index_dict[key],
                           **kwargs)
            else:
                out = conv(x=(x_dict[src], x_dict[dst]),
                           edge_index=edge_index_dict[key], **kwargs)

            out_dict[dst].append(out)

        for key, values in out_dict.items():
            out_dict[key] = group(values, self.aggr)

        return out_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'
