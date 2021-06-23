from typing import Tuple, Dict, Union, Optional, Any
from torch_geometric.typing import NodeType, EdgeType, Metadata

import re
import copy
import warnings
from collections import defaultdict, deque

import torch
from torch.nn import Module

from torch_geometric.nn.fx import Transformer

try:
    from torch.fx import GraphModule, Graph, Node
except ImportError:
    GraphModule = Graph = Node = 'GraphModule', 'Graph', 'Node'


def to_hetero(module: Module, metadata: Metadata, aggr: str = "sum",
              input_map: Optional[Dict[str, str]] = None,
              debug: bool = False) -> GraphModule:
    r"""Converts a homogeneous GNN model into its heterogeneous equivalent in
    which node representations are learned for each node type in
    :obj:`metadata[0]`, and messages are exchanged between edge type in
    :obj:`metadata[1]`, as denoted in the `"Modeling Relational Data with Graph
    Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ paper:

    .. code-block:: python

        import torch
        from torch_geometric.nn import SAGEConv, to_hetero

        Net(torch.nn.Module):
            def __init__(self):
                self.lin = Linear(16, 16)
                self.conv = SAGEConv(16, 16)

            def forward(self, x, edge_index):
                x = self.lin(x)
                h = self.conv(x, edge_index)
                return torch.cat([x, h], dim=-1)

        model = Net()

        metadata = (
            ['paper', 'author'],
            [('paper' 'written_by', 'author'), ('author', 'writes', 'paper')],
        )

        model = to_hetero(model, metadata)
        model(x_dict, edge_index_dict)

    where :obj:`x_dict` and :obj:`edge_index_dict` denote dictionaries that
    hold node features and edge connectivity information for each node type and
    edge type, respectively.

    Args:
        module (torch.nn.Module): The homogeneous model to transform.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        aggr (string, optional): The aggregation scheme to use for grouping
            node embeddings generated by different relations.
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"sum"`)
        input_map (Dict[str, str], optional): A dictionary holding information
            about the type of input arguments of :obj:`module.forward`.
            For example, in case :obj:`arg` is a node-level argument, then
            :obj:`input_map['arg'] = 'node'`, and
            :obj:`input_map['arg'] = 'edge'` otherwise.
            In case :obj:`input_map` is not further specified, will try to
            automatically determine the correct type of input arguments.
            (default: :obj:`None`)
        debug: (bool, optional): If set to :obj:`True`, will perform
            transformation in debug mode. (default: :obj:`False`)
    """
    transformer = ToHeteroTransformer(module, metadata, aggr, input_map, debug)
    return transformer.transform()


class ToHeteroTransformer(Transformer):

    aggrs = {
        'sum': torch.add,
        'mean': torch.add,
        'max': torch.max,
        'min': torch.min,
        'mul': torch.mul,
    }

    def __init__(
        self,
        module: Module,
        metadata: Metadata,
        aggr: str = 'sum',
        input_map: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ):
        super().__init__(module, debug)
        self.metadata = metadata
        self.aggr = aggr
        self.input_map = input_map or {}
        assert len(metadata) == 2
        assert len(metadata[0]) > 1 and len(metadata[1]) > 1
        assert aggr in self.aggrs.keys()

    def placeholder(self, node: Node, target: Any, name: str):
        # Add a `get` call to the input dictionary for every node-type or
        # edge-type.

        input_type = self.input_map.get(name, None)
        if input_type is None and bool(re.search('(edge|adj)', name)):
            input_type = 'edge'
        is_edge_level_placeholder = input_type == 'edge'

        if node.type is not None:
            Type = EdgeType if is_edge_level_placeholder else NodeType
            node.type = Dict[Type, node.type]

        self.graph.inserting_after(node)
        for key in self.metadata[int(is_edge_level_placeholder)]:
            out = self.graph.create_node('call_method', target='get',
                                         args=(node, key),
                                         name=f'{name}__{key2str(key)}')
            self.graph.inserting_after(out)

    def get_attr(self, node: Node, target: Any, name: str):
        raise NotImplementedError

    def call_message_passing_module(self, node: Node, target: Any, name: str):
        # Add calls to edge type-wise `MessagePassing` modules and aggregate
        # the outputs to node type-wise embeddings afterwards.

        # Group edge-wise keys per destination:
        key_name, keys_per_dst = {}, defaultdict(list)
        for key in self.metadata[1]:
            keys_per_dst[key[-1]].append(key)
            key_name[key] = f'{name}__{key[-1]}{len(keys_per_dst[key[-1]])}'

        for dst, keys in dict(keys_per_dst).items():
            # In case there is only a single edge-wise connection, there is no
            # need for any destination-wise aggregation, and we can already set
            # the intermediate variable name to the final output name.
            if len(keys) == 1:
                key_name[keys[0]] = f'{name}__{dst}'
                del keys_per_dst[dst]

        self.graph.inserting_after(node)
        for key in self.metadata[1]:
            args, kwargs = self.map_args_kwargs(node, key)
            out = self.graph.create_node('call_module',
                                         target=f'{target}.{key2str(key)}',
                                         args=args, kwargs=kwargs,
                                         name=key_name[key])
            self.graph.inserting_after(out)

        # Perform destination-wise aggregation.
        # Here, we aggregate in pairs, popping the first two elements of
        # `keys_per_dst` and append the result to the list.
        for dst, keys in keys_per_dst.items():
            queue = deque([key_name[key] for key in keys])
            i = len(queue) + 1
            while len(queue) >= 2:
                key1, key2 = queue.popleft(), queue.popleft()
                args = (self.find_by_name(key1), self.find_by_name(key2))

                new_name = f'{name}__{dst}'
                if self.aggr == 'mean' or len(queue) > 2:
                    new_name += f'{i}'

                out = self.graph.create_node('call_function',
                                             target=self.aggrs[self.aggr],
                                             args=args, name=new_name)
                self.graph.inserting_after(out)
                queue.append(new_name)
                i += 1

            if self.aggr == 'mean':
                key = queue.popleft()
                out = self.graph.create_node(
                    'call_function', target=torch.div,
                    args=(self.find_by_name(key), len(keys_per_dst[dst])),
                    name=f'{name}__{dst}')
                self.graph.inserting_after(out)

    def call_module(self, node: Node, target: Any, name: str):
        # Add calls to node type-wise or edge type-wise modules.
        self.graph.inserting_after(node)
        for key in self.metadata[int(self.has_edge_level_arg_kwarg(node))]:
            args, kwargs = self.map_args_kwargs(node, key)
            out = self.graph.create_node('call_module',
                                         target=f'{target}.{key2str(key)}',
                                         args=args, kwargs=kwargs,
                                         name=f'{name}__{key2str(key)}')
            self.graph.inserting_after(out)

    def call_method(self, node: Node, target: Any, name: str):
        # Add calls to node type-wise or edge type-wise methods.
        self.graph.inserting_after(node)
        for key in self.metadata[int(self.has_edge_level_arg_kwarg(node))]:
            args, kwargs = self.map_args_kwargs(node, key)
            out = self.graph.create_node('call_method', target=target,
                                         args=args, kwargs=kwargs,
                                         name=f'{name}__{key2str(key)}')
            self.graph.inserting_after(out)

    def call_function(self, node: Node, target: Any, name: str):
        # Add calls to node type-wise or edge type-wise functions.
        self.graph.inserting_after(node)
        for key in self.metadata[int(self.has_edge_level_arg_kwarg(node))]:
            args, kwargs = self.map_args_kwargs(node, key)
            out = self.graph.create_node('call_function', target=target,
                                         args=args, kwargs=kwargs,
                                         name=f'{name}__{key2str(key)}')
            self.graph.inserting_after(out)

    def output(self, node: Node, target: Any, name: str):
        # Replace the output by dictionaries, holding either node type-wise or
        # edge type-wise data.
        def _recurse(value: Any) -> Any:
            if isinstance(value, Node):
                return {
                    key: self.find_by_name(f'{value.name}__{key2str(key)}')
                    for key in self.metadata[int(self.is_edge_level(value))]
                }
            elif isinstance(value, dict):
                return {k: _recurse(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_recurse(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(_recurse(v) for v in value)
            else:
                return value

        if node.type is not None and isinstance(node.args[0], Node):
            output = node.args[0]
            Type = EdgeType if self.is_edge_level(output) else NodeType
            node.type = Dict[Type, node.type]
        else:
            node.type = None

        node.args = (_recurse(node.args[0]), )

    def init_submodule(self, module: Module, target: str) -> Module:
        # Replicate each module for each node type or edge type.
        has_edge_level_target = bool(
            self.find_by_target(f'{target}.{key2str(self.metadata[1][0])}'))

        module_dict = torch.nn.ModuleDict()
        for key in self.metadata[int(has_edge_level_target)]:
            module_dict[key2str(key)] = copy.deepcopy(module)
            if hasattr(module, 'reset_parameters'):
                module_dict[key2str(key)].reset_parameters()
            elif sum([p for p in module.parameters()]) > 0:
                warnings.warn((f"'{target}' will be duplicated, but its "
                               f"parameters cannot be reset"))
        return module_dict

    # Helper methods ##########################################################

    def map_args_kwargs(self, node: Node,
                        key: Union[NodeType, EdgeType]) -> Tuple[Tuple, Dict]:
        def _recurse(value: Any) -> Any:
            if isinstance(value, Node):
                out = self.find_by_name(f'{value.name}__{key2str(key)}')
                if out is None and isinstance(key, tuple):
                    out = (self.find_by_name(f'{value.name}__{key[0]}'),
                           self.find_by_name(f'{value.name}__{key[-1]}'))
                return out
            elif isinstance(value, dict):
                return {k: _recurse(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_recurse(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(_recurse(v) for v in value)
            else:
                return value

        args = tuple(_recurse(v) for v in node.args)
        kwargs = {k: _recurse(v) for k, v in node.kwargs.items()}
        return args, kwargs

    def is_edge_level(self, node: Node) -> bool:
        key = self.metadata[1][0]
        return bool(self.find_by_name(f'{node.name}__{key2str(key)}'))

    def has_edge_level_arg_kwarg(self, node: Node) -> bool:
        def _recurse(value: Any) -> bool:
            if isinstance(value, Node):
                return self.is_edge_level(value)
            elif isinstance(value, dict):
                return any([_recurse(v) for v in value.values()])
            elif isinstance(value, (list, tuple)):
                return any([_recurse(v) for v in value])
            else:
                return False

        has_edge_level_arg = any([_recurse(value) for value in node.args])
        has_edge_level_kwarg = any([_recurse(v) for v in node.kwargs.values()])
        return has_edge_level_arg or has_edge_level_kwarg


def key2str(key: Union[NodeType, EdgeType]) -> str:
    return '__'.join(key) if isinstance(key, tuple) else key
