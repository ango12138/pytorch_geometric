from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader.utils import (
    edge_type_to_str,
    to_csc,
    to_hetero_csc,
)
from torch_geometric.sampler.base import (
    BaseSampler,
    SamplerInput,
    SamplerOutput,
)
from torch_geometric.typing import NumNeighbors


class NeighborSampler(BaseSampler):
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: NumNeighbors,
        replace: bool = False,
        directed: bool = True,
        input_type: Optional[Any] = None,
        time_attr: Optional[str] = None,
        is_sorted: bool = False,
        share_memory: bool = False,
    ):
        super().__init__()
        self.data_cls = data.__class__ if isinstance(
            data, (Data, HeteroData)) else 'custom'
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = directed
        self.node_time = None

        # TODO Unify the following conditionals behind the `FeatureStore`
        # and `GraphStore` API

        # If we are working with a `Data` object, convert the edge_index to
        # CSC and store it:
        if isinstance(data, Data):
            if time_attr is not None:
                # TODO `time_attr` support for homogeneous graphs
                raise ValueError(
                    f"'time_attr' attribute not yet supported for "
                    f"'{data.__class__.__name__}' object")

            # Convert the graph data into a suitable format for sampling.
            out = to_csc(data, device='cpu', share_memory=share_memory,
                         is_sorted=is_sorted)
            self.colptr, self.row, self.perm = out
            assert isinstance(num_neighbors, (list, tuple))

        # If we are working with a `HeteroData` object, convert each edge
        # type's edge_index to CSC and store it:
        elif isinstance(data, HeteroData):
            if time_attr is not None:
                self.node_time_dict = data.collect(time_attr)
            else:
                self.node_time_dict = None

            # Convert the graph data into a suitable format for sampling.
            # NOTE: Since C++ cannot take dictionaries with tuples as key as
            # input, edge type triplets are converted into single strings.
            out = to_hetero_csc(data, device='cpu', share_memory=share_memory,
                                is_sorted=is_sorted)
            self.colptr_dict, self.row_dict, self.perm_dict = out

            self.node_types, self.edge_types = data.metadata()
            self._set_num_neighbors_and_num_hops(num_neighbors)

            assert input_type is not None
            self.input_type = input_type

        # If we are working with a `Tuple[FeatureStore, GraphStore]` object,
        # obtain edges from GraphStore and convert them to CSC if necessary,
        # storing the resulting representations:
        elif isinstance(data, tuple):
            # TODO support `FeatureStore` with no edge types (e.g. `Data`)
            feature_store, graph_store = data

            # TODO support `collect` on `FeatureStore`
            self.node_time_dict = None
            if time_attr is not None:
                # We need to obtain all features with 'attr_name=time_attr'
                # from the feature store and store them in node_time_dict. To
                # do so, we make an explicit feature store GET call here with
                # the relevant 'TensorAttr's
                time_attrs = [
                    attr for attr in feature_store.get_all_tensor_attrs()
                    if attr.attr_name == time_attr
                ]
                for attr in time_attrs:
                    attr.index = None
                time_tensors = feature_store.multi_get_tensor(time_attrs)
                self.node_time_dict = {
                    time_attr.group_name: time_tensor
                    for time_attr, time_tensor in zip(time_attrs, time_tensors)
                }

            # Obtain all node and edge metadata:
            node_attrs = feature_store.get_all_tensor_attrs()
            edge_attrs = graph_store.get_all_edge_attrs()

            self.node_types = list(
                set(node_attr.group_name for node_attr in node_attrs))
            self.edge_types = list(
                set(edge_attr.edge_type for edge_attr in edge_attrs))

            # Set other required parameters:
            self._set_num_neighbors_and_num_hops(num_neighbors)

            assert input_type is not None
            self.input_type = input_type

            # Obtain CSC representations for in-memory sampling:
            row_dict, colptr_dict, perm_dict = graph_store.csc()
            self.row_dict = {
                edge_type_to_str(k): v
                for k, v in row_dict.items()
            }
            self.colptr_dict = {
                edge_type_to_str(k): v
                for k, v in colptr_dict.items()
            }
            self.perm_dict = {
                edge_type_to_str(k): v
                for k, v in perm_dict.items()
            }

        else:
            raise TypeError(f'NeighborLoader found invalid type: {type(data)}')

    def _set_num_neighbors_and_num_hops(self, num_neighbors):
        if isinstance(num_neighbors, (list, tuple)):
            num_neighbors = {key: num_neighbors for key in self.edge_types}
        assert isinstance(num_neighbors, dict)
        self.num_neighbors = {
            edge_type_to_str(key): value
            for key, value in num_neighbors.items()
        }
        # Add at least one element to the list to ensure `max` is well-defined
        self.num_hops = max([0] + [len(v) for v in num_neighbors.values()])

    def _sparse_neighbor_sample(self, index: Tensor) -> SamplerOutput:
        fn = torch.ops.torch_sparse.neighbor_sample
        node, row, col, edge = fn(
            self.colptr,
            self.row,
            index,
            self.num_neighbors,
            self.replace,
            self.directed,
        )
        return node, row, col, edge

    def _hetero_sparse_neighbor_sample(self, index_dict: Dict[str, Tensor],
                                       **kwargs) -> SamplerOutput:
        if self.node_time_dict is None:
            fn = torch.ops.torch_sparse.hetero_neighbor_sample
            node_dict, row_dict, col_dict, edge_dict = fn(
                self.node_types,
                self.edge_types,
                self.colptr_dict,
                self.row_dict,
                index_dict,
                self.num_neighbors,
                self.num_hops,
                self.replace,
                self.directed,
            )
        else:
            try:
                fn = torch.ops.torch_sparse.hetero_temporal_neighbor_sample
            except RuntimeError as e:
                raise RuntimeError(
                    "'torch_sparse' operator "
                    "'hetero_temporal_neighbor_sample' not "
                    "found. Currently requires building "
                    "'torch_sparse' from master.", e)

            node_dict, row_dict, col_dict, edge_dict = fn(
                self.node_types,
                self.edge_types,
                self.colptr_dict,
                self.row_dict,
                index_dict,
                self.num_neighbors,
                kwargs.get('node_time_dict', self.node_time_dict),
                self.num_hops,
                self.replace,
                self.directed,
            )
        return node_dict, row_dict, col_dict, edge_dict

    def __call__(self, index: SamplerInput) -> SamplerOutput:
        if not isinstance(index, torch.LongTensor):
            index = torch.LongTensor(index)

        if self.data_cls != 'custom' and issubclass(self.data_cls, Data):
            return self._sparse_neighbor_sample(index) + (index.numel(), )

        elif self.data_cls == 'custom' or issubclass(self.data_cls,
                                                     HeteroData):
            return self._hetero_sparse_neighbor_sample(
                {self.input_type: index}) + (index.numel(), )
