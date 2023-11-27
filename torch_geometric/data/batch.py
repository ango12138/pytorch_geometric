import inspect
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import pad
from typing_extensions import Self

from torch_geometric.data.collate import collate
from torch_geometric.data.data import BaseData, Data
from torch_geometric.data.dataset import IndexType
from torch_geometric.data.separate import separate


class DynamicInheritance(type):
    # A meta class that sets the base class of a `Batch` object, e.g.:
    # * `Batch(Data)` in case `Data` objects are batched together
    # * `Batch(HeteroData)` in case `HeteroData` objects are batched together
    def __call__(cls, *args, **kwargs):
        base_cls = kwargs.pop('_base_cls', Data)

        if issubclass(base_cls, Batch):
            new_cls = base_cls
        else:
            name = f'{base_cls.__name__}{cls.__name__}'

            # NOTE `MetaResolver` is necessary to resolve metaclass conflict
            # problems between `DynamicInheritance` and the metaclass of
            # `base_cls`. In particular, it creates a new common metaclass
            # from the defined metaclasses.
            class MetaResolver(type(cls), type(base_cls)):
                pass

            if name not in globals():
                globals()[name] = MetaResolver(name, (cls, base_cls), {})
            new_cls = globals()[name]

        params = list(inspect.signature(base_cls.__init__).parameters.items())
        for i, (k, v) in enumerate(params[1:]):
            if k == 'args' or k == 'kwargs':
                continue
            if i < len(args) or k in kwargs:
                continue
            if v.default is not inspect.Parameter.empty:
                continue
            kwargs[k] = None

        return super(DynamicInheritance, new_cls).__call__(*args, **kwargs)


class DynamicInheritanceGetter:
    def __call__(self, cls, base_cls):
        return cls(_base_cls=base_cls)


class Batch(metaclass=DynamicInheritance):
    r"""A data object describing a batch of graphs as one big (disconnected)
    graph.
    Inherits from :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData`.
    In addition, single graphs can be identified via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.

    :pyg:`PyG` allows modification to the underlying batching procedure by
    overwriting the :meth:`~Data.__inc__` and :meth:`~Data.__cat_dim__`
    functionalities.
    The :meth:`~Data.__inc__` method defines the incremental count between two
    consecutive graph attributes.
    By default, :pyg:`PyG` increments attributes by the number of nodes
    whenever their attribute names contain the substring :obj:`index`
    (for historical reasons), which comes in handy for attributes such as
    :obj:`edge_index` or :obj:`node_index`.
    However, note that this may lead to unexpected behavior for attributes
    whose names contain the substring :obj:`index` but should not be
    incremented.
    To make sure, it is best practice to always double-check the output of
    batching.
    Furthermore, :meth:`~Data.__cat_dim__` defines in which dimension graph
    tensors of the same attribute should be concatenated together.
    """
    @classmethod
    def from_batch_index(cls, batch_idx: Tensor) -> Self:
        batch = Batch()

        if not batch_idx.dtype == torch.long:
            raise Exception("Batch index dtype must be torch.long")
        if not (batch_idx.diff() >= 0).all():
            raise Exception("Batch index must be increasing")
        if not batch_idx.dim() == 1:
            raise Exception()

        batch.batch = batch_idx
        batch.ptr = batch.__ptr_from_batchidx(batch_idx)
        batch._num_graphs = int(batch.batch.max() + 1)

        batch._slice_dict = defaultdict(dict)
        batch._inc_dict = defaultdict(dict)
        return batch

    @classmethod
    def from_data_list(cls, data_list: List[BaseData],
                       follow_batch: Optional[List[str]] = None,
                       exclude_keys: Optional[List[str]] = None) -> Self:
        r"""Constructs a :class:`~torch_geometric.data.Batch` object from a
        list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects.
        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`.
        """
        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], Batch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict

        return batch

    @classmethod
    def from_batch_list(
        cls,
        batches: List[Self],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> Self:
        r"""Same as :meth:`~Batch.from_data_list```,
        but for concatenating existing batches.
        Constructs a :class:`~torch_geometric.data.Batch` object from a
        list of :class:`~torch_geometric.data.Batch` objects.
        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`.
        """
        batch = cls.from_data_list(batches, follow_batch, exclude_keys)

        del batch._slice_dict["batch"], batch._inc_dict["batch"]

        batch.ptr = cls.__ptr_from_batchidx(cls, batch.batch)
        batch._num_graphs = batch.ptr.numel() - 1

        for k in set(batch.keys()) - {"batch", "ptr"}:
            # slice_shift = [0] + [be._slice_dict[k][-1] for be in batches ]
            batch._slice_dict[k] = batch._pad_zero(
                torch.concat([be._slice_dict[k].diff()
                              for be in batches]).cumsum(0))
            if k != "edge_index":
                inc_shift = batch._pad_zero(
                    torch.tensor([sum(be._inc_dict[k])
                                  for be in batches])).cumsum(0)
            else:
                inc_shift = batch._pad_zero(
                    torch.tensor([be.num_nodes for be in batches])).cumsum(0)

            batch._inc_dict[k] = torch.cat([
                be._inc_dict[k] + inc_shift[ibatch]
                for ibatch, be in enumerate(batches)
            ])
        return batch

    def get_example(self, idx: int) -> BaseData:
        r"""Gets the :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object at index :obj:`idx`.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial object.
        """
        if not hasattr(self, '_slice_dict'):
            raise RuntimeError(
                ("Cannot reconstruct 'Data' object from 'Batch' because "
                 "'Batch' was not created via 'Batch.from_data_list()'"))

        data = separate(
            cls=self.__class__.__bases__[-1],
            batch=self,
            idx=idx,
            slice_dict=self._slice_dict,
            inc_dict=self._inc_dict,
            decrement=True,
        )

        return data

    def index_select(self, idx: IndexType) -> Self:
        r"""Creates a new :class:`~torch_geometric.data.Batch`
        object from specified indices :obj:`idx`. Indices
        :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`,
        a list, a tuple, or a :obj:`torch.Tensor` or
        :obj:`np.ndarray` of type long or bool.
        """
        if isinstance(idx, slice):
            idx = list(range(self.num_graphs)[idx])

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            idx = idx.flatten().tolist()

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False).flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            idx = idx.flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0].flatten().tolist()

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            pass

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        dev = self.ptr.device

        subbatch = separate(
            cls=self.__class__.__bases__[0],
            batch=self,
            idx=idx,
            slice_dict=self._slice_dict,
            inc_dict=self._inc_dict,
            decrement=True,
        )

        idx = torch.tensor(idx).long().to(dev)
        nodes_per_graph = self.ptr.diff()
        new_nodes_per_graph = nodes_per_graph[idx]

        # Construct batch index and ptr
        subbatch.batch = torch.arange(
            len(idx)).to(dev).long().repeat_interleave(new_nodes_per_graph)
        subbatch.ptr = self.__ptr_from_batchidx(subbatch.batch)

        # fix the _slice_dict and _inc_dict
        subbatch._slice_dict = defaultdict(dict)
        subbatch._inc_dict = defaultdict(dict)
        for k in set(self.keys()) - {"ptr", "batch"}:
            if k not in self._slice_dict:
                continue
            subbatch._slice_dict[k] = pad(self._slice_dict[k].diff()[idx],
                                          (1, 0)).cumsum(0)
            if k not in self._inc_dict:
                continue
            if self._inc_dict[k] is None:
                subbatch._inc_dict[k] = None
                continue
            subbatch._inc_dict[k] = pad(self._inc_dict[k].diff()[idx[:-1]],
                                        (1, 0)).cumsum(0)
        return subbatch

    def __getitem__(self, idx: Union[int, np.integer, str, IndexType]) -> Any:
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):
            return self.get_example(idx)
        elif isinstance(idx, str) or (isinstance(idx, tuple)
                                      and isinstance(idx[0], str)):
            # Accessing attributes or node/edge types:
            return super().__getitem__(idx)
        else:
            return self.index_select(idx)

    def to_data_list(self) -> List[BaseData]:
        r"""Reconstructs the list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects from the
        :class:`~torch_geometric.data.Batch` object.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial objects.
        """
        return [self.get_example(i) for i in range(self.num_graphs)]

    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the batch."""
        if hasattr(self, '_num_graphs'):
            return self._num_graphs
        elif hasattr(self, 'ptr'):
            return self.ptr.numel() - 1
        elif hasattr(self, 'batch'):
            return int(self.batch.max()) + 1
        else:
            raise ValueError("Can not infer the number of graphs")

    @property
    def batch_size(self) -> int:
        r"""Alias for :obj:`num_graphs`."""
        return self.num_graphs

    def __len__(self) -> int:
        return self.num_graphs

    def __reduce__(self):
        state = self.__dict__.copy()
        return DynamicInheritanceGetter(), self.__class__.__bases__, state

    def add_node_attr(self, attrname: str, attr: Tensor) -> None:
        r"""Adds an attribute to the nodes in an existing batch.
        The first dimension of the :obj:`attr` must match the number
        of nodes in the batch. The exisiting
        :obj:`~torch_geometric.data.Batch.batch` will be used to
        assign the elements to the correct graph.
        """
        assert attr.device == self.batch.device
        batch_idxs = self.batch

        self[attrname] = attr
        out = batch_idxs.unique(return_counts=True)[1]
        out = out.cumsum(dim=0)
        self._slice_dict[attrname] = self._pad_zero(out).cpu()

        self._inc_dict[attrname] = torch.zeros(self._num_graphs,
                                               dtype=torch.long)

    def add_graph_attr(self, attrname: str, attr: Tensor) -> None:
        r"""Adds an attribute to the graphs in an existing batch.
        The first dimension of the :obj:`attr` must match the
        number of nodes in the batch. The exisiting :obj:`~Batch.batch`
        will be used to assign the elements to the correct graph.
        """
        assert attr.device == self.batch.device

        self[attrname] = attr
        self._slice_dict[attrname] = torch.arange(self.num_graphs + 1,
                                                  dtype=torch.long)

        self._inc_dict[attrname] = torch.zeros(self.num_graphs,
                                               dtype=torch.long)

    def set_edge_index(self, edge_index: Tensor,
                       batchidx_per_edge: Tensor) -> None:
        r"""Sets or overwrites :obj:`edge_index` in an existing batch.
        For this, :obj:`batchidx_per_edge` should contain, to which of
        the graphs each of the pair of nodes belongs. :obj:`edge_index`
        must have the shape :obj:`[2:num_edges]`; :obj:`batchidx_per_edge`
        must have the shape :obj:`[num_edges]`. Both must be instances of
        :class:`torch.LongTensor`. The exisiting :obj:`~Batch.ptr` will
        be used to assign the elements to the correct graph.
        """
        assert edge_index.dtype == batchidx_per_edge.dtype == torch.long
        assert (edge_index.device == batchidx_per_edge.device ==
                self.batch.device)
        assert (batchidx_per_edge.diff()
                >= 0).all(), "Edges must be ordered by batch"
        if 'edge_index' not in self._store.keys():
            self.edge_index = torch.empty(2, 0, dtype=torch.long,
                                          device=self.batch.device)
        # Edges must be shifted by the number sum of the nodes
        # in the previous graphs
        edge_index += self.ptr[batchidx_per_edge]
        self.edge_index = torch.hstack((self.edge_index.clone(), edge_index))
        # Fix _slice_dict
        edges_per_graph = batchidx_per_edge.unique(return_counts=True)[1]
        self._slice_dict["edge_index"] = self._pad_zero(
            edges_per_graph.cumsum(0)).cpu()
        self._inc_dict["edge_index"] = self.ptr[:-1].cpu()

    def set_edge_attr(self, edge_attr: Tensor) -> None:
        r"""Sets or overwrites :obj:`edge_attr` in an existing batch.
        The first dimension of :obj:`edge_attr` must match the number
        of edges in the batch. The exisiting :obj:`~Batch.edge_index`
        will be used to assign the elements to the correct graph.
        """
        assert (hasattr(self, "edge_index")
                and self["edge_index"].dtype == torch.long)
        self.edge_attr = edge_attr
        self._slice_dict["edge_attr"] = self._slice_dict["edge_index"]
        self._inc_dict["edge_attr"] = torch.zeros(self.num_graphs)

    def _pad_zero(self, arr: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            torch.tensor(0, dtype=arr.dtype, device=arr.device).unsqueeze(0),
            arr
        ])

    def __ptr_from_batchidx(self, batch_idx: Tensor) -> torch.Tensor:
        # Construct the ptr to adress single graphs
        assert batch_idx.dtype == torch.long
        # graph[idx].x== batch.x[batch.ptr[idx]:batch.ptr[idx]+1]
        # Get delta with diff
        # Get idx of diff >0 with nonzero
        # shift by -1
        # add the batch size -1 as last element and add 0 in front
        dev = batch_idx.device
        ptr = torch.concatenate((
            torch.tensor(0).long().to(dev).unsqueeze(0),
            (batch_idx.diff()).nonzero().reshape(-1) + 1,
            torch.tensor(len(batch_idx)).long().to(dev).unsqueeze(0),
        ))
        return ptr
