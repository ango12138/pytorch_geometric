from collections.abc import Mapping, Sequence
from typing import Any, List, Union

import torch
from torch import Tensor
from torch.nn.functional import pad

from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import BaseStorage
from torch_geometric.typing import SparseTensor, TensorFrame


def separate(cls, batch: BaseData, idx: Union[int, List[int]], slice_dict: Any,
             inc_dict: Any = None, decrement: bool = True) -> BaseData:
    # Separates the individual element from a `batch` at index `idx`.
    # `separate` can handle both homogeneous and heterogeneous data objects by
    # individually separating all their stores.
    # In addition, `separate` can handle nested data structures such as
    # dictionaries and lists.

    if isinstance(idx, int):
        idx = [idx]

    return_batch = len(idx) > 1
    # Make sure the correct cls is passed
    # If we want to return a batch, the cls must be Batch
    # otherwise it must be BaseData
    if (return_batch == cls is BaseData):
        raise Exception

    data = cls().stores_as(batch)

    # We iterate over each storage object and recursively separate all its
    # attributes:
    for batch_store, data_store in zip(batch.stores, data.stores):
        key = batch_store._key
        if key is not None:
            attrs = slice_dict[key].keys()
        else:
            attrs = set(batch_store.keys())
            attrs = [attr for attr in slice_dict.keys() if attr in attrs]
        for attr in attrs:
            if key is not None:
                slices = slice_dict[key][attr]
                incs = inc_dict[key][attr] if decrement else None
            else:
                slices = slice_dict[attr]
                incs = inc_dict[attr] if decrement else None
            data_store[attr] = _separate(attr, batch_store[attr], idx, slices,
                                         incs, batch, batch_store, decrement)

        # The `num_nodes` attribute needs special treatment, as we cannot infer
        # the real number of nodes from the total number of nodes alone:
        if hasattr(batch_store, '_num_nodes'):
            if return_batch:
                data_store._num_nodes = [
                    batch_store._num_nodes[i] for i in idx
                ]
            else:
                data_store.num_nodes = batch_store._num_nodes[idx[0]]

    return data


def _separate(
    key: str,
    value: Any,
    idx: Union[int, List[int]],
    slices: Any,
    incs: Any,
    batch: BaseData,
    store: BaseStorage,
    decrement: bool,
) -> Any:

    if isinstance(value, Tensor):
        graph_slice = torch.concat([
            torch.arange(int(slices[i]), int(slices[i + 1])) for i in idx
        ]).to(value.device)
        valid_inc = incs is not None and (incs.dim() > 1
                                          or any(incs[idx] != 0))
        if isinstance(idx, torch.Tensor):
            raise Exception

        # Narrow a `torch.Tensor` based on `slices`.
        # NOTE: We need to take care of decrementing elements appropriately.
        key = str(key)
        cat_dim = batch.__cat_dim__(key, value, store)
        value = torch.index_select(value, cat_dim or 0, graph_slice)
        value = value.squeeze(0) if cat_dim is None else value

        if (decrement and incs is not None and valid_inc):
            # remove the old offset
            nelem_new = slices.diff()[idx]
            if len(idx) == 1:
                old_offset = incs[idx[0]]
                new_offset = torch.zeros_like(old_offset)
                shift = torch.ones_like(value) * (-old_offset + new_offset)
            else:
                idx = torch.tensor(idx).long().to(value.device)
                old_offset = incs[idx]
                # add the new offset
                # for this we compute the number of nodes in the batch before
                new_offset = pad(incs.diff()[idx[:-1]], (1, 0)).cumsum(0)
                shift = (-old_offset + new_offset).repeat_interleave(
                    nelem_new, dim=cat_dim or 0)
            value = value + shift
        return value

    elif isinstance(value, SparseTensor) and decrement:
        # Narrow a `SparseTensor` based on `slices`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        if len(idx) > 1:
            raise NotImplementedError
        idx = idx[0]

        key = str(key)
        cat_dim = batch.__cat_dim__(key, value, store)
        cat_dims = (cat_dim, ) if isinstance(cat_dim, int) else cat_dim
        for i, dim in enumerate(cat_dims):
            start, end = int(slices[idx][i]), int(slices[idx + 1][i])
            value = value.narrow(dim, start, end - start)
        return value

    elif isinstance(value, TensorFrame):
        if len(idx) > 1:
            raise NotImplementedError
        idx = idx[0]
        key = str(key)
        start, end = int(slices[idx]), int(slices[idx + 1])
        value = value[start:end]
        return value

    elif isinstance(value, Mapping):
        # Recursively separate elements of dictionaries.
        return {
            key:
            _separate(
                key,
                elem,
                idx,
                slices=slices[key],
                incs=incs[key] if decrement else None,
                batch=batch,
                store=store,
                decrement=decrement,
            )
            for key, elem in value.items()
        }

    elif (isinstance(value, Sequence) and isinstance(value[0], Sequence)
          and not isinstance(value[0], str) and len(value[0]) > 0
          and isinstance(value[0][0], (Tensor, SparseTensor))
          and isinstance(slices, Sequence)):
        if len(idx) > 1:
            raise NotImplementedError
        idx = idx[0]
        # Recursively separate elements of lists of lists.
        return [elem[idx] for elem in value]

    elif (isinstance(value, Sequence) and not isinstance(value, str)
          and isinstance(value[0], (Tensor, SparseTensor))
          and isinstance(slices, Sequence)):
        # Recursively separate elements of lists of Tensors/SparseTensors.
        return [
            _separate(
                key,
                elem,
                idx,
                slices=slices[i],
                incs=incs[i] if decrement else None,
                batch=batch,
                store=store,
                decrement=decrement,
            ) for i, elem in enumerate(value)
        ]
    elif isinstance(value, list) and batch._num_graphs == len(value):
        if len(idx) == 1:
            return value[idx[0]]
        else:
            return [value[i] for i in idx]
    else:
        return value
