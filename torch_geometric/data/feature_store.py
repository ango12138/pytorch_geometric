r"""
This class defines the abstraction for a backend-agnostic feature store. The
goal of a feature store is to abstract away all node and edge feature memory
management so that varying implementations can allow for independent scale-out.

This particular feature store abstraction makes a few key assumptions:
    * The features we care about storing are node and edge features of a graph.
        To this end, the attributes that the feature store supports include a
        group_name (e.g. a heterogeneous node name, a heterogeneous edge type,
        etc.), an attr_name (which defines the name of the feature tensor,
        e.g. `x`, `edge_attr`, etc.), and an index.
    * A feature can be uniquely identified from any associated attributes
        specified in :class:`TensorAttr`.

It is the job of a feature store implementor class to handle these assumptions
properly. For example, a simple in-memory feature store implementation may
concatenate all metadata values with a feature index and use this as a unique
index in a KV store. More complicated implementations may choose to partition
features in interesting manners based on the provided metadata.

Major TODOs for future implementation:
* Async `put` and `get` functionality
"""
import copy
from abc import abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import torch

from torch_geometric.typing import FeatureTensorType
from torch_geometric.utils.mixin import CastMixin

_field_status = Enum("FieldStatus", "UNSET")

# We allow indexing with a tensor, numpy array, Python slicing, or a single
# integer index.
IndexType = Union[torch.Tensor, np.ndarray, slice, int]


@dataclass
class TensorAttr(CastMixin):
    r"""Defines the attributes of a :obj:`FeatureStore` tensor; in particular,
    all the parameters necessary to uniquely identify a tensor from the feature
    store.

    Note that the order of the attributes is important; this is the order in
    which attributes must be provided for indexing calls. Feature store
    implementor classes can define a different ordering by overriding
    TensorAttr.__init__.
    """

    # The group name that the tensor corresponds to. Defaults to None.
    group_name: Optional[str] = None

    # The name of the tensor within its group. Defaults to None.
    attr_name: Optional[str] = None

    # The node indices the rows of the tensor correspond to. Defaults to UNSET.
    index: Optional[IndexType] = _field_status.UNSET

    # Convenience methods #####################################################

    def is_set(self, attr):
        r"""Whether an attribute is set in :obj:`TensorAttr`."""
        assert attr in self.__dataclass_fields__
        return getattr(self, attr) != _field_status.UNSET

    def is_fully_specified(self):
        r"""Whether the :obj:`TensorAttr` has no unset fields."""
        return all([
            getattr(self, field) != _field_status.UNSET
            for field in self.__dataclass_fields__
        ])

    def update(self, attr: 'TensorAttr'):
        r"""Updates an :obj:`TensorAttr` with set attributes from another
        :obj:`TensorAttr`."""
        for field in self.__dataclass_fields__:
            val = getattr(attr, field)
            if val != _field_status.UNSET:
                setattr(self, field, val)


class AttrView(CastMixin):
    r"""Defines a view of a :class:`FeatureStore` that is obtained from a
    specification of attributes on the feature store. The view stores a
    reference to the backing feature store as well as a :class:`TensorAttr`
    object that represents the view's state.

    Users can create views either using the :obj:`AttrView` constructor,
    :obj:`FeatureStore.view`, or by incompletely indexing a feature store. For
    example, the following calls all create views:

    .. code-block:: python

        store[group_name]
        store[group_name].feat
        store[group_name, feat]

    While the following calls all materialize those views and produce tensors
    by either calling the view or fully-specifying the view:

    .. code-block:: python

        store[group_name]()
        store[group_name].feat[index]
        store[group_name, feat][index]

    """
    def __init__(self, store: 'FeatureStore', attr: TensorAttr):
        self.__dict__['_store'] = store
        self.__dict__['_attr'] = attr

    # Python built-ins ########################################################

    def __getattr__(self, key: str) -> 'AttrView':
        r"""Sets the attr_name field of the backing :obj:`TensorAttr` object to
        the attribute. This allows for :obj:`AttrView` to be indexed by
        different values of attr_name. In particular, for a feature store that
        has `feat` as an `attr_name`, the following code indexes into `feat`:

        .. code-block:: python

            store[group_name].feat[:]

        """
        out = copy.copy(self)
        out._attr.attr_name = key
        if out._attr.is_fully_specified():
            return out._store.get_tensor(out._attr)
        return out

    def __setattr__(self, key, value):
        r"""Supports attribute assignment to the backing :obj:`TensorAttr` of
        an :obj:`AttrView`. This allows for :obj:`AttrView` objects to set
        their backing attribute values. In particular, the following operation
        sets the `index` of an :obj:`AttrView`:

        .. code-block:: python

            view = store.view(TensorAttr(group_name))
            view.index = torch.Tensor([1, 2, 3])

        """
        setattr(self._attr, key, value)

    def __getitem__(
        self,
        index: IndexType,
    ) -> Union['AttrView', FeatureTensorType]:
        r"""Supports indexing the backing :obj:`TensorAttr` object by an
        index or a slice. If the index operation results in a fully-specified
        :obj:`AttrView`, a Tensor is returned. Otherwise, the :obj:`AttrView`
        object is returned. The following operation returns a Tensor object
        as a result of the index specification:

        .. code-block:: python

            store[group_name, attr_name][:]

        """
        out = copy.copy(self)
        out._attr.index = index
        if out._attr.is_fully_specified():
            return out._store.get_tensor(out._attr)
        return out

    def __call__(self) -> FeatureTensorType:
        r"""Supports :obj:`AttrView` as a callable to force retrieval from
        the currently specified attributes. In particular, this passes the
        current :obj:`TensorAttr` object to a GET call, regardless of whether
        all attributes have been specified. It returns the result of this
        call. In particular, the following operation returns a Tensor by
        performing a GET operation on the backing feature store:

        .. code-block:: python

            store[group_name, attr_name]()

        """
        return self._store.get_tensor(self._attr)

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        return out

    def __eq__(self, __o: object) -> bool:
        r"""Compares two :obj:`AttrView` objects by checking equality of their
        :obj:`FeatureStore` references and :obj:`TensorAttr` attributes."""
        if not isinstance(__o, AttrView):
            return False

        return self._store == __o._store and self._attr == __o._attr

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(store={self._store}, '
                f'attr={self._attr})')


class FeatureStore(MutableMapping):
    def __init__(self, attr_cls: Any = TensorAttr):
        r"""Initializes the feature store. Implementor classes can customize
        the ordering and require nature of their :obj:`TensorAttr` tensor
        attributes by subclassing :class:`TensorAttr` and passing the subclass
        as `attr_cls`."""
        super().__init__()
        self._attr_cls = attr_cls

    # Core (CRUD) #############################################################

    @abstractmethod
    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        r"""Implemented by :obj:`FeatureStore` subclasses."""
        pass

    def put_tensor(self, tensor: FeatureTensorType, *args, **kwargs) -> bool:
        r"""Synchronously adds a :obj:`FeatureTensorType` object to the feature
        store.

        Args:
            tensor (FeatureTensorType): the features to be added.
            attr (TensorAttr): any relevant tensor attributes that correspond
                to the feature tensor. See the :obj:`TensorAttr` documentation
                for required and optional attributes. It is the job of
                implementations of a FeatureStore to store this metadata in a
                meaningful way that allows for tensor retrieval from a
                :obj:`TensorAttr` object.
        Returns:
            bool: whether insertion was successful.
        """
        attr = self._attr_cls.cast(*args, **kwargs)
        return self._put_tensor(tensor, attr)

    @abstractmethod
    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        r"""Implemented by :obj:`FeatureStore` subclasses."""
        pass

    def get_tensor(self, *args, **kwargs) -> Optional[FeatureTensorType]:
        r"""Synchronously obtains a :obj:`FeatureTensorType` object from the
        feature store. Feature store implementors guarantee that the call
        get_tensor(put_tensor(tensor, attr), attr) = tensor holds.

        Args:
            attr (TensorAttr): any relevant tensor attributes that correspond
                to the tensor to obtain. See :obj:`TensorAttr` documentation
                for required and optional attributes. It is the job of
                implementations of a FeatureStore to store this metadata in a
                meaningful way that allows for tensor retrieval from a
                :obj:`TensorAttr` object.
        Returns:
            FeatureTensorType, optional: a tensor of the same type as the
            index, or None if no tensor was found.
        """
        def to_type(tensor):
            if tensor is None:
                return None
            if isinstance(attr.index, torch.Tensor):
                return torch.from_numpy(tensor) if isinstance(
                    tensor, np.ndarray) else tensor
            if isinstance(attr.index, np.ndarray):
                return tensor.numpy() if isinstance(tensor,
                                                    torch.Tensor) else tensor
            return tensor

        attr = self._attr_cls.cast(*args, **kwargs)
        if isinstance(attr.index,
                      slice) and (attr.index.start, attr.index.stop,
                                  attr.index.step) == (None, None, None):
            attr.index = None

        return to_type(self._get_tensor(attr))

    @abstractmethod
    def _remove_tensor(self, attr: TensorAttr) -> bool:
        r"""Implemented by :obj:`FeatureStore` subclasses."""
        pass

    def remove_tensor(self, *args, **kwargs) -> bool:
        r"""Removes a :obj:`FeatureTensorType` object from the feature store.

        Args:
            attr (TensorAttr): any relevant tensor attributes that correspond
                to the tensor to remove. See :obj:`TensorAttr` documentation
                for required and optional attributes. It is the job of
                implementations of a FeatureStore to store this metadata in a
                meaningful way that allows for tensor deletion from a
                :obj:`TensorAttr` object.

        Returns:
            bool: whether deletion was succesful.
        """
        attr = self._attr_cls.cast(*args, **kwargs)
        self._remove_tensor(attr)

    def update_tensor(self, tensor: FeatureTensorType, *args,
                      **kwargs) -> bool:
        r"""Updates a :obj:`FeatureTensorType` object with a new value.
        implementor classes can choose to define more efficient update methods;
        the default performs a removal and insertion.

        Args:
            tensor (FeatureTensorType): the features to be added.
            attr (TensorAttr): any relevant tensor attributes that correspond
                to the old tensor. See :obj:`TensorAttr` documentation
                for required and optional attributes. It is the job of
                implementations of a FeatureStore to store this metadata in a
                meaningful way that allows for tensor update from a
                :obj:`TensorAttr` object.

        Returns:
            bool: whether the update was succesful.
        """
        attr = self._attr_cls.cast(*args, **kwargs)
        self.remove_tensor(attr)
        return self.put_tensor(tensor, attr)

    # :obj:`AttrView` methods #################################################

    def view(self, *args, **kwargs) -> AttrView:
        r"""Returns an :obj:`AttrView` of the feature store, with the defined
        attributes set."""
        return AttrView(self, self._attr_cls.cast(*args, **kwargs))

    # Python built-ins ########################################################

    def __setitem__(self, key: TensorAttr, value: FeatureTensorType):
        r"""Supports store[tensor_attr] = tensor."""
        key = self._attr_cls.cast(key)
        self.put_tensor(value, key)

    def __getitem__(self, key: TensorAttr):
        r"""Supports pythonic indexing into the feature store. In particular,
        the following rules are followed for indexing:

        * Fully-specified indexes will produce a Tensor output. A
            fully-specified index specifies all the required attributes in
            :obj:`TensorAttr`.

        * Partially-specified indexes will produce an AttrView output, which
            is a view on the FeatureStore. If a view is called, it will produce
            a Tensor output from the corresponding (partially specified)
            attributes.
        """
        # CastMixin will handle the case of key being a tuple or TensorAttr
        # object.
        attr = self._attr_cls.cast(key)
        if attr.is_fully_specified():
            return self.get_tensor(attr)
        return AttrView(self, attr)

    def __delitem__(self, key: TensorAttr):
        r"""Supports del store[tensor_attr]."""
        key = self._attr_cls.cast(key)
        self.remove_tensor(key)

    def __iter__(self):
        raise NotImplementedError

    def __eq__(self, __o: object) -> bool:
        return id(self) == id(__o)

    @abstractmethod
    def __len__(self):
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
