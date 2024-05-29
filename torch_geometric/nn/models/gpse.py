import logging
import os
import os.path as osp
import time
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.graphgym.init import init_weights
from torch_geometric.graphgym.models.encoder import (
    AtomEncoder,
    BondEncoder,
    IntegerFeatureEncoder,
)
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.nn import (
    ResGatedGraphConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_dense_batch,
)


class GeneralLayer(nn.Module):
    r"""General wrapper for layers, based on PyG GraphGym.

    Args:
        name (str): Name of the layer registered in :obj:`layer_dict`.
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        has_bn (bool): Whether to apply batch normalization to layer outputs.
        bn_eps (float): Epsilon for batch normalization.
        bn_mom (float): Momentum for batch normalization.
        has_l2norm (bool): Whether to apply L2 normalization to the layer
            outputs.
        dropout (float): Dropout ratio at layer output.
        has_act (bool): Whether to apply an activation after the layers.
        act (str): Activation to apply to layer outputs.
        **kwargs (optional): Additional args for :class:`GPSE` layer classes.
    """
    def __init__(self, name: str, dim_in: int, dim_out: int, has_bn: bool,
                 bn_eps: float, bn_mom: float, has_l2norm: bool,
                 dropout: float, has_act: bool, act: str, **kwargs):
        super().__init__()
        layer_dict = {
            'linear': Linear,
            'resgatedgcnconv': ResGatedGCNConvGraphGymLayer
        }
        self.has_l2norm = has_l2norm
        has_bias = not has_bn
        self.layer = layer_dict[name](dim_in, dim_out, bias=has_bias, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                nn.BatchNorm1d(dim_out, eps=bn_eps, momentum=bn_mom))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout, inplace=False))
        if has_act:
            layer_wrapper.append(activation_resolver(act))
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch


class GeneralMultiLayer(nn.Module):
    r"""General wrapper for a stack of :class:`GeneralLayer`s,
    based on PyG GraphGym.

    Args:
        name (str): Name of the layer registered in :obj:`layer_dict` of
            :class:`GeneralLayer`.
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        dim_inner (int): Dimension of the inner layers.
        num_layers (int): Number of layers.
        has_bn (bool): Whether to apply batch normalization to layer outputs.
        bn_eps (float): Epsilon for batch normalization.
        bn_mom (float): Momentum for batch normalization.
        has_l2norm (bool): Whether to apply L2 normalization to the layer
            outputs.
        dropout (float): Dropout ratio at layer output.
        final_act (bool): Whether to apply an activation after the final layer.
        act (str): Activation to apply to layer outputs.
        **kwargs (optional): Additional args for :class:`GeneralLayer`.
    """
    def __init__(self, name: str, dim_in: int, dim_out: int, dim_inner: int,
                 num_layers: int, has_bn: bool, bn_eps: float, bn_mom: float,
                 has_l2norm: bool, dropout: float, final_act: bool, act: str,
                 **kwargs):
        super().__init__()
        dim_inner = dim_out if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            layer = GeneralLayer(name, d_in, d_out, has_bn, bn_eps, bn_mom,
                                 has_l2norm, dropout, has_act, act, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


class Linear(nn.Module):
    r"""Basic Linear layer, wrapper on :class:`~torch_geometric.nn.Linear`
    based on PyG GraphGym.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        bias (bool): Whether the layer has a bias term.
        **kwargs (optional): Additional args for
            :class:`~torch_geometric.nn.Linear`.
    """
    def __init__(self, dim_in: int, dim_out: int, bias: bool, **kwargs):
        super().__init__()
        self.model = Linear_pyg(dim_in, dim_out, bias=bias, **kwargs)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class ResGatedGCNConvGraphGymLayer(nn.Module):
    r"""ResGatedGCN layer, wrapper on
    :class:`~torch_geometric.nn.ResGatedGraphConv` based on PyG GraphGym.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        bias (bool): Whether the layer has a bias term.
        **kwargs (optional): Additional args for
            :class:`~torch_geometric.nn.ResGatedGraphConv`.
    """
    def __init__(self, dim_in: int, dim_out: int, bias: bool, **kwargs):
        super().__init__()
        self.model = ResGatedGraphConv(dim_in, dim_out, bias=bias, **kwargs)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


class BatchNorm1dNode(nn.Module):
    r"""Batch normalization for node features, wrapper on
        :class:`~torch_geometric.nn.BatchNorm1d` based on PyG GraphGym.

    Args:
        dim_in (int): Input dimension.
        bn_eps (float): Epsilon for batch normalization.
        bn_mom (float): Momentum for batch normalization.
    """
    def __init__(self, dim_in: int, bn_eps: float, bn_mom: float):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=bn_eps, momentum=bn_mom)

    def forward(self, batch):
        batch.x = self.bn(batch.x)
        return batch


class BatchNorm1dEdge(nn.Module):
    r"""Batch normalization for edge features, wrapper on
        :class:`~torch_geometric.nn.BatchNorm1d` based on PyG GraphGym.

    Args:
        dim_in (int): Input dimension.
        bn_eps (float): Epsilon for batch normalization.
        bn_mom (float): Momentum for batch normalization.
    """
    def __init__(self, dim_in: int, bn_eps: float, bn_mom: float):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=bn_eps, momentum=bn_mom)

    def forward(self, batch):
        batch.edge_attr = self.bn(batch.edge_attr)
        return batch


class MLP(nn.Module):
    r"""Basic MLP model, based on PyG GraphGym. Here, 1-layer MLP is equivalent
    to a :class:`Linear` layer. A multi-layer MLP is a stack of :class:`Linear`
    layers, in the form of a :class:`GeneralMultiLayer`.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        dim_inner (int): The dimension for the inner layers. If None, it is set
            to the input dimension.
        num_layers (int): Number of hidden layers in the MLP.
        has_bn (bool, optional): Whether to apply batch normalization to layer
            outputs. (default: :obj:`True`)
        bn_eps (float, optional): Epsilon for batch normalization.
            (default: :obj:`1.0e-05`)
        bn_mom (float, optional): Momentum for batch normalization.
            (default: :obj:`0.1`)
        has_l2norm (bool, optional): Whether to apply L2 normalization to the
            layer outputs. (default: :obj:`True`)
        dropout (float, optional): Dropout ratio at layer output.
            (default: :obj:`0.2`)
        act (str, optional): Activation to apply to layer output.
            (default: :obj:`relu`)
        **kwargs (optional): Additional args for GeneralLayer.
    """
    def __init__(self, dim_in: int, dim_out: int, dim_inner: int,
                 num_layers: int, has_bn: bool = True, bn_eps: float = 1.0e-05,
                 bn_mom: float = 0.1, has_l2norm: bool = True,
                 dropout: float = 0.2, act: str = 'relu', **kwargs):
        super().__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        layers = []
        if num_layers > 1:
            layers.append(
                GeneralMultiLayer('linear', dim_in, dim_inner, dim_inner,
                                  num_layers - 1, has_bn, bn_eps, bn_mom,
                                  has_l2norm, dropout, final_act=True, act=act,
                                  **kwargs))
            layers.append(Linear(dim_inner, dim_out, bias=True))
        else:
            layers.append(Linear(dim_inner, dim_out, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class GNNInductiveHybridMultiHead(nn.Module):
    r"""GNN prediction head for inductive node and graph prediction tasks using
    individual MLP for each task.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. Not used, as the dimension is
            determined by :obj:`num_node_targets` and :obj:`num_graph_targets`
            instead.
        num_node_targets (int): Number of individual PSEs used as node-level
            targets in pretraining :class:`GPSE`.
        num_graph_targets (int): Number of graph-level targets used in
            pretraining :class:`GPSE`.
        layers_post_mp (int): Number of MLP layers after GNN message-passing.
        virtual_node (bool, optional): Whether a virtual node is added to
            graphs in :class:`GPSE` computation. (default: :obj:`True`)
        multi_head_dim_inner (int, optional): Width of MLPs for PSE target
            prediction heads. (default: :obj:`32`)
        graph_pooling (str, optional): Type of graph pooling applied before
            post_mp. Options are :obj:`add`, :obj:`max`, :obj:`mean`.
            (default: :obj:`add`)
        has_bn (bool, optional): Whether to apply batch normalization to layer
            outputs. (default: :obj:`True`)
        bn_eps (float, optional): Epsilon for batch normalization.
            (default: :obj:`1.0e-05`)
        bn_mom (float, optional): Momentum for batch normalization.
            (default: :obj:`0.1`)
        has_l2norm (bool, optional): Wheter to apply L2 normalization to the
            layer outputs. (default: :obj:`True`)
        dropout (float, optional): Dropout ratio at layer output.
            (default: :obj:`0.2`)
        act (str, optional): Activation to apply to layer outputs if
            :obj:`has_act` is :obj:`True`. (default: :obj:`relu`)
    """
    def __init__(self, dim_in: int, dim_out: int, num_node_targets: int,
                 num_graph_targets: int, layers_post_mp: int,
                 virtual_node: bool = True, multi_head_dim_inner: int = 32,
                 graph_pooling: str = 'add', has_bn: bool = True,
                 bn_eps: float = 1.0e-05, bn_mom: float = 0.1,
                 has_l2norm: bool = True, dropout: float = 0.2,
                 act: str = 'relu'):
        super().__init__()
        pool_dict = {
            'add': global_add_pool,
            'max': global_max_pool,
            'mean': global_mean_pool
        }
        self.node_target_dim = num_node_targets
        self.graph_target_dim = num_graph_targets
        self.virtual_node = virtual_node
        num_layers = layers_post_mp

        self.node_post_mps = nn.ModuleList([
            MLP(dim_in, 1, multi_head_dim_inner, num_layers, has_bn, bn_eps,
                bn_mom, has_l2norm, dropout, act)
            for _ in range(self.node_target_dim)
        ])

        self.graph_pooling = pool_dict[graph_pooling]

        self.graph_post_mp = MLP(dim_in, self.graph_target_dim, dim_in,
                                 num_layers, has_bn, bn_eps, bn_mom,
                                 has_l2norm, dropout, act)

    def _pad_and_stack(self, x1: torch.Tensor, x2: torch.Tensor, pad1: int,
                       pad2: int):
        padded_x1 = nn.functional.pad(x1, (0, pad2))
        padded_x2 = nn.functional.pad(x2, (pad1, 0))
        return torch.vstack([padded_x1, padded_x2])

    def _apply_index(self, batch, virtual_node: bool, pad_node: int,
                     pad_graph: int):
        graph_pred, graph_true = batch.graph_feature, batch.y_graph
        node_pred, node_true = batch.node_feature, batch.y
        if virtual_node:
            # Remove virtual node
            idx = torch.concat([
                torch.where(batch.batch == i)[0][:-1]
                for i in range(batch.batch.max().item() + 1)
            ])
            node_pred, node_true = node_pred[idx], node_true[idx]

        # Stack node predictions on top of graph predictions and pad with zeros
        pred = self._pad_and_stack(node_pred, graph_pred, pad_node, pad_graph)
        true = self._pad_and_stack(node_true, graph_true, pad_node, pad_graph)

        return pred, true

    def forward(self, batch):
        batch.node_feature = torch.hstack(
            [m(batch.x) for m in self.node_post_mps])
        graph_emb = self.graph_pooling(batch.x, batch.batch)
        batch.graph_feature = self.graph_post_mp(graph_emb)
        return self._apply_index(batch, self.virtual_node,
                                 self.node_target_dim, self.graph_target_dim)


def GNNLayer(name: str, dim_in: int, dim_out: int, has_bn: bool = True,
             bn_eps: float = 1.0e-05, bn_mom: float = 0.1,
             has_l2norm: bool = True, dropout: float = 0.2,
             has_act: bool = True, act: str = 'relu', **kwargs):
    r"""Wrapper for a single GNN layer.

    Args:
        name (str): Name of the layer in registered :obj:`layer_dict`
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_bn (bool, optional): Whether to apply batch normalization to layer
            outputs. (default: :obj:`True`)
        bn_eps (float, optional): Epsilon for batch normalization.
            (default: :obj:`1.0e-05`)
        bn_mom (float, optional): Momentum for batch normalization.
            (default: :obj:`0.1`)
        has_l2norm (bool, optional): Whether to apply L2 normalization to the
            layer outputs. (default: :obj:`True`)
        dropout (float, optional): Dropout ratio at layer output.
            (default: :obj:`0.2`)
        has_act (bool, optional): Whether the layer has activation.
            (default: :obj:`True`)
        act (str, optional): Activation to apply to layer output if
            :obj:`has_act` is :obj:`True`. (default: :obj:`relu`)
        **kwargs (optional): Additional args for :class:`GeneralLayer`.

    Returns:
        GeneralLayer: Instance of GeneralLayer with specified configurations.
    """
    return GeneralLayer(name, dim_in, dim_out, has_bn, bn_eps, bn_mom,
                        has_l2norm, dropout, has_act, act, **kwargs)


def GNNPreMP(dim_in: int, dim_out: int, dim_inner: int, num_layers: int,
             has_bn: bool = True, bn_eps: float = 1.0e-05, bn_mom: float = 0.1,
             has_l2norm: bool = True, dropout: float = 0.2,
             final_act: bool = True, act: str = 'relu', **kwargs):
    r"""Wrapper for multiple NN layers before GNN message passing.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        dim_inner (int): Dimension of the inner layers
        num_layers (int): Number of layers
        has_bn (bool, optional): Whether to apply batch normalization to layer
            outputs. (default: :obj:`True`)
        bn_eps (float, optional): Epsilon for batch normalization.
            (default: :obj:`1.0e-05`)
        bn_mom (float, optional): Momentum for batch normalization.
        (default: :obj:`0.1`)
        has_l2norm (bool, optional): Whether to apply L2 normalization to the
            layer outputs. (default: :obj:`True`)
        dropout (float, optional): Dropout ratio at layer output.
            (default: :obj:`0.2`)
        final_act (bool, optional): Whether to apply an activation after the
            final layer. (default: :obj:`True`)
        act (str, optional): Activation to apply to layer output if
            :obj:`has_act` is :obj:`True`. (default: :obj:`relu`)
        **kwargs (optional): Additional args for :class:`GeneralMultiLayer`.

    Returns:
        GeneralMultiLayer: Instance of GeneralMultiLayer with specified
            configurations.
    """
    return GeneralMultiLayer('linear', dim_in, dim_out, dim_inner, num_layers,
                             has_bn, bn_eps, bn_mom, has_l2norm, dropout,
                             final_act, act, **kwargs)


class IdentityHead(nn.Module):
    r"""A placeholder prediction head that only returns batch.x and
    batch.y.
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        return batch.x, batch.y


class GNNStackStage(nn.Module):
    r"""Stacks a number of GNN layers, alongside skip connections and L2
    normalization.

    Args:
        dim_in (int): The input dimension.
        dim_out (int): The output dimension.
        num_layers (int): The number of layers.
        layer_type (str): The type of GNN layer to use.
        stage_type (str, optional): The type of staging to apply. Possible
            values are: :obj:`skipsum`, :obj:`skipconcat`. Any other value will
            default to no skip connections. (default: 'skipsum')
        final_l2norm (bool, optional): Whether to apply L2 normalization to the
            outputs. (default: :obj:`True`)
        has_bn (bool, optional): Whether to apply batch normalization to layer
            outputs. (default: :obj:`True`)
        bn_eps (float, optional): Epsilon for batch normalization.
            (default: :obj:`1.0e-05`)
        bn_mom (float, optional): Momentum for batch normalization.
            (default: :obj:`0.1`)
        has_l2norm (bool, optional): Whether the layer has L2 normalization.
            (default: :obj:`True`)
        dropout (float, optional): Dropout ratio at layer output.
            (default: :obj:`0.2`)
        has_act (bool, optional): Whether the layer has activation.
            (default: :obj:`True`)
        act (str, optional): Activation to apply to layer output if
            :obj:`has_act` is :obj:`True`. (default: :obj:`relu`)
    """
    def __init__(self, dim_in: int, dim_out: int, num_layers: int,
                 layer_type: str, stage_type: str = 'skipsum',
                 final_l2norm: bool = True, has_bn: bool = True,
                 bn_eps: float = 1.0e-05, bn_mom: float = 0.1,
                 has_l2norm: bool = True, dropout: float = 0.2,
                 has_act: bool = True, act: str = 'relu'):
        super().__init__()
        self.num_layers = num_layers
        self.stage_type = stage_type
        self.l2norm = final_l2norm
        for i in range(num_layers):
            if stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(layer_type, d_in, dim_out, has_bn, bn_eps, bn_mom,
                             has_l2norm, dropout, has_act, act)
            self.add_module(f'layer{i}', layer)

    def forward(self, batch):
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if self.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif (self.stage_type == 'skipconcat' and i < self.num_layers - 1):
                batch.x = torch.cat([x, batch.x], dim=1)
        if self.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch


class FeatureEncoder(nn.Module):
    r"""Encoder for :class:`GPSE` module that embeds raw node and edge
    features. Typically not used on downstream :class:`GPSE` usage.

    Args:
        dim_in (int): Input feature dimension.
        dim_inner (int): Width of the encoder.
        node_encoder (bool): Whether to use a node encoder.
        node_encoder_name (str): Name of the node encoder, depends on the
            pre-training dataset (e.g. pre-training on ogbg-molpcba requires an
            AtomEncoder).
        node_encoder_bn (bool): Whether to use batch normalization in the node
            encoder.
        edge_encoder (bool): Whether to use an edge encoder.
        edge_encoder_name (str): Name of the edge encoder, depends on the
            pre-training dataset (e.g. pre-training on ogbg-molpcba requires a
            BondEncoder).
        edge_encoder_bn (bool): Whether to use batch normalization in the edge
            encoder.
        bn_eps (float): Epsilon for batch normalization.
        bn_mom (float): Momentum for batch normalization.
    """
    def __init__(self, dim_in: int, dim_inner: int, node_encoder: bool,
                 node_encoder_name: str, node_encoder_bn: bool,
                 edge_encoder: bool, edge_encoder_name: str,
                 edge_encoder_bn: bool, bn_eps: float, bn_mom: float):
        super().__init__()
        node_encoder_dict = {
            'Integer': IntegerFeatureEncoder,
            'Atom': AtomEncoder
        }
        edge_encoder_dict = {'Bond': BondEncoder}
        self.dim_in = dim_in
        if node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = node_encoder_dict[node_encoder_name]
            self.node_encoder = NodeEncoder(dim_inner)
            if node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(dim_inner, bn_eps,
                                                       bn_mom)
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = dim_inner
        if edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = edge_encoder_dict[edge_encoder_name]
            self.edge_encoder = EdgeEncoder(dim_inner)
            if edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(dim_inner, bn_eps,
                                                       bn_mom)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class GPSE(nn.Module):
    r"""The Graph Positional and Structural Encoder (GPSE) model from the
    `"Graph Positional and Structural Encoder"
    <https://arxiv.org/abs/2307.07107>`_ paper.
    The GPSE model consists of (1) an (optional) encoder, (2) a deep GNN that
    consists of stacked message-passing layers, (3) a prediction head to
    predict pre-computed positional and structural encodings (PSE). When used
    on downstream datasets, these prediction heads are removed and the final
    fully-connected layer outputs are used as learned PSE embeddings.

    GPSE also provides a static method :obj:`from_pretrained` to load
    pre-trained GPSE models trained on a variety of molecular datasets. One can
    then use the pre-trained model to pre-compute GPSE encodings for a given
    dataset using the :obj:`precompute_gpse` method. Alternatively, the model
    can be used to generate GPSE encodings as part of a transform or
    pre_transform in a PyG dataset.

    .. code-block:: python

        from torch_geometric.nn import GPSE, GPSENodeEncoder,
        from torch_geometric.transforms import AddGPSE
        from torch_geometric.nn.models.gpse import precompute_GPSE

        gpse_model = GPSE.from_pretrained('molpcba')

        # Option 1: Precompute GPSE encodings in-place for a given dataset
        dataset = ZINC(path, subset=True, split='train')
        precompute_gpse(gpse_model, dataset)

        # Option 2: Use the GPSE model with AddGPSE as a pre_transform to save
        # the encodings
        dataset = ZINC(path, subset=True, split='train',
                       pre_transform=AddGPSE(gpse_model, vn=True,
                       rand_type='NormalSE'))

    Both approaches append the generated encodings to the :obj:`pestat_GPSE`
    attribute of :class:`~torch_geometric.data.Data` objects. To use the GPSE
    encodings for a downstream task, one may need to add these encodings to the
    :obj:`x` attribute of the :class:`~torch_geometric.data.Data` objects. To
    do so, one can use the :class:`GPSENodeEncoder` provided to map these
    encodings to a desired dimension before appending them to :obj:`x`.

    Let's say we have a graph dataset with 64 original node features, and we
    have generated  GPSE encodings of dimension 32, i.e.
    :obj:`data.pestat_GPSE` = 32. Additionally, we want to use a GNN with an
    inner dimension of 128. To do so, we can map the 32-dimensional GPSE
    encodings to a higher dimension of 64, and then append them to the :obj:`x`
    attribute of the :class:`~torch_geometric.data.Data` objects to obtain a
    128-dimensional node feature representation.
    :class:`~torch_geometric.nn.GPSENodeENcoder` handles both this mapping and
    concatenation to :obj:`x`, the outputs of which can be used as input to a
    GNN:

    .. code-block:: python

        encoder = GPSENodeEncoder(dim_emb=128, dim_pe_in=32, dim_pe_out=64,
                                  expand_x=False)
        gnn = GNN(dim_in=128, dim_out=128, num_layers=4)

        for batch in loader:
            batch = encoder(batch)
            batch = gnn(batch)
            # Do something with the batch, which now includes 128-dimensional
            # node representations


    Args:
        dim_in (int, optional): Input dimension. (default: :obj:`20`)
        dim_out (int, optional): Output dimension. (default: :obj:`51`)
        dim_inner (int, optional): Width of the encoder layers.
            (default: :obj:`512`)
        layer_type (str, optional): Type of graph convolutional layer for
            message-passing. (default: :obj:`resgatedgcnconv`)
        layers_pre_mp (int, optional): Number of MLP layers before
            message-passing. (default: :obj:`1`)
        layers_mp (int, optional): Number of layers for message-passing.
            (default: :obj:`20`)
        layers_post_mp (int, optional): Number of MLP layers after
            message-passing. (default: :obj:`2`)
        num_node_targets (int, optional): Number of individual PSEs used as
            node-level targets in pretraining :class:`GPSE`.
            (default: :obj:`51`)
        num_graph_targets (int, optional): Number of graph-level targets used
            in pretraining :class:`GPSE`. (default: :obj:`11`)
        node_encoder (bool, optional): Whether to use a node encoder.
            (default: :obj:`False`)
        node_encoder_name (str, optional): Name of the node encoder, depends on
            the pre-training dataset (e.g. pre-training on ogbg-molpcba
            requires an AtomEncoder). (default: :obj:`Atom`)
        node_encoder_bn (bool, optional): Whether to use batch normalization in
        the node encoder. (default: :obj:`False`)
        edge_encoder (bool, optional): Whether to use an edge encoder.
            (default: :obj:`False`)
        edge_encoder_name (str, optional): Name of the edge encoder, depends on
            the pre-training dataset (e.g. pre-training on ogbg-molpcba
            requires a BondEncoder). (default: :obj:`Bond`)
        edge_encoder_bn (bool, optional): Whether to use batch normalization in
            the edge encoder. (default: :obj:`False`)
        stage_type (str, optional): The type of staging to apply. Possible
            values are: :obj:`skipsum`, :obj:`skipconcat`. Any other value will
            default to no skip connections. (default: :obj:`skipsum`)
        has_bn (bool, optional): Whether to apply batch normalization in the
            layer. (default: :obj:`True`)
        bn_eps (float, optional): Epsilon for batch normalization.
            (default: :obj:`1.0e-05`)
        bn_mom (float, optional): Momentum for batch normalization.
            (default: :obj:`0.1`)
        final_l2norm (bool, optional): Whether to apply L2 normalization to the
            outputs. (default: :obj:`True`)
        has_l2norm (bool, optional): Whether to apply L2 normalization after
        the layer. (default: :obj:`True`)
        dropout (float, optional): Dropout ratio at layer output.
            (default: :obj:`0.2`)
        has_act (bool, optional): Whether has activation after the layer.
            (default: :obj:`True`)
        final_act (bool, optional): Whether to apply activation after the layer
            stack. (default: :obj:`True`)
        act (str, optional): Activation to apply to layer output if
            :obj:`has_act` is :obj:`True`. (default: :obj:`relu`)
        virtual_node (bool, optional): Whether a virtual node is added to
            graphs in :class:`GPSE` computation. (default: :obj:`True`)
        multi_head_dim_inner (int, optional): Width of MLPs for PSE target
            prediction heads. (default: :obj:`32`)
        graph_pooling (str, optional): Type of graph pooling applied before
            post_mp. Options are :obj:`add`, :obj:`max`, :obj:`mean`.
            (default: :obj:`add`)
        use_repr (bool, optional): Whether to use the hidden representation of
            the final layer as :class:`GPSE` encodings. (default: :obj:`True`)
        repr_type (str, optional): Type of representation to use. Options are
            :obj:`no_post_mp`, :obj:`one_layer_before`.
            (default: :obj:`no_post_mp`)
        bernoulli_threshold (float, optional): Threshold for Bernoulli sampling
        of virtual nodes. (default: :obj:`0.5`)
    """

    url_dict = {
        'molpcba':
        'https://zenodo.org/record/8145095/files/'
        'gpse_model_molpcba_1.0.pt',
        'zinc':
        'https://zenodo.org/record/8145095/files/gpse_model_zinc_1.0.pt',
        'pcqm4mv2':
        'https://zenodo.org/record/8145095/files/'
        'gpse_model_pcqm4mv2_1.0.pt',
        'geom':
        'https://zenodo.org/record/8145095/files/gpse_model_geom_1.0.pt',
        'chembl':
        'https://zenodo.org/record/8145095/files/gpse_model_chembl_1.0.pt'
    }

    def __init__(
            self, dim_in: int = 20, dim_out: int = 51, dim_inner: int = 512,
            layer_type: str = 'resgatedgcnconv', layers_pre_mp: int = 1,
            layers_mp: int = 20, layers_post_mp: int = 2,
            num_node_targets: int = 51, num_graph_targets: int = 11,
            node_encoder: bool = False, node_encoder_name: str = 'Atom',
            node_encoder_bn: bool = False, edge_encoder: bool = False,
            edge_encoder_name: str = 'Bond', edge_encoder_bn: bool = False,
            stage_type: str = 'skipsum', has_bn: bool = True,
            head_bn: bool = False, bn_eps: float = 1.0e-05,
            bn_mom: float = 0.1, final_l2norm: bool = True,
            has_l2norm: bool = True, dropout: float = 0.2,
            has_act: bool = True, final_act: bool = True, act: str = 'relu',
            virtual_node: bool = True, multi_head_dim_inner: int = 32,
            graph_pooling: str = 'add', use_repr: bool = True,
            repr_type: str = 'no_post_mp', bernoulli_threshold: float = 0.5):

        super().__init__()
        self.use_repr = use_repr
        self.repr_type = repr_type
        self.bernoulli_threshold = bernoulli_threshold

        self.encoder = FeatureEncoder(dim_in, dim_inner, node_encoder,
                                      node_encoder_name, node_encoder_bn,
                                      edge_encoder, edge_encoder_name,
                                      edge_encoder_bn, bn_eps, bn_mom)
        dim_in = self.encoder.dim_in

        if layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, dim_inner, dim_inner, layers_pre_mp,
                                   has_bn, bn_eps, bn_mom, has_l2norm, dropout,
                                   final_act, act)
            dim_in = dim_inner
        if layers_mp > 0:
            self.mp = GNNStackStage(dim_in, dim_inner, layers_mp, layer_type,
                                    stage_type, final_l2norm, has_bn, bn_eps,
                                    bn_mom, has_l2norm, dropout, has_act, act)
        self.post_mp = GNNInductiveHybridMultiHead(
            dim_inner, dim_out, num_node_targets, num_graph_targets,
            layers_post_mp, virtual_node, multi_head_dim_inner, graph_pooling,
            head_bn, bn_eps, bn_mom, has_l2norm, dropout, act)

        self.apply(init_weights)

    def reset_parameters(self):
        self.apply(init_weights)

    @classmethod
    def from_pretrained(cls, name: str, root: str = 'GPSE_pretrained'):
        r"""Returns a :class:`GPSE` model pre-trained on the dataset registered
        in :obj:`param_dict` under :obj:`name`.

        Args:
            name (str): The name of the dataset to pre-train on. Options are:
                :obj:`molpcba`, :obj:`zinc`, :obj:`pcqm4mv2`, :obj:`geom`,
                :obj:`chembl`.
            root (str, optional): Root directory to save the pre-trained model.
                (default: :obj:`'GPSE_pretrained'`)
        """
        root = osp.expanduser(osp.normpath(root))
        os.makedirs(root, exist_ok=True)
        path = download_url(cls.url_dict[name], root)

        model = GPSE()  # All pretrained models use the default arguments
        model_state = torch.load(path, map_location='cpu')['model_state']
        model_state_new = OrderedDict([(k.split('.', 1)[1], v)
                                       for k, v in model_state.items()])
        model.load_state_dict(model_state_new)
        # Set the final linear layer to identity if we use hidden reprs
        if model.use_repr:
            if model.repr_type == 'one_layer_before':
                model.post_mp.layer_post_mp.model[-1] = torch.nn.Identity()
            elif model.repr_type == 'no_post_mp':
                model.post_mp = IdentityHead()
            else:
                raise ValueError(f'Unknown repr_type {model.repr_type}')
        model.eval()
        return model

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class VirtualNodePatchSingleton(T.VirtualNode):
    r"""PyG transform that appends a virtual node to a single data object, with
    better handling of singleton graphs with no edges than
    :class:`~torch_geometric.transforms.VirtualNode`.

    Args:
        data (torch_geometric.data.Data): The
        :class:`~torch_geometric.data.Data` object to add VN to.
    """
    def __call__(self, data: Data):
        if data.edge_index.numel() == 0:
            data.edge_index, data.edge_attr = add_self_loops(
                data.edge_index, data.edge_attr, num_nodes=data.num_nodes)
            data = super().__call__(data)
            if hasattr(data, "y_graph"):  # potentially fix hybrid head
                data.y_graph = data.y_graph[:1]
            data.edge_index, data.edge_attr = remove_self_loops(
                data.edge_index, data.edge_attr)
        else:
            data = super().__call__(data)
        return data


class GPSENodeEncoder(nn.Module):
    r"""A helper linear/MLP encoder that takes the :class:`GPSE` encodings
    (based on the `"Graph Positional and Structural Encoder"
    <https://arxiv.org/abs/2307.07107>`_ paper) precomputed as
    :obj:`batch.pestat_GPSE` in the input graphs, maps them to a desired
    dimension defined by :obj:`dim_pe_out` and appends them to node features.

    Let's say we have a graph dataset with 64 original node features, and we
    have generated GPSE encodings of dimension 32, i.e.
    :obj:`data.pestat_GPSE` = 32. Additionally, we want to use a GNN with an
    inner dimension of 128. To do so, we can map the 32-dimensional GPSE
    encodings to a higher dimension of 64, and then append them to the
    :obj:`x` attribute of the :class:`~torch_geometric.data.Data` objects to
    obtain a 128-dimensional node feature representation.
    :class:`~torch_geometric.nn.GPSENodeENcoder` handles both this mapping and
    concatenation to :obj:`x`, the outputs of which can be used as input to a
    GNN:

    .. code-block:: python

        encoder = GPSENodeEncoder(dim_emb=128, dim_pe_in=32, dim_pe_out=64,
                                  expand_x=False)
        gnn = GNN(dim_in=128, dim_out=128, num_layers=4)

        for batch in loader:
            batch = encoder(batch)
            batch = gnn(batch)
            # Do something with the batch, which now includes 128-dimensional
            # node representations

    Args:
        dim_emb (int): Size of final node embedding.
        dim_pe_in (int): Original dimension of :obj:`batch.pestat_GPSE`.
        dim_pe_out (int): Desired dimension of :class:`GPSE` after the encoder.
        dim_in (int, optional): Original dimension of input node features,
            required only if :obj:`expand_x` is set to :obj:`True`.
            (default: :obj:`None`)
        expand_x (bool, optional): Expand node features :obj:`x` from
            :obj:`dim_in` to (:obj:`dim_emb` - :obj:`dim_pe`)
        norm_type (str, optional): Type of normalization to apply.
            (default: :obj:`batchnorm`)
        model_type (str, optional): Type of encoder, either :obj:`mlp` or
            :obj:`linear`. (default: :obj:`mlp`)
        n_layers (int, optional): Number of MLP layers if :obj:`model_type` is
            :obj:`mlp`. (default: :obj:`2`)
        dropout_be (float, optional): Dropout ratio of inputs to encoder, i.e.
            before encoding. (default: :obj:`0.5`)
        dropout_ae (float, optional): Dropout ratio of outputs, i.e. after
            encoding. (default: :obj:`0.2`)
    """
    def __init__(self, dim_emb: int, dim_pe_in: int, dim_pe_out: int,
                 dim_in: int = None, expand_x=False, norm_type='batchnorm',
                 model_type='mlp', n_layers=2, dropout_be=0.5, dropout_ae=0.2):
        super().__init__()

        assert dim_emb > dim_pe_out, ('Desired GPSE dimension (dim_pe_out) '
                                      'must be smaller than the final node '
                                      'embedding dimension (dim_emb).')

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe_out)
        self.expand_x = expand_x

        self.raw_norm = None
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(dim_pe_in)

        self.dropout_be = nn.Dropout(p=dropout_be)
        self.dropout_ae = nn.Dropout(p=dropout_ae)

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(dim_pe_in, dim_pe_out))
                layers.append(activation())
            else:
                layers.append(nn.Linear(dim_pe_in, 2 * dim_pe_out))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe_out, 2 * dim_pe_out))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe_out, dim_pe_out))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(dim_pe_in, dim_pe_out)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        if not hasattr(batch, 'pestat_GPSE'):
            raise ValueError('Precomputed "pestat_GPSE" variable is required '
                             'for GNNNodeEncoder; either run '
                             '`precompute_GPSE(gpse_model, dataset)` on your '
                             'dataset or add `AddGPSE(gpse_model)` as a (pre) '
                             'transform.')

        pos_enc = batch.pestat_GPSE

        pos_enc = self.dropout_be(pos_enc)
        pos_enc = self.raw_norm(pos_enc) if self.raw_norm else pos_enc
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe
        pos_enc = self.dropout_ae(pos_enc)

        # Expand node features if needed
        h = self.linear_x(batch.x) if self.expand_x else batch.x

        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)

        return batch


@torch.no_grad()
def gpse_process(model: GPSE, data: Data, rand_type: str, use_vn: bool = True,
                 bernoulli_thresh: float = 0.5, neighbor_loader: bool = False,
                 num_neighbors: List[int] = [30, 20, 10], fillval: int = 5,
                 layers_mp: int = None, **kwargs) -> torch.Tensor:
    r"""Processes the data using the :class:`GPSE` model to generate and append
    GPSE encodings. Identical to :obj:`gpse_process_batch`, but operates on a
    single :class:`~torch_geometric.data.Dataset` object.

    Unlike transform-based GPSE processing (i.e.
    :class:`~torch_geometric.transforms.AddGPSE`), the :obj:`use_vn` argument
    does not append virtual nodes if set to :obj:`True`, and instead assumes
    the input graphs to :obj:`gpse_process` already have virtual nodes. Under
    normal circumstances, one does not need to call this function; running
    :obj:`precompute_GPSE` on your whole dataset is advised instead.

    Args:
        model (GPSE): The :class:`GPSE` model.
        data (torch_geometric.data.Data): A :class:`~torch_geometric.data.Data`
            object.
        rand_type (str, optional): Type of random features to use. Options are
            :obj:`NormalSE`, :obj:`UniformSE`, :obj:`BernoulliSE`.
            (default: :obj:`NormalSE`)
        use_vn (bool, optional): Whether the input graphs have virtual nodes.
            (default: :obj:`True`)
        bernoulli_thresh (float, optional): Threshold for Bernoulli sampling of
            virtual nodes. (default: :obj:`0.5`)
        neighbor_loader (bool, optional): Whether to use :obj:`NeighborLoader`.
            (default: :obj:`False`)
        num_neighbors (List[int], optional): Number of neighbors to consider
            for each message-passing layer. (default: :obj:`[30, 20, 10]`)
        fillval (int, optional): Value to fill for missing
            :obj:`num_neighbors`. (default: :obj:`5`)
        layers_mp (int, optional): Number of message-passing layers.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments for :obj:`NeighborLoader`.

    Returns:
        torch.Tensor: A tensor corresponding to the original
        :class:`~torch_geometric.data.Data` object, with :class:`GPSE`
        encodings appended as :obj:`out.pestat_GPSE` attribute.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Generate random features for the encoder
    n = data.num_nodes
    dim_in = model.state_dict()[list(model.state_dict())[0]].shape[1]

    # Prepare input distributions for GPSE
    if rand_type == 'NormalSE':
        rand = np.random.normal(loc=0, scale=1.0, size=(n, dim_in))
    elif rand_type == 'UniformSE':
        rand = np.random.uniform(low=0.0, high=1.0, size=(n, dim_in))
    elif rand_type == 'BernoulliSE':
        rand = np.random.uniform(low=0.0, high=1.0, size=(n, dim_in))
        rand = (rand < bernoulli_thresh)
    else:
        raise ValueError(f'Unknown {rand_type=!r}')
    data.x = torch.from_numpy(rand.astype('float32'))

    if use_vn:
        data.x[-1] = 0

    model, data = model.to(device), data.to(device)
    # Generate encodings using the pretrained encoder
    if neighbor_loader:
        if layers_mp is None:
            raise ValueError('Please provide the number of message-passing '
                             'layers as "layers_mp".')
        diff = layers_mp - len(num_neighbors)
        if fillval > 0 and diff > 0:
            num_neighbors += [fillval] * diff

        loader = NeighborLoader(data, num_neighbors=num_neighbors,
                                shuffle=False, pin_memory=True, **kwargs)
        out_list = []
        pbar = trange(data.num_nodes, position=2)
        for i, batch in enumerate(loader):
            out, _ = model(batch.to(device))
            out = out[:batch.batch_size].to("cpu", non_blocking=True)
            out_list.append(out)
            pbar.update(batch.batch_size)
        out = torch.vstack(out_list)
    else:
        out, _ = model(data)
        out = out.to("cpu")

    return out


@torch.no_grad()
def gpse_process_batch(model: GPSE, batch, rand_type: str, use_vn: bool = True,
                       bernoulli_thresh: float = 0.5,
                       neighbor_loader: bool = False,
                       num_neighbors: List[int] = [30, 20, 10],
                       fillval: int = 5, layers_mp: int = None,
                       **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Process a batch of data using the :class:`GPSE` model to generate and
    append :class:`GPSE` encodings. Identical to `gpse_process`, but operates
    on a batch of :class:`~torch_geometric.data.Data` objects.

    Unlike transform-based GPSE processing (i.e.
    :class:`~torch_geometric.transforms.AddGPSE`), the :obj:`use_vn` argument
    does not append virtual nodes if set to :obj:`True`, and instead assumes
    the input graphs to :obj:`gpse_process` already have virtual nodes. This is
    because the virtual nodes are already added to graphs before the call to
    :obj:`gpse_process_batch` in :obj:`precompute_GPSE` for better efficiency.
    Under normal circumstances, one does not need to call this function;
    running :obj:`precompute_GPSE` on your whole dataset is advised instead.

    Args:
        model (GPSE): The :class:`GPSE` model.
        batch: A batch of PyG Data objects.
        rand_type (str, optional): Type of random features to use. Options are
            :obj:`NormalSE`, :obj:`UniformSE`, :obj:`BernoulliSE`.
            (default: :obj:`NormalSE`)
        use_vn (bool, optional): Whether the input graphs have virtual nodes.
            (default: :obj:`True`)
        bernoulli_thresh (float, optional): Threshold for Bernoulli sampling of
            virtual nodes. (default: :obj:`0.5`)
        neighbor_loader (bool, optional): Whether to use :obj:`NeighborLoader`.
            (default: :obj:`False`)
        num_neighbors (List[int], optional): Number of neighbors to consider
            for each message-passing layer. (default: :obj:`[30, 20, 10]`)
        fillval (int, optional): Value to fill for missing
            :obj:`num_neighbors`. (default: :obj:`5`)
        layers_mp (int, optional): Number of message-passing layers.
            (default: :obj:`None`)
        **kwargs: Additional keyword arguments for :obj:`NeighborLoader`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A two-tuple of tensors corresponding
            to the stacked :class:`GPSE` encodings and the pointers indicating
            individual graphs.
    """
    n = batch.num_nodes
    dim_in = model.state_dict()[list(model.state_dict())[0]].shape[1]

    # Prepare input distributions for GPSE
    if rand_type == 'NormalSE':
        rand = np.random.normal(loc=0, scale=1.0, size=(n, dim_in))
    elif rand_type == 'UniformSE':
        rand = np.random.uniform(low=0.0, high=1.0, size=(n, dim_in))
    elif rand_type == 'BernoulliSE':
        rand = np.random.uniform(low=0.0, high=1.0, size=(n, dim_in))
        rand = (rand < bernoulli_thresh)
    else:
        raise ValueError(f'Unknown {rand_type=!r}')
    batch.x = torch.from_numpy(rand.astype('float32'))

    if use_vn:
        # HACK: We need to reset virtual node features to zeros to match the
        # pretraining setting (virtual node applied after random node features
        # are set, and the default node features for the virtual node are all
        # zeros). Can potentially test if initializing virtual node features to
        # random features is better than setting them to zeros.
        for i in batch.ptr[1:]:
            batch.x[i - 1] = 0

    # Generate encodings using the pretrained encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if neighbor_loader:
        if layers_mp is None:
            raise ValueError('Please provide the number of message-passing '
                             'layers as "layers_mp".')
        diff = layers_mp - len(num_neighbors)
        if fillval > 0 and diff > 0:
            num_neighbors += [fillval] * diff

        loader = NeighborLoader(batch, num_neighbors=num_neighbors,
                                shuffle=False, pin_memory=True, **kwargs)
        out_list = []
        pbar = trange(batch.num_nodes, position=2)
        for i, batch in enumerate(loader):
            out, _ = model(batch.to(device))
            out = out[:batch.batch_size].to('cpu', non_blocking=True)
            out_list.append(out)
            pbar.update(batch.batch_size)
        out = torch.vstack(out_list)
    else:
        out, _ = model(batch.to(device))
        out = out.to('cpu')

    return out, batch.ptr


@torch.no_grad()
def precompute_GPSE(model: GPSE, dataset: Dataset, use_vn: bool = True,
                    rand_type: str = 'NormalSE', **kwargs):
    r"""Precomputes :class:`GPSE` encodings in-place for a given dataset using
    a :class:`GPSE` model.

    Args:
        model (GPSE): The :class:`GPSE` model.
        dataset (Dataset): A PyG Dataset.
        use_vn (bool, optional): Whether to append virtual nodes to graphs in
            :class:`GPSE` computation. Should match the setting used when
            pre-training the :class:`GPSE` model. (default :obj:`True`)
        rand_type (str, optional): The type of randomization to use.
            (default :obj:`NormalSE`)
        **kwargs (optional): Additional arguments for
            :class:`~torch_geometric.data.DataLoader`.
    """
    # Temporarily replace the transformation
    orig_dataset_transform = dataset.transform
    dataset.transform = None
    if use_vn:
        dataset.transform = VirtualNodePatchSingleton()

    # Remove split indices, to be recovered at the end of the precomputation
    tmp_store = {}
    for name in [
            'train_mask', 'val_mask', 'test_mask', 'train_graph_index',
            'val_graph_index', 'test_graph_index', 'train_edge_index',
            'val_edge_index', 'test_edge_index'
    ]:
        if (name in dataset.data) and (dataset.slices is None
                                       or name in dataset.slices):
            tmp_store_data = dataset.data.pop(name)
            tmp_store_slices = dataset.slices.pop(name) \
                if dataset.slices else None
            tmp_store[name] = (tmp_store_data, tmp_store_slices)

    loader = DataLoader(dataset, shuffle=False, pin_memory=True, **kwargs)

    # Batched GPSE precomputation loop
    data_list = []
    curr_idx = 0
    pbar = trange(len(dataset), desc='Pre-computing GPSE')
    tic = time.perf_counter()
    for batch in loader:
        batch_out, batch_ptr = gpse_process_batch(model, batch, rand_type,
                                                  **kwargs)

        batch_out = batch_out.to('cpu', non_blocking=True)
        # Need to wait for batch_ptr to finish transfering so that start and
        # end indices are ready to use
        batch_ptr = batch_ptr.to('cpu', non_blocking=False)

        for start, end in zip(batch_ptr[:-1], batch_ptr[1:]):
            data = dataset.get(curr_idx)
            if use_vn:
                end = end - 1
            data.pestat_GPSE = batch_out[start:end]
            data_list.append(data)
            curr_idx += 1

        pbar.update(len(batch_ptr) - 1)
    pbar.close()

    # Collate dataset and reset indicies and data list
    dataset.transform = orig_dataset_transform
    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)

    # Recover split indices
    for name, (tmp_store_data, tmp_store_slices) in tmp_store.items():
        dataset.data[name] = tmp_store_data
        if tmp_store_slices is not None:
            dataset.slices[name] = tmp_store_slices
    dataset._data_list = None

    timestr = time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - tic))
    logging.info(f'Finished GPSE pre-computation, took {timestr}')

    # Release resource and recover original configs
    del model
    torch.cuda.empty_cache()


def cosim_col_sep(pred: torch.Tensor, true: torch.Tensor,
                  batch_idx: torch.Tensor) -> torch.Tensor:
    r"""Calculates the average cosine similarity between predicted and true
    features on a batch of graphs.

    Args:
        pred (torch.Tensor): Predicted outputs.
        true (torch.Tensor): Value of ground truths.
        batch_idx (torch.Tensor): Batch indices to separate the graphs.

    Returns:
        torch.Tensor: Average cosine similarity per graph in batch.

    Raises:
        ValueError: If batch_index is not specified.
    """
    if batch_idx is None:
        raise ValueError("mae_cosim_col_sep requires batch index as "
                         "input to distinguish different graphs.")
    batch_idx = batch_idx + 1 if batch_idx.min() == -1 else batch_idx
    pred_dense = to_dense_batch(pred, batch_idx)[0]
    true_dense = to_dense_batch(true, batch_idx)[0]
    mask = (true_dense == 0).all(1)  # exclude trivial features from loss
    loss = 1 - F.cosine_similarity(pred_dense, true_dense, dim=1)[~mask].mean()
    return loss


def gpse_loss(pred: torch.Tensor, true: torch.Tensor,
              batch_idx: torch.Tensor = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculates :class:`GPSE` loss as the sum of MAE loss and cosine
    similarity loss over a batch of graphs.

    Args:
        pred (torch.Tensor): Predicted outputs.
        true (torch.Tensor): Value of ground truths.
        batch_idx (torch.Tensor): Batch indices to separate the graphs.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A two-tuple of tensors corresponding
        to the :class:`GPSE` loss and the predicted node-and-graph level
        outputs.

    """
    if batch_idx is None:
        raise ValueError("mae_cosim_col_sep requires batch index as "
                         "input to distinguish different graphs.")
    mae_loss = F.l1_loss(pred, true)
    cosim_loss = cosim_col_sep(pred, true, batch_idx)
    loss = mae_loss + cosim_loss
    return loss, pred


def process_batch_idx(batch_idx, true, use_vn=True):
    r"""Processes batch indices to adjust for the removal of virtual nodes, and
    pads batch index for hybrid tasks.

    Args:
        batch_idx: Batch indices to separate the graphs.
        true: Value of ground truths.
        use_vn: If input graphs have virtual nodes that need to be removed.

    Returns:
        torch.Tensor: Batch indices that separate the graphs.
    """
    if batch_idx is None:
        return
    if use_vn:  # remove virtual node
        batch_idx = torch.concat([
            batch_idx[batch_idx == i][:-1]
            for i in range(batch_idx.max().item() + 1)
        ])
    # Pad batch index for hybrid tasks (set batch index for graph heads to -1)
    if (pad := true.shape[0] - batch_idx.shape[0]) > 0:
        pad_idx = -torch.ones(pad, dtype=torch.long, device=batch_idx.device)
        batch_idx = torch.hstack([batch_idx, pad_idx])
    return batch_idx
