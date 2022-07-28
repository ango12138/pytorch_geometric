import pytest
import torch

import torch_geometric
from torch_geometric.nn.resolver import (
    activation_resolver,
    aggregation_resolver,
    normalization_resolver,
)


def test_activation_resolver():
    assert isinstance(activation_resolver(torch.nn.ELU()), torch.nn.ELU)
    assert isinstance(activation_resolver(torch.nn.ReLU()), torch.nn.ReLU)
    assert isinstance(activation_resolver(torch.nn.PReLU()), torch.nn.PReLU)

    assert isinstance(activation_resolver('elu'), torch.nn.ELU)
    assert isinstance(activation_resolver('relu'), torch.nn.ReLU)
    assert isinstance(activation_resolver('prelu'), torch.nn.PReLU)


@pytest.mark.parametrize('aggr_tuple', [
    (torch_geometric.nn.aggr.MeanAggregation(), 'mean', ()),
    (torch_geometric.nn.aggr.SumAggregation(), 'sum', ()),
    (torch_geometric.nn.aggr.MaxAggregation(), 'max', ()),
    (torch_geometric.nn.aggr.MinAggregation(), 'min', ()),
    (torch_geometric.nn.aggr.MulAggregation(), 'mul', ()),
    (torch_geometric.nn.aggr.VarAggregation(), 'var', ()),
    (torch_geometric.nn.aggr.StdAggregation(), 'std', ()),
    (torch_geometric.nn.aggr.SoftmaxAggregation(), 'softmax', ()),
    (torch_geometric.nn.aggr.PowerMeanAggregation(), 'power_mean', ()),
    (torch_geometric.nn.aggr.LSTMAggregation(6, 6), 'lstm', (6, 6)),
    (torch_geometric.nn.aggr.Set2Set(6, 6), 'set2set', (6, 6)),
])
def test_aggregation_resolver(aggr_tuple):
    aggr_module, aggr_repr, aggr_args = aggr_tuple
    aggr_cls = type(aggr_module)
    assert isinstance(aggregation_resolver(aggr_module), aggr_cls)
    assert isinstance(aggregation_resolver(aggr_repr, *aggr_args), aggr_cls)
    assert aggregation_resolver(aggr_module, reverse=True) == aggr_repr
    assert aggregation_resolver(aggr_repr, reverse=True) == aggr_repr


@pytest.mark.parametrize('norm_tuple', [
    (torch_geometric.nn.norm.BatchNorm, 'batch_norm', (16, )),
    (torch_geometric.nn.norm.InstanceNorm, 'instance_norm', (16, )),
    (torch_geometric.nn.norm.LayerNorm, 'layer_norm', (16, )),
    (torch_geometric.nn.norm.GraphNorm, 'graph_norm', (16, )),
    (torch_geometric.nn.norm.GraphSizeNorm, 'graphsize_norm', ()),
    (torch_geometric.nn.norm.PairNorm, 'pair_norm', ()),
    (torch_geometric.nn.norm.MessageNorm, 'message_norm', ()),
    (torch_geometric.nn.norm.DiffGroupNorm, 'diffgroup_norm', (16, 4)),
])
def test_normalization_resolver(norm_tuple):
    norm_module, norm_repr, norm_args = norm_tuple
    assert isinstance(normalization_resolver(norm_module(*norm_args)),
                      norm_module)
    assert isinstance(normalization_resolver(norm_repr, *norm_args),
                      norm_module)
