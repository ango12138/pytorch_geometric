import pytest
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import (
    GATConv,
    GCNConv,
    MessagePassing,
    PGExplainer,
    global_add_pool,
)


def assert_mask_clear(model):
    for layer in model.modules():
        if isinstance(layer, MessagePassing):
            assert ~layer._explain
            assert layer._edge_mask is None
            assert layer._loop_mask is None


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index, get_embedding=False):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        if get_embedding:
            return x
        return x.log_softmax(dim=1)


class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(3, 16, heads=2, concat=True)
        self.conv2 = GATConv(2 * 16, 7, heads=2, concat=False)

    def forward(self, x, edge_index, get_embedding=False):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        if get_embedding:
            return x
        return x.log_softmax(dim=1)


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)

    def forward(self, x, edge_index, batch, get_embedding=False):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        if get_embedding:
            return x
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return x.log_softmax(dim=1)


return_types = ['log_prob', 'regression']


@pytest.mark.parametrize('model', [GAT(), GCN()])
@pytest.mark.parametrize('return_type', return_types)
def test_node(model, return_type):

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    z = model(x, edge_index, get_embedding=True)

    explainer = PGExplainer(model, out_channels=z.shape[1], task='node',
                            log=False, return_type=return_type)
    assert repr(explainer) == (
        f'PGExplainer({model}'
        f', {z.shape[1]}, epochs=30, lr=0.003, task=node)')
    explainer.train_explainer(x, z, edge_index, node_idxs=torch.arange(0, 3))
    assert_mask_clear(model)

    mask = explainer.explain(x, z, edge_index, node_idx=1)
    assert mask.shape[0] == edge_index.shape[1]
    assert mask.max() <= 1 and mask.min() >= 0

    # test with isolated node:
    explainer = PGExplainer(model, out_channels=z.shape[1], task='node',
                            log=False, return_type=return_type)
    with torch.autograd.set_detect_anomaly(True):
        explainer.train_explainer(x[:4], z[:4], edge_index[:, :4],
                                  node_idxs=torch.arange(2, 4))
    assert_mask_clear(model)
    mask = explainer.explain(x[:2], z[:2], edge_index[:, :1], node_idx=1)
    assert mask.shape[0] == 1


@pytest.mark.parametrize('model', [GNN()])
@pytest.mark.parametrize('return_type', return_types)
def test_graph(model, return_type):
    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    z = model(x, edge_index, batch, get_embedding=True)

    explainer = PGExplainer(model, out_channels=z.shape[1], task='graph',
                            log=False, return_type=return_type)
    assert repr(explainer) == (
        f'PGExplainer({model}'
        f', {z.shape[1]}, epochs=30, lr=0.003, task=graph)')
    explainer.train_explainer(x, z, edge_index, batch=batch)
    assert_mask_clear(model)

    mask = explainer.explain(x, z, edge_index)
    assert mask.shape[0] == edge_index.shape[1]
    assert mask.max() <= 1 and mask.min() >= 0
