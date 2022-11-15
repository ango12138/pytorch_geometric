import pytest
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import (
    ExplainerConfig,
    MaskType,
    ModelConfig,
)
from torch_geometric.nn import GATConv, GCNConv, global_add_pool


# --------------------------------------------------------------------
# model for node level tasks
# --------------------------------------------------------------------
class GCN_multioutput_regression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)


class GCN_single_output_regression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)


class GCN_single_output_classification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)


class GCN_multioutput_classification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)
        self.conv2_2 = GCNConv(16, 7)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        out_1 = self.conv2(x, edge_index).log_softmax(dim=1)
        out_2 = self.conv2_2(x, edge_index).log_softmax(dim=1)
        return torch.stack([out_1, out_2], dim=0)


class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(3, 16, heads=2, concat=True)
        self.conv2 = GATConv(2 * 16, 7, heads=2, concat=False)

    def forward(
        self,
        x,
        edge_index,
        batch=None,
        **kwargs,
    ):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)


# --------------------------------------------------------------------
# model for graph level tasks
# --------------------------------------------------------------------


class GNN_regression_multioutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        return self.lin(x)


class GNN_regression_singleoutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        return self.lin(x)


class GNN_classification_singleoutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return x.log_softmax(dim=1)


class GNN_classification_multioutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)
        self.lin2 = Linear(16, 7)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        out_1 = self.lin(x).log_softmax(dim=1)
        out_2 = self.lin2(x).log_softmax(dim=1)
        return torch.stack([out_1, out_2], dim=0)


# --------------------------------------------------------------------
# check explanations
# --------------------------------------------------------------------
def check_explanation(
    edge_mask_type,
    node_mask_type,
    x,
    edge_index,
    explanation,
):
    if node_mask_type == MaskType.attributes:
        assert explanation.node_features_mask.shape == x.shape
        assert explanation.node_features_mask.min() >= 0
        assert explanation.node_features_mask.max() <= 1
    elif node_mask_type == MaskType.object:
        assert explanation.node_mask.shape == x.shape[0]
        assert explanation.node_mask.min() >= 0
        assert explanation.node_mask.max() <= 1

    if edge_mask_type == MaskType.object:
        assert explanation.edge_mask.shape == (edge_index.shape[1], )
        assert explanation.edge_mask.min() >= 0
        assert explanation.edge_mask.max() <= 1


# type of masks allowed for GNNExplainer
node_mask_types = ["attributes", "object", "common_attributes"]
edge_mask_types = ["object", None]
return_types_classification = ["log_probs", "raw", "probs"]
return_types_regression = ["raw"]


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [GCN_single_output_regression])
@pytest.mark.parametrize('return_type', return_types_regression)
def test_gnn_explainer_node_regression_single_output(
    edge_mask_type,
    node_mask_type,
    model,
    return_type,
):
    explainer_config = ExplainerConfig(
        explanation_type="model",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='node',
        return_type=return_type,
        mode="regression",
    )

    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index)

    explainer = GNNExplainer()
    # try to explain prediction for node 2
    explanation = explainer(
        x=x,
        edge_index=edge_index,
        model=model,
        target=out,
        target_index=None,
        explainer_config=explainer_config,
        model_config=model_config,
        node_index=2,
    )

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [GCN_multioutput_regression, GAT])
@pytest.mark.parametrize('return_type', return_types_regression)
def test_gnn_explainer_node_regression_multioutput(
    edge_mask_type,
    node_mask_type,
    model,
    return_type,
):
    explainer_config = ExplainerConfig(
        explanation_type="model",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='node',
        return_type=return_type,
        mode="regression",
    )

    explainer = GNNExplainer()
    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index)

    # try to explain prediction for node 0
    explanation = explainer(
        x=x,
        edge_index=edge_index,
        model=model,
        target=out,
        target_index=1,
        explainer_config=explainer_config,
        model_config=model_config,
        node_index=2,
    )

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [GCN_single_output_classification, GAT])
@pytest.mark.parametrize('return_type', return_types_classification)
def test_gnn_explainer_node_classification_single_output(
    edge_mask_type,
    node_mask_type,
    model,
    return_type,
):
    explainer_config = ExplainerConfig(
        explanation_type="model",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='node',
        return_type=return_type,
        mode="classification",
    )

    explainer = GNNExplainer()
    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index).argmax(dim=1)

    # try to explain prediction for node 0
    explanation = explainer(
        x=x,
        edge_index=edge_index,
        model=model,
        target=out,
        target_index=None,
        explainer_config=explainer_config,
        model_config=model_config,
        node_index=2,
    )

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [GCN_multioutput_classification])
@pytest.mark.parametrize('return_type', return_types_classification)
def test_gnn_explainer_node_classification_multioutput(
    edge_mask_type,
    node_mask_type,
    model,
    return_type,
):
    explainer_config = ExplainerConfig(
        explanation_type="model",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='node',
        return_type=return_type,
        mode="classification",
    )

    explainer = GNNExplainer()
    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index).argmax(dim=-1)

    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, model=model,
                            target=out, target_index=0,
                            explainer_config=explainer_config,
                            model_config=model_config, node_index=2)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [
    GNN_classification_multioutput,
    GNN_classification_singleoutput,
])
@pytest.mark.parametrize('return_type', return_types_classification)
def test_gnn_explainer_graph_classification(edge_mask_type, node_mask_type,
                                            model, return_type):
    explainer_config = ExplainerConfig(
        explanation_type="model",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='graph',
        return_type=return_type,
        mode="classification",
    )

    explainer = GNNExplainer()
    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index, None, None).argmax(dim=-1)

    if isinstance(model, GNN_classification_multioutput):
        target_index = 0
    else:
        target_index = None

    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, model=model,
                            target=out, target_index=target_index,
                            explainer_config=explainer_config,
                            model_config=model_config, edge_attr=None,
                            batch=None)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [
    GNN_regression_singleoutput,
    GNN_regression_multioutput,
])
@pytest.mark.parametrize('return_type', return_types_regression)
def test_gnn_explainer_graph_regression(edge_mask_type, node_mask_type, model,
                                        return_type):
    explainer_config = ExplainerConfig(
        explanation_type="model",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='graph',
        return_type=return_type,
        mode="regression",
    )

    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index, None, None)

    if isinstance(model, GNN_regression_multioutput):
        target_index = 0
    else:
        target_index = None

    explainer = GNNExplainer()
    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, model=model,
                            target=out, target_index=target_index,
                            explainer_config=explainer_config,
                            model_config=model_config, edge_attr=None,
                            batch=None)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [
    GNN_regression_singleoutput,
    GNN_regression_multioutput,
])
@pytest.mark.parametrize('return_type', return_types_regression)
@pytest.mark.parametrize("explanation_type", ["model", "phenomenon"])
def test_gnn_explainer_with_meta_explainer_regression_graph(
        edge_mask_type, node_mask_type, model, return_type, explanation_type):
    explainer_config = ExplainerConfig(
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='graph',
        return_type=return_type,
        mode="regression",
    )

    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index, None, None)

    if isinstance(
            model,
            GNN_regression_multioutput,
    ):
        target_index = 0
    else:
        target_index = None

    explainer = Explainer(model=model, algorithm=GNNExplainer(),
                          explainer_config=explainer_config,
                          model_config=model_config)

    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, target=out,
                            target_index=target_index, edge_attr=None,
                            batch=None)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize(
    'model', [GNN_classification_singleoutput, GNN_classification_multioutput])
@pytest.mark.parametrize("explanation_type", ["model", "phenomenon"])
@pytest.mark.parametrize('return_type', return_types_regression)
def test_gnn_explainer_with_meta_explainer_classification_graph(
    edge_mask_type,
    node_mask_type,
    model,
    return_type,
    explanation_type,
):
    explainer_config = ExplainerConfig(
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='graph',
        return_type=return_type,
        mode="classification",
    )

    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index, None, None).argmax(dim=-1)

    if isinstance(model, GNN_classification_multioutput):
        target_index = 0
    else:
        target_index = None

    explainer = Explainer(model=model, algorithm=GNNExplainer(),
                          explainer_config=explainer_config,
                          model_config=model_config)

    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, target=out,
                            target_index=target_index, edge_attr=None,
                            batch=None)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize(
    'model',
    [GCN_multioutput_classification, GCN_single_output_classification])
@pytest.mark.parametrize("explanation_type", ["model", "phenomenon"])
@pytest.mark.parametrize('return_type', return_types_regression)
def test_gnn_explainer_with_meta_explainer_classification_node(
    edge_mask_type,
    node_mask_type,
    model,
    return_type,
    explanation_type,
):
    explainer_config = ExplainerConfig(
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='node',
        return_type=return_type,
        mode="classification",
    )

    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index).argmax(dim=-1)

    if isinstance(model, GCN_multioutput_classification):
        target_index = 0
    else:
        target_index = None

    explainer = Explainer(model=model, algorithm=GNNExplainer(),
                          explainer_config=explainer_config,
                          model_config=model_config)

    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, target=out,
                            target_index=target_index, node_index=2)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize(
    'model', [GCN_single_output_regression, GCN_multioutput_regression])
@pytest.mark.parametrize('return_type', return_types_regression)
@pytest.mark.parametrize("explanation_type", ["model", "phenomenon"])
def test_gnn_explainer_with_meta_explainer_regression_node(
    edge_mask_type,
    node_mask_type,
    model,
    return_type,
    explanation_type,
):
    explainer_config = ExplainerConfig(
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='node',
        return_type=return_type,
        mode="regression",
    )

    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index)

    if isinstance(model, GCN_multioutput_regression):
        target_index = 0
    else:
        target_index = None

    explainer = Explainer(model=model, algorithm=GNNExplainer(),
                          explainer_config=explainer_config,
                          model_config=model_config)

    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, target=out,
                            target_index=target_index, node_index=2)

    check_explanation(edge_mask_type, node_mask_type, x, edge_index,
                      explanation)
