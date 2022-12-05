import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import PGMExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    NNConv,
    global_mean_pool,
    graclus,
    max_pool,
    max_pool_x,
)
from torch_geometric.utils import normalized_cut

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
transform = T.Cartesian(cat=False)
train_dataset = MNISTSuperpixels(path, True, transform=transform)
test_dataset = MNISTSuperpixels(path, False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
d = train_dataset


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(2, 25), nn.ReLU(),
                            nn.Linear(25, d.num_features * 32))
        self.conv1 = NNConv(d.num_features, 32, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(2, 25), nn.ReLU(),
                            nn.Linear(25, 32 * 64))
        self.conv2 = NNConv(32, 64, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, d.num_classes)

    def forward(self, x, data, **kwargs):
        data = data.detach().clone()
        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, x.size(0))
        data.edge_attr = None
        data.x = x
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


def train(model, dataloader, epoch):
    model.train()

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 26:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        F.nll_loss(model(data.x, data), data.y).backward()
        optimizer.step()


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # for epoch in range(2):
    #     train(model, test_loader, epoch)

    explainer = Explainer(
        model=model, algorithm=PGMExplainer(perturb_feature_list=[0]),
        explainer_config=dict(explanation_type="phenomenon",
                              node_mask_type="attributes",
                              edge_mask_type="object"),
        model_config=dict(mode="classification", task_level="graph",
                          return_type="raw"))
    i = 0
    # todo bit of a hack--nicer to find a way to get the correct
    #  graph with the collate?
    # somthing like explain_dataset[10]is wrong shape for model??
    for explain_dataset in test_loader:
        explain_dataset.to(device)
        node_idx = 10
        explanation = explainer(x=explain_dataset.x,
                                edge_index=explain_dataset.edge_index,
                                index=node_idx, target=explain_dataset.y,
                                edge_attr=explain_dataset.edge_attr,
                                data=explain_dataset)
        print(explanation.available_explanations)
        i += 1
        if i > 2:
            break
