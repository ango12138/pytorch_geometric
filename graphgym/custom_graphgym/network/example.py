import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch_geometric.graphgym.config import cfg
import torch_geometric.graphgym.models.head
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_network


class ExampleGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, num_layers=2, model_type='GCN'):
        super(ExampleGNN, self).__init__()
        conv_model = self.build_conv_model(model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(dim_in, dim_in))

        for _ in range(num_layers - 1):
            self.convs.append(conv_model(dim_in, dim_in))

        GNNHead = register.head_dict[cfg.dataset.task]
        self.post_mp = GNNHead(dim_in=dim_in, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GAT':
            return pyg_nn.GATConv
        elif model_type == "GraphSage":
            return pyg_nn.SAGEConv
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        x, edge_index, x_batch = \
            batch.x, batch.edge_index, batch.batch

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)

        batch.x = x
        batch = self.post_mp(batch)

        return batch


register_network('example', ExampleGNN)
