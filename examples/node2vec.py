import os.path as osp
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec


def main():
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset)
    data = dataset[0]
    # data = torch.load('./graph_disease.pt')                                    # an example of data with attrs

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(
        data.edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        edge_attr=data.edge_attr,  # Add data attribute to the input
        fast_mode=True,  # Fast Estimate or Exact Computation
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True,
    ).to(device)

    num_workers = 0 if sys.platform.startswith('win') else 4
    loader = model.loader(batch_size=128, shuffle=True,
                          num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
        return acc

    for epoch in range(1, 101):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    @torch.no_grad()
    def plot_points():  # Liitle changes in plot for having labels in graph
        model.eval()
        z = model(torch.arange(data.num_nodes, device=device))
        z_cpu = TSNE(n_components=2,
                     perplexity=min([len(data.x) / 2,
                                     30.0])).fit_transform(z.cpu().numpy())
        y_cpu = data.y.cpu().numpy()

        plt.figure(figsize=(8, 8))
        for i in range(len(data.y.unique())):
            plt.scatter(z_cpu[y_cpu == i, 0], z_cpu[y_cpu == i, 1], s=20)

        try:
            for i, txt in enumerate(data.labels):
                plt.annotate(txt, (z_cpu[i, 0], z_cpu[i, 1]))
        except:
            print('no labels')
        plt.axis('off')
        plt.show()

    plot_points()


if __name__ == "__main__":
    main()
