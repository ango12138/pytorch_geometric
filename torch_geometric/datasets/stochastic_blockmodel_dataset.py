import os
from typing import Optional, Union, List

import torch
from sklearn.datasets import make_classification
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import stochastic_blockmodel_graph


class StochasticBlockModelDataset(InMemoryDataset):

    def __init__(self, root,
                 block_sizes: Union[List[int], Tensor],
                 edge_probs: Union[List[List[float]], Tensor],
                 num_channels: Optional[int] = None,
                 is_undirected: bool = True,
                 x_kws: Optional[dict] = None,
                 transform=None, pre_transform=None):
        r"""A synthetic graph dataset generated by the stochastic block
        model. The node features of each block are sampled from normal
        distributions where the centers of clusters are vertices of a
        hypercube by :meth:`sklearn.datasets.make_classification` method.

        Args:
            root (string): Root directory where the dataset should be saved.
            block_sizes ([int] or LongTensor): The sizes of blocks.
            edge_probs ([[float]] or FloatTensor): The density of edges going
                from each block to each other block. Must be symmetric if the
                graph is undirected.
            num_channels (int, optional): The number of node features. If given
                as :obj:`None`, node features are not generated.
            is_undirected (bool): Whether the graph to generate is undirected.
            **kwargs (optional): The keyword arguments that are passed down
                to :meth:`sklearn.datasets.make_classification` method in
                drawing node features.
            transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                every access. (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes
                in an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed
                before being saved to disk. (default: :obj:`None`)
        """
        if not isinstance(block_sizes, torch.Tensor):
            block_sizes = torch.tensor(block_sizes, dtype=torch.long)
        if not torch.is_tensor(edge_probs):
            edge_probs = torch.tensor(edge_probs, dtype=torch.float)

        self.block_sizes = block_sizes
        self.edge_probs = edge_probs
        self.num_channels = num_channels
        self.is_undirected = is_undirected

        self.x_kws = {'n_informative': num_channels,
                      'n_redundant': 0,
                      'flip_y': 0.0,
                      'shuffle': False}
        self.x_kws.update(x_kws or {})

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def name(self):
        return f'sbm-{self.num_channels}-{self.block_sizes.tolist()}' \
               f'-{self.edge_probs.tolist()}'

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.__class__.__name__,
                            self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.__class__.__name__,
                            self.name, 'processed')

    @property
    def processed_file_names(self):
        return f'{self.name}.pt'
        return 'data.pt'

    def process(self):
        edge_index = stochastic_blockmodel_graph(
            self.block_sizes, self.edge_probs, directed=not self.is_undirected)
        if self.num_channels is not None:
            x, _ = make_classification(
                n_samples=sum(self.block_sizes),
                n_features=self.num_channels,
                n_classes=len(self.block_sizes),
                weights=torch.tensor(self.block_sizes) / sum(self.block_sizes),
                **self.x_kws,
            )
            x = torch.tensor(x, dtype=torch.float)
        else:
            x = None
        y = torch.cat([torch.full((b,), i)
                       for i, b in enumerate(self.block_sizes)])

        data = Data(x=x, edge_index=edge_index, y=y)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])


class RandomPartitionGraphDataset(StochasticBlockModelDataset):

    def __init__(self, root,
                 num_classes: int,
                 num_nodes_per_class: int,
                 node_homophily_ratio: float,
                 average_degree: float,
                 num_channels: Optional[int] = None,
                 is_undirected: bool = True,
                 x_kws: Optional[dict] = None,
                 transform=None, pre_transform=None):
        r"""The random partition graph dataset from the `"How to Find Your
        Friendly Neighborhood: Graph Attention Design with Self-Supervision"
        <https://openreview.net/forum?id=Wi5KUNlqWty>`_ paper. This is a
        synthetic graph of communities controlled by the node homophily
        and the average degree, and each community is considered as a class.
        The node features are sampled from normal distributions where the
        centers of clusters are vertices of a hypercube by
        :obj:`'sklearn.datasets.make_classification'` method.

        Args:
            root (string): Root directory where the dataset should be saved.
            num_classes (int): The number of classes.
            num_nodes_per_class (int): The number of nodes per class.
            node_homophily_ratio (float): The degree of node homophily.
            average_degree (float): The average degree of the graph.
            num_channels (int, optional): The number of node features. If given
                as :obj:`None`, node features are not generated.
            is_undirected (bool): Whether the graph to generate is undirected.
            x_kws (dict, optional): The keyword arguments that are passed down
                to :obj:`'sklearn.datasets.make_classification'` method in
                drawing node features.
            transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                every access. (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes
                in an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
        """
        self._num_classes = num_classes
        self.num_nodes_per_class = num_nodes_per_class
        self.node_homophily_ratio = node_homophily_ratio
        self.average_degree = average_degree

        # (p_in + (C - 1) * p_out) / C = |E|/|V|^2
        # i.e., p_in + (C - 1) * p_out = average_degree / num_nodes_per_class
        ec_over_v2 = average_degree / num_nodes_per_class
        p_in = node_homophily_ratio * ec_over_v2
        p_out = (ec_over_v2 - p_in) / (num_classes - 1)

        block_sizes = [num_nodes_per_class for _ in range(num_classes)]
        edge_probs = [[p_out for _ in range(num_classes)]
                      for _ in range(num_classes)]
        for r in range(num_classes):
            edge_probs[r][r] = p_in

        super().__init__(root, block_sizes, edge_probs, num_channels,
                         is_undirected, x_kws, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def name(self):
        return f'rpg-{self.num_channels}-{self._num_classes}' \
               f'-{self.num_nodes_per_class}-{self.node_homophily_ratio}' \
               f'-{self.average_degree}'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        return super().process()
