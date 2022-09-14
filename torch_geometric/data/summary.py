import pandas as pd
from tabulate import tabulate
from tqdm import tqdm


class Summary:
    r"""Summary of graph datasets

    Args:
        dataset (Dataset): :obj:`torch_geometric.data.Dataset`
    """
    def __init__(self, dataset):
        self.dataset_str = repr(dataset)

        def map_data(data):
            return data.num_nodes, data.num_edges

        iter = map(map_data, tqdm(dataset))
        df = pd.DataFrame(list(iter), columns=["nodes", "edges"])
        self._desc = df.describe()

    @property
    def num_graphs(self) -> int:
        r"""The number of graphs in the dataset"""
        # note can use either nodes or edges, the counts are the same
        return int(self._desc['nodes']['count'])

    @property
    def min_num_nodes(self) -> int:
        r"""The minimum number of nodes"""
        return int(self._desc['nodes']['min'])

    @property
    def max_num_nodes(self) -> int:
        r"""The maximum number of nodes"""
        return int(self._desc['nodes']['max'])

    @property
    def median_num_nodes(self) -> int:
        r"""The median number of nodes"""
        return int(self._desc['nodes']['median'])

    @property
    def mean_num_nodes(self) -> float:
        r"""The mean number of nodes"""
        return self._desc['nodes']['mean']

    @property
    def std_num_nodes(self) -> float:
        r"""The standard deviation of the number of nodes"""
        return self._desc['nodes']['std']

    @property
    def min_num_edges(self) -> int:
        r"""The minimum number of edges"""
        return int(self._desc['edges']['min'])

    @property
    def max_num_edges(self) -> int:
        r"""The maximum number of edges"""
        return int(self._desc['edges']['max'])

    @property
    def median_num_edges(self) -> int:
        r"""The median number of edges"""
        return int(self._desc['edges']['median'])

    @property
    def mean_num_edges(self) -> float:
        r"""The mean number of edges"""
        return self._desc['edges']['mean']

    @property
    def std_num_edges(self) -> float:
        r"""The standard deviation of the number of edges"""
        return self._desc['edges']['std']

    def __repr__(self) -> str:
        prefix = self.__class__.__name__ + " " + self.dataset_str + "\n"
        t = self._desc.drop('count')
        body = tabulate(t, headers=t.columns)
        return prefix + body
