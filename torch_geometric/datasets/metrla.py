from pathlib import Path
from typing import Callable, Optional

import torch
from google_drive_downloader import GoogleDriveDownloader as gdd

from torch_geometric.data import Data, Dataset
from torch_geometric.io.metrla import MetrLaIo
from torch_geometric.utils.sparse import dense_to_sparse


class MetrLa(Dataset):
    r"""The Los Angeles Metropolitan (MetrLA) highway traffic dataset
    introducted in `"Big Data and its Technical Challenges"
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.681.7248&rep=rep1&type=pdf>`_
    paper.

    Nodes represent sensors in a sensor network, each being able to
    detect cars passing by it. The data points represent the traffic
    volumes (how many cars have passed by a sensor), aggregated in 5 minutes
    intervals. The traffic prediciton task is a time series regression,
    which aims to predict the traffic volumen values in each node for the
    next *n_next* time steps, given the volumnes in the *n_prev*  previous
    time steps. Both n_next and n_prev are pre-set values. Given the
    time-series nature of the data, each node has an additional, temporal
    dimension.

    The initial adjacency matrix is constructed by applying a thresholded
    Gaussian kernel over the real-world distances between the sensors:

    :math:`W_{ij} = exp(-\frac{dist(v_{i}, v_{j})^2}{\sigma^{2}})`,
    if :math:`dist(v_{i}, v_{j}) >= K`, otherwise :math:`0`, where K is a
    pre-fixed threshold, and dist(, ) is the real-world road distance
    between two sensors, and :math:`\sigma^{2}` is the variance of the
    distances between sensors. A zero-value in the adjacency matrix does
    not indicate that thers is no road connection between two sensors (
    because there almost always is, since road networks are connected
    graphs. Instead, it means that the distance between them is larger than
    the threshold value.

    Args:
        root (string): Root directory where the dataset should besaved.
        n_previous_steps (int): The number of previous time steps to consider
            when building the predictor variable.
        n_future_steps (int): The number of next time steps to consdier when
            building the target variable.
        add_time_of_day (bool): Whether to inject day of week information in
            the features.
        add_day_of_week (bool): Whether to inject time of day information in
            the features.
        normalized_k (float):  The threshold for constructing the adjacency
            matrix based on the thresholded Gaussian kernel.
        transform: A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform: A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before being saved to
            disk
    """
    gdrive_ids = {
        'distances.csv': '19Td6JafGnF8CD2H64jWCBV7k5GyFix2H',
        'locations.csv': '1FnioVF2jZuOl_St1ssLnvQgSHuEuDzvL',
        'sensors.txt': '1235bXBxe4X73dJk3zwRzaIQ-8A7K_OEV',
        'sensor-readings.csv': '1QZpu2GAeH6veewwF1WZXX-ErDoGxk7uF'
    }

    def __init__(self, root: Optional[str], n_previous_steps: int,
                 n_future_steps: int, add_time_of_day: bool = False,
                 add_day_of_week: bool = False, normalized_k: float = .1,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None) -> None:
        """This is a circular dependency. You need this to calculate processed
        file names, but you need it before the call call to the "process"
        method in the superclass. The only in-between place you could
        effectively calculate this is before the process, so either in
        __init__ (where you don't have access to the downloaded file
        containing the readings, or in download, but download gets skipped
        after the first time, so that's not reliable. Open to any
        workarounds, but the dataset has a fixed size, so
        this is not a deal breaker imo."""
        self.n_readings = 34272

        self.io = MetrLaIo(n_readings=self.n_readings,
                           n_previous_steps=n_previous_steps,
                           n_future_steps=n_future_steps,
                           add_time_of_day=add_time_of_day,
                           add_day_of_week=add_day_of_week,
                           normalized_k=normalized_k)
        super().__init__(root, transform, pre_transform)

        self.normalized_k = normalized_k

    @property
    def raw_file_names(self) -> str:
        r"""The name of the files in the :obj:`self.raw_dir`
        folder that must be present in order to skip downloading."""
        return list(self.gdrive_ids.keys())

    @property
    def processed_file_names(self) -> str:
        r"""The name of the files in the :obj:`self.processed_dir`
        folder that must be present in order to skip processing."""
        return [f'data_{i}.pt' for i in range(self.io.dataset_len)
                ] + ["edge_index.pt", "edge_attributes.pt"]

    @property
    def edge_index_file_name(self) -> str:
        r"""The name of the file containing the edge index."""
        return self.processed_file_names[-2]

    @property
    def edge_attributes_file_name(self) -> str:
        r"""The name of the file containing the edge attributes."""
        return self.processed_file_names[-1]

    def download(self) -> None:
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        for file_name, gdrive_id in self.gdrive_ids.items():
            gdd.download_file_from_google_drive(
                file_id=gdrive_id, dest_path=Path(self.raw_dir) / file_name)

    def process(self) -> None:
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""

        x, y = self.io.get_metrla_data(data_path=self.raw_paths[3])

        adjacency_matrix = self.io.generate_adjacency_matrix(
            distances_path=self.raw_paths[0],
            sensor_ids_path=self.raw_paths[2])

        adjacency_matrix = torch.from_numpy(adjacency_matrix)

        edge_index, edge_attributes = dense_to_sparse(adjacency_matrix)

        edge_index_path = Path(self.processed_dir) / self.edge_index_file_name
        edge_attributes_path = Path(
            self.processed_dir) / self.edge_attributes_file_name

        torch.save(obj=edge_index, f=edge_index_path)
        torch.save(obj=edge_attributes, f=edge_attributes_path)

        for idx in range(self.io.dataset_len):
            # Select the "slice" among the first dimension
            # x_index (n_previous_steps, n_nodes, n_features)
            x_index = torch.tensor(data=x[idx, ...], dtype=torch.float32)

            # y_index (n_next_steps, n_nodes, n_features)
            y_index = torch.tensor(data=x[idx, ...], dtype=torch.float32)

            data = Data(x=x_index, y=y_index)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            file_name = self.processed_file_names[idx]
            torch.save(obj=data, f=Path(self.processed_dir) / file_name)

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        return len(self.processed_file_names) - 2

    def get(self, idx: int) -> Data:
        r"""Gets the data object at index :obj:`idx`."""
        file_name = self.processed_file_names[idx]
        edge_index = torch.load(f=Path(self.processed_dir) /
                                self.edge_index_file_name)
        edge_attributes = torch.load(f=Path(self.processed_dir) /
                                     self.edge_attributes_file_name)

        data = torch.load(f=Path(self.processed_dir) / file_name)
        setattr(data, 'edge_index', edge_index)
        setattr(data, 'edge_attr', edge_attributes)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name.capitalize()}()'
