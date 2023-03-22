from utils import NormalizedDegree
from torch_geometric.datasets import Planetoid,TUDataset
import torch
from torch_geometric.utils import degree
from quantizer import ProbabilisticHighDegreeMask

# Function to sample the datset based on the node degree. Uses One hot representation of degree as features for dataset with no features. 
def get_dataset(path, name, sparse=True, cleaned=False, DQ=None):
    
    if name in ['Cora', 'Citeseer']:
      dataset = Planetoid(path, name)
    else:
      dataset = TUDataset(path, name, cleaned=cleaned )
      dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == "REDDIT-BINARY":
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset[torch.tensor(indices)]

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose([dataset.transform, T.ToDense(num_nodes)])

    if DQ is not None:
        print(f"Generating ProbabilisticHighDegreeMask: {DQ}")
        dq_transform = ProbabilisticHighDegreeMask(
            DQ["prob_mask_low"], min(DQ["prob_mask_low"] + DQ["prob_mask_change"], 1.0)
        )
        # NOTE: see issue #1 if you are customizing for your own dataset
        # dataset.transform may be None (not the case here)
        if dataset.transform is None:
            dataset.transform = dq_transform
        else:
            dataset.transform = T.Compose([dataset.transform, dq_transform])

    return dataset