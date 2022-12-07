from typing import Optional, Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import segregate_self_loops


@functional_transform('remove_duplicate_self_loops')
class RemoveDuplicateSelfLoops(BaseTransform):
    r"""Removes duplicate self-loops to the given homogeneous or heterogeneous graph. It will change the original order of dataset by concatenating unique self-looping edges at the end of the dataset. It can be used to clean up known issue with ogbn-products dataset:
    <https://ogb.stanford.edu/docs/nodeprop/#:~:text=Note%3A%20A%20very%20small%20number%20of%20self%2Dconnecting%20edges%20are%20repeated%20(see%20here)%3B%20you%20may%20remove%20them%20if%20necessary>
    (functional name: :obj:`remove_duplicate_self_loops`).
    """

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite() or 'edge_index' not in store:
                continue

            (edge_index, edge_weight, loop_edge_index, loop_edge_attr) = segregate_self_loops(store.edge_index, getattr(store, 'edge_weight', None))
            
            loop_edge_index = torch.unique(loop_edge_index[0])
            loop_edge_index = torch.stack((loop_edge_index, loop_edge_index), dim=0)

            edge_index = torch.cat((edge_index, loop_edge_index), dim=1)
            setattr(store, 'edge_index', edge_index)
            
            if edge_weight is not None:
                loop_edge_attr = torch.unique(loop_edge_attr[0])
                loop_edge_attr = torch.stack((loop_edge_attr, loop_edge_attr), dim=0) 
                torch.cat((edge_weight, loop_edge_attr), dim=1)
                setattr(store, 'edge_weight', edge_weight)
        
        return data
