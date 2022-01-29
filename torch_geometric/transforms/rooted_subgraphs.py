import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RootedSubgraphs(BaseTransform):
    r""" Record rooted subgraphs for each node in the graph. Two types of rooted subgraphs are supported.
    One is k-hop Egonet. The other is random-walk (node2vec) subgraph. The object transform a Data object
    to RootedSubgraphsData object. 

    Args:
        --------------- k-hop Egonet args ---------------
        hops (int): k for k-hop Egonet. 
        ----------- random-walk subgraph args -----------
        walk_length (int, optional): the length of random walk. When it is 0 use k-hop Egonet. 
        p (float, optional): parameters of node2vec's random walk. 
        q (float, optional): parameters of node2vec's random walk.
        repeat (int, optional): times of repeating the random walk to reduce randomness. 
    """
    def __init__(self, hops: int, walk_length: int=0, p: float=1., q: float=1., repeat: int=1):
        super().__init__()
        self.num_hops = hops
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.repeat = repeat

    def __call__(self, data: Data) -> Data:
        subgraphs_nodes_mask, subgraphs_edges_mask, hop_indicator_dense = extract_subgraphs(
            data.edge_index, data.num_nodes, self.num_hops, self.walk_length,
            self.p, self.q, self.repeat)
        subgraphs_nodes, subgraphs_edges, hop_indicator = to_sparse(
            subgraphs_nodes_mask, subgraphs_edges_mask, hop_indicator_dense)
        combined_subgraphs = combine_subgraphs(data.edge_index,
                                               subgraphs_nodes,
                                               subgraphs_edges,
                                               num_selected=data.num_nodes,
                                               num_nodes=data.num_nodes)

        data = RootedSubgraphsData(**{k: v for k, v in data})
        data.subgraphs_batch = subgraphs_nodes[0]
        data.subgraphs_nodes_mapper = subgraphs_nodes[1]
        data.subgraphs_edges_mapper = subgraphs_edges[1]
        data.combined_subgraphs = combined_subgraphs
        data.hop_indicator = hop_indicator
        data.__num_nodes__ = data.num_nodes
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.hops}  )'


import re


class RootedSubgraphsData(Data):
    r""" A data object describing a homogeneous graph with recording each node's rooted subgraph. 
    It contains several additional propreties that hold the information of all nodes' rooted subgraphs.
    Assume the data represents a graph with :math:'N' nodes and :math:'M' edges, also each node 
    :math:'i\in \[N\]' has a rooted subgraph with :math:'N_i' nodes and :math:'M_i' edges.
    
    Additional Properties:
        subgraphs_nodes_mapper (LongTensor): map each node in rooted subgraphs to a node in the original graph.
            Size: :math:'\sum_{i=1}^{N}N_i x 1'
        subgraphs_edges_mapper (LongTensor): map each edge in rotted subgraphs to a edge in the original graph.
            Size: :math:'\sum_{i=1}^{N}M_i x 1'
        subgraphs_batch: map each node in rooted subgraphs to its corresponding rooted subgraph index. 
            Size: :math:'\sum_{i=1}^{N}N_i x 1'
        combined_rooted_subgraphs: edge_index of a giant graph which represents a stacking of all rooted subgraphs. 
            Size: :math:'2 x \sum_{i=1}^{N}M_i'
        
    The class works as a wrapper for the data with these properties, and automatically handles mini batching for
    them. 
    
    """
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        if bool(re.search('(combined_subgraphs)', key)):
            return getattr(
                self, key[:-len('combined_subgraphs')] +
                'subgraphs_nodes_mapper').size(0)
        elif bool(re.search('(subgraphs_batch)', key)):
            # should use number of subgraphs or number of supernodes.
            return 1 + getattr(self, key)[-1]
        elif bool(re.search('(nodes_mapper)|(selected_supernodes)', key)):
            return num_nodes
        elif bool(re.search('(edges_mapper)', key)):
            # batched_edge_attr[subgraphs_edges_mapper] shoud be batched_combined_subgraphs_edge_attr
            return num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_subgraphs)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


def extract_subgraphs(edge_index, num_nodes, num_hops, walk_length=0, p=1, q=1,
                      repeat=1):
    if walk_length > 0:
        node_mask, hop_indicator = random_walk_subgraph(
            edge_index, num_nodes, walk_length, p=p, q=q, repeat=repeat,
            cal_hops=True)
    else:
        node_mask, hop_indicator = k_hop_subgraph(edge_index, num_nodes,
                                                  num_hops)
    edge_mask = node_mask[:, edge_index[0]] & node_mask[:, edge_index[
        1]]  # N x E dense mask matrix
    return node_mask, edge_mask, hop_indicator


def to_sparse(node_mask, edge_mask, hop_indicator):
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    if hop_indicator is not None:
        hop_indicator = hop_indicator[subgraphs_nodes[0], subgraphs_nodes[1]]
    return subgraphs_nodes, subgraphs_edges,


def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges,
                      num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1

    combined_subgraphs = edge_index[:, subgraphs_edges[1]]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]] = torch.arange(
        len(subgraphs_nodes[1]))
    node_label_mapper = node_label_mapper.reshape(-1)

    inc = torch.arange(num_selected) * num_nodes
    combined_subgraphs += inc[subgraphs_edges[0]]
    combined_subgraphs = node_label_mapper[combined_subgraphs]
    return combined_subgraphs


##########################
# Helpers
##########################
from torch_sparse import SparseTensor


def k_hop_subgraph(edge_index, num_nodes, num_hops):
    # return k-hop subgraphs for all nodes in the graph
    row, col = edge_index
    sparse_adj = SparseTensor(row=row, col=col,
                              sparse_sizes=(num_nodes, num_nodes))
    hop_masks = [
        torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)
    ]  # each one contains <= i hop masks
    hop_indicator = row.new_full((num_nodes, num_nodes), -1)
    hop_indicator[hop_masks[0]] = 0
    for i in range(num_hops):
        next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
        hop_masks.append(next_mask)
        hop_indicator[(hop_indicator == -1) & next_mask] = i + 1
    hop_indicator = hop_indicator.T  # N x N
    node_mask = (hop_indicator >= 0)  # N x N dense mask matrix
    return node_mask, hop_indicator


from torch_cluster import random_walk


def random_walk_subgraph(edge_index, num_nodes, walk_length, p=1, q=1,
                         repeat=1, cal_hops=True, max_hops=10):
    """
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)  Setting it to a high value (> max(q, 1)) ensures 
            that we are less likely to sample an already visited node in the following two steps.
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
            if q > 1, the random walk is biased towards nodes close to node t.
            if q < 1, the walk is more inclined to visit nodes which are further away from the node t.
        p, q ∈ {0.25, 0.50, 1, 2, 4}.
        Typical values:
        Fix p and tune q 
        repeat: restart the random walk many times and combine together for the result
    """
    row, col = edge_index
    start = torch.arange(num_nodes, device=edge_index.device)
    walks = [
        random_walk(row, col, start=start, walk_length=walk_length, p=p, q=q,
                    num_nodes=num_nodes) for _ in range(repeat)
    ]
    walk = torch.cat(walks, dim=-1)
    node_mask = row.new_empty((num_nodes, num_nodes), dtype=torch.bool)
    # print(walk.shape)
    node_mask.fill_(False)
    node_mask[start.repeat_interleave((walk_length + 1) * repeat),
              walk.reshape(-1)] = True
    if cal_hops:  # this is fast enough
        sparse_adj = SparseTensor(row=row, col=col,
                                  sparse_sizes=(num_nodes, num_nodes))
        hop_masks = [
            torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)
        ]
        hop_indicator = row.new_full((num_nodes, num_nodes), -1)
        hop_indicator[hop_masks[0]] = 0
        for i in range(max_hops):
            next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
            hop_masks.append(next_mask)
            hop_indicator[(hop_indicator == -1) & next_mask] = i + 1
            if hop_indicator[node_mask].min() != -1:
                break
        return node_mask, hop_indicator
    return node_mask, None
