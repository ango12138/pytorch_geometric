import logging
from heapq import heapify, heappop, heappush
from math import ceil
from typing import List, Optional, Tuple, Union

import numba
import numpy as np
import torch
from python_tsp.heuristics import solve_tsp_simulated_annealing
from scipy.sparse import coo_matrix, csr_matrix
from torch.utils.data import DataLoader, RandomSampler
from torch_sparse import SparseTensor
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader.ibmb_sampler import (
    IBMBOrderedSampler,
    IBMBWeightedSampler,
)
from torch_geometric.utils import coalesce, is_undirected, subgraph


def get_partitions(
        edge_index: Union[torch.LongTensor, SparseTensor], num_partitions: int,
        indices: torch.LongTensor, num_nodes: int,
        output_weight: Optional[float] = None) -> List[torch.LongTensor]:
    assert isinstance(
        edge_index,
        (torch.LongTensor,
         SparseTensor)), f'Unsupported edge_index type {type(edge_index)}'
    if isinstance(edge_index, torch.LongTensor):
        edge_index = SparseTensor.from_edge_index(
            edge_index, sparse_sizes=(num_nodes, num_nodes))

    if output_weight is not None and output_weight != 1:
        node_weight = torch.ones(num_nodes)
        node_weight[indices] = output_weight
    else:
        node_weight = None

    _, partptr, perm = edge_index.partition(num_parts=num_partitions,
                                            recursive=False, weighted=False,
                                            node_weight=node_weight)

    partitions = []
    for i in range(len(partptr) - 1):
        partitions.append(perm[partptr[i]:partptr[i + 1]])

    return partitions


def get_pair_wise_distance(ys: List, num_classes: int,
                           dist_type: str = 'kl') -> np.ndarray:
    num_batches = len(ys)

    counts = np.zeros((num_batches, num_classes), dtype=np.int32)
    for i in range(num_batches):
        unique, count = np.unique(ys[i], return_counts=True)
        counts[i, unique] = count

    counts += 1
    counts = counts / counts.sum(1).reshape(-1, 1)
    pairwise_dist = np.zeros((num_batches, num_batches), dtype=np.float64)

    for i in range(0, num_batches - 1):
        for j in range(i + 1, num_batches):
            if dist_type == 'l1':
                pairwise_dist[i, j] = np.sum(np.abs(counts[i] - counts[j]))
            elif dist_type == 'kl':

                def kl_divergence(p: np.ndarray, q: np.ndarray):
                    return (p * np.log(p / q)).sum()

                pairwise_dist[i, j] = kl_divergence(counts[i],
                                                    counts[j]) + kl_divergence(
                                                        counts[j], counts[i])
            else:
                raise ValueError

    pairwise_dist += pairwise_dist.T
    pairwise_dist += 1e-5  # for numerical stability
    np.fill_diagonal(pairwise_dist, 0.)

    return pairwise_dist


def indices_complete_check(loader: List[Tuple[Union[torch.Tensor, np.ndarray],
                                              Union[torch.Tensor,
                                                    np.ndarray]]],
                           output_indices: Union[torch.Tensor, np.ndarray]):
    if isinstance(output_indices, torch.Tensor):
        output_indices = output_indices.cpu().numpy()

    outs = []
    for out, aux in loader:
        if isinstance(out, torch.Tensor):
            out = out.cpu().numpy()
        if isinstance(aux, torch.Tensor):
            aux = aux.cpu().numpy()

        assert np.all(np.in1d(out,
                              aux)), "Not all output nodes are in aux nodes!"
        outs.append(out)

    outs = np.sort(np.concatenate(outs))
    assert np.all(
        outs == np.sort(output_indices)), "Output nodes missing or duplicate!"


def get_subgraph(out_indices: torch.Tensor, graph: Data,
                 return_edge_index_type: str, adj: SparseTensor, **kwargs):
    if return_edge_index_type == 'adj':
        assert adj is not None

    if return_edge_index_type == 'adj':
        subg = Data(x=graph.x[out_indices], y=graph.y[out_indices],
                    edge_index=adj[out_indices, :][:, out_indices])
    elif return_edge_index_type == 'edge_index':
        edge_index, edge_attr = subgraph(out_indices, graph.edge_index,
                                         graph.edge_attr, relabel_nodes=True,
                                         num_nodes=graph.num_nodes,
                                         return_edge_mask=False)
        subg = Data(x=graph.x[out_indices], y=graph.y[out_indices],
                    edge_index=edge_index, edge_attr=edge_attr)
    else:
        raise NotImplementedError

    for k, v in kwargs.items():
        subg[k] = v

    return subg


def define_sampler(batch_order: str, ys: List[Union[torch.Tensor, np.ndarray,
                                                    List]], num_classes: int,
                   dist_type: str = 'kl'):
    if batch_order == 'rand':
        logging.info("Running with random order")
        sampler = RandomSampler(ys)
    elif batch_order in ['order', 'sample']:
        kl_div = get_pair_wise_distance(ys, num_classes, dist_type=dist_type)
        if batch_order == 'order':
            best_perm, _ = solve_tsp_simulated_annealing(kl_div)
            logging.info(f"Running with given order: {best_perm}")
            sampler = IBMBOrderedSampler(best_perm)
        else:
            logging.info("Running with weighted sampling")
            sampler = IBMBWeightedSampler(kl_div)
    else:
        raise ValueError

    return sampler


def create_batchwise_out_aux_pairs(
        adj: SparseTensor, partitions: List[Union[torch.LongTensor,
                                                  np.ndarray]],
        prime_indices: Union[torch.LongTensor, np.ndarray], topk: int,
        num_outnodeset_per_batch: int = 50, alpha: float = 0.2,
        ppr_iterations: int = 50) -> List[Tuple[np.ndarray, np.ndarray]]:
    def ppr_power_method(adj: SparseTensor,
                         batch: List[Union[np.ndarray,
                                           torch.LongTensor]], topk: int,
                         num_iter: int, alpha: float) -> List[np.ndarray]:
        topk_neighbors = []
        logits = torch.zeros(
            adj.size(0), len(batch),
            device=adj.device())  # each column contains a set of output nodes
        for i, tele_set in enumerate(batch):
            logits[tele_set, i] = 1. / len(tele_set)

        new_logits = logits.clone()
        for i in range(num_iter):
            new_logits = adj @ new_logits * (1 - alpha) + alpha * logits

        inds = new_logits.argsort(0)
        nonzeros = (new_logits > 0).sum(0)
        nonzeros = torch.minimum(
            nonzeros,
            torch.tensor([topk], dtype=torch.int64, device=adj.device()))
        for i in range(new_logits.shape[1]):
            topk_neighbors.append(inds[-nonzeros[i]:, i].cpu().numpy())

        return topk_neighbors

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    if isinstance(prime_indices, torch.Tensor):
        prime_indices = prime_indices.cpu().numpy()

    adj = adj.to(device)

    cur_output_nodes = []
    loader = []

    pbar = tqdm(range(len(partitions)))
    pbar.set_description("Processing topic-sensitive PPR batches")
    for n in pbar:
        part = partitions[n]
        if isinstance(part, torch.Tensor):
            part = part.cpu().numpy()

        primes_in_part, *_ = np.intersect1d(part, prime_indices,
                                            assume_unique=True,
                                            return_indices=True)
        if len(primes_in_part):  # There ARE output nodes in this partition
            cur_output_nodes.append(primes_in_part)

        # accumulate enough output nodes for a batch, to make good use of GPU memory
        if len(cur_output_nodes
               ) >= num_outnodeset_per_batch or n == len(partitions) - 1:
            topk_neighbors = ppr_power_method(adj, cur_output_nodes, topk,
                                              ppr_iterations, alpha)
            for i in range(len(cur_output_nodes)):
                # force output nodes to be aux nodes
                auxiliary_nodes = np.union1d(cur_output_nodes[i],
                                             topk_neighbors[i])
                loader.append((cur_output_nodes[i], auxiliary_nodes))
            cur_output_nodes = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return loader


def get_pairs(ppr_mat: csr_matrix) -> np.ndarray:
    ppr_mat = ppr_mat + ppr_mat.transpose()

    ppr_mat = ppr_mat.tocoo(
    )  # find issue: https://github.com/scipy/scipy/blob/v1.7.1/scipy/sparse/extract.py#L12-L40
    row, col, data = ppr_mat.row, ppr_mat.col, ppr_mat.data
    mask = (row > col)  # lu

    row, col, data = row[mask], col[mask], data[mask]
    sort_arg = np.argsort(data)[::-1]
    # sort_arg = parallel_sort.parallel_argsort(data)[::-1]

    # map prime_nodes to arange
    ppr_pairs = np.vstack((row[sort_arg], col[sort_arg])).T
    return ppr_pairs


@numba.njit(cache=True)
def prime_orient_merge(ppr_pairs: np.ndarray, primes_per_batch: int,
                       num_nodes: int):
    # cannot use list for id_primes_list, updating node_id_list[id_primes_list[id2]] require id_primes_list to be array
    id_primes_list = list(np.arange(num_nodes, dtype=np.int32).reshape(-1, 1))
    node_id_list = np.arange(num_nodes, dtype=np.int32)
    placeholder = np.zeros(0, dtype=np.int32)
    # size_flag = [{a} for a in np.arange(num_nodes, dtype=np.int32)]

    for i, j in ppr_pairs:
        id1, id2 = node_id_list[i], node_id_list[j]
        if id1 > id2:
            id1, id2 = id2, id1

        # if not (id1 in size_flag[id2] or id2 in size_flag[id1])
        if id1 != id2 and len(id_primes_list[id1]) + len(
                id_primes_list[id2]) <= primes_per_batch:
            id_primes_list[id1] = np.concatenate(
                (id_primes_list[id1], id_primes_list[id2]))
            node_id_list[id_primes_list[id2]] = id1
            # node_id_list[j] = id1
            id_primes_list[id2] = placeholder

    prime_lst = list()
    ids = np.unique(node_id_list)

    for _id in ids:
        prime_lst.append(list(id_primes_list[_id]))

    return list(prime_lst)


def prime_post_process(loader, merge_max_size):
    h = [(
        len(p),
        p,
    ) for p in loader]
    heapify(h)

    while len(h) > 1:
        len1, p1 = heappop(h)
        len2, p2 = heappop(h)
        if len1 + len2 <= merge_max_size:
            heappush(h, (len1 + len2, p1 + p2))
        else:
            heappush(h, (
                len1,
                p1,
            ))
            heappush(h, (
                len2,
                p2,
            ))
            break

    new_batch = []

    while len(h):
        _, p = heappop(h)
        new_batch.append(p)

    return new_batch


def topk_ppr_matrix(
        edge_index: torch.Tensor, num_nodes: int, alpha: float, eps: float,
        output_node_indices: Union[np.ndarray, torch.LongTensor], topk: int,
        normalization='row') -> Tuple[csr_matrix, List[np.ndarray]]:
    if isinstance(output_node_indices, torch.Tensor):
        output_node_indices = output_node_indices.numpy()

    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    edge_index_np = edge_index.cpu().numpy()

    _, indptr, out_degree = np.unique(edge_index_np[0], return_index=True,
                                      return_counts=True)
    indptr = np.append(indptr, len(edge_index_np[0]))

    neighbors, weights = calc_ppr_topk_parallel(indptr, edge_index_np[1],
                                                out_degree, alpha, eps,
                                                output_node_indices)

    def construct_sparse(neighbors, weights, shape):
        i = np.repeat(np.arange(len(neighbors)),
                      np.fromiter(map(len, neighbors), dtype=np.int))
        j = np.concatenate(neighbors)
        return coo_matrix((np.concatenate(weights), (i, j)), shape)

    ppr_matrix = construct_sparse(neighbors, weights,
                                  (len(output_node_indices), num_nodes))
    ppr_matrix = ppr_matrix.tocsr()

    def sparsify(neighbors: List[np.ndarray], weights: List[np.ndarray],
                 topk: int):
        new_neighbors = []
        for n, w in zip(neighbors, weights):
            idx_topk = np.argsort(w)[-topk:]
            new_neighbor = n[idx_topk]
            new_neighbors.append(new_neighbor)

        return new_neighbors

    neighbors = sparsify(neighbors, weights, topk)
    neighbors = [
        np.union1d(nei, pr) for nei, pr in zip(neighbors, output_node_indices)
    ]

    if normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg_sqrt = np.sqrt(np.maximum(out_degree, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt

        row, col = ppr_matrix.nonzero()
        ppr_matrix.data = deg_sqrt[
            output_node_indices[row]] * ppr_matrix.data * deg_inv_sqrt[col]
    elif normalization == 'col':
        # Assume undirected (symmetric) adjacency matrix
        deg_inv = 1. / np.maximum(out_degree, 1e-12)

        row, col = ppr_matrix.nonzero()
        ppr_matrix.data = out_degree[
            output_node_indices[row]] * ppr_matrix.data * deg_inv[col]
    elif normalization == 'row':
        pass
    else:
        raise ValueError(f"Unknown PPR normalization: {normalization}")

    return ppr_matrix, neighbors


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/gdc.html#GDC.diffusion_matrix_approx
@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes) \
        -> Tuple[List[np.ndarray], List[np.ndarray]]:
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        js[i] = np.array(j)
        vals[i] = np.array(val)
    return js, vals


@numba.njit(cache=True, locals={
    '_val': numba.float32,
    'res': numba.float32,
    'res_vnode': numba.float32
})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]

    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)

    return list(p.keys()), list(p.values())


class IBMBBaseLoader(DataLoader):
    def __init__(self, data_source: Union[List[Data],
                                          List[Tuple]], graph: Data,
                 adj: SparseTensor, return_edge_index_type: str, **kwargs):

        self.graph = graph
        self.adj = adj
        self.return_edge_index_type = return_edge_index_type
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        super().__init__(data_source, collate_fn=self.__collate__, **kwargs)

    def create_loader(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def prepare_cache(cls, graph: Data,
                      batch_wise_out_aux_pairs: List[Tuple[np.ndarray,
                                                           np.ndarray]],
                      adj: Optional[SparseTensor],
                      return_edge_index_type: str):
        subgraphs = []

        pbar = tqdm(batch_wise_out_aux_pairs)
        pbar.set_description(
            f"Caching data with type {return_edge_index_type}")

        if return_edge_index_type == 'adj':
            assert adj is not None, "Trying to cache adjacency matrix, got None type."

        for out, aux in pbar:
            mask = torch.from_numpy(np.in1d(aux, out))
            if isinstance(aux, np.ndarray):
                aux = torch.from_numpy(aux)
            subg = get_subgraph(aux, graph, return_edge_index_type, adj,
                                output_node_mask=mask)
            subgraphs.append(subg)

        return subgraphs

    @classmethod
    def create_adj_from_edge_index(cls, edge_index: torch.Tensor,
                                   num_nodes: int, normalization: str):
        assert normalization in [
            'sym', 'rw'
        ], f"Unsupported normalization type {normalization}"
        adj = SparseTensor.from_edge_index(edge_index,
                                           sparse_sizes=(num_nodes, num_nodes))
        adj = adj.fill_value(1.)
        degree = adj.sum(0)

        degree[degree == 0.] = 1e-12
        deg_inv = 1 / degree

        if normalization == 'sym':
            deg_inv_sqrt = deg_inv**0.5
            adj = adj * deg_inv_sqrt.reshape(1, -1)
            adj = adj * deg_inv_sqrt.reshape(-1, 1)
        elif normalization == 'rw':
            adj = adj * deg_inv.reshape(-1, 1)

        return adj

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __collate__(self, data_list: List[Union[Data, Tuple]]):
        if len(data_list) == 1 and isinstance(data_list[0], Data):
            return data_list[0]

        out, aux = zip(*data_list)
        out = np.concatenate(out)
        aux = np.unique(np.concatenate(aux))
        mask = torch.from_numpy(np.in1d(aux, out))
        aux = torch.from_numpy(aux)

        subg = get_subgraph(aux, self.graph, self.return_edge_index_type,
                            self.adj, output_node_mask=mask)
        return subg


class IBMBBatchLoader(IBMBBaseLoader):
    r"""
    A batch-wise IBMB dataloader of `"Influence-Based Mini-Batching for Graph Neural Networks"<https://openreview.net/forum?id=b9g0vxzYa_>`_ paper.

    First METIS graph partitioning algorithm is used, then we find the output nodes within each partition. Afterwards we find auxiliary nodes for each set of output nodes with topic-sensitvie pagerank as a batch.

    We end up with a list of :obj:`[(output: np.ndarray, auxiliary: np.ndarray), ...]` nodes pairs. If :obj:`batch_size` of the dataloader is 1, we precalculate and cache the subgraphs in memory. Otherwise, we might need to
    merge the subgraphs, e.g. :obj:`(Union(output1, output2, ...), Union(auxiliary1, auxiliary2, ...))`. Therefore, we do not cache them and instead collate the potentially overlapping subgraphs dynamically.

    Args:
        graph (torch_geometric.data.Data): The given graph data object, assumes to be undirected.
        batch_order (str): A string indicating the batch order type, should be one of :obj:`order`, :obj:`sample` and :obj:`rand`. If :obj:`order`, we calculate the pair wise KL divergence between every two batches, and organize an optimal order. If :obj:`sample`, we sample the next batch wrt the last one, a batch with higher KL divergence score is more likely to be sampled. If :obj:`rand`, we shuffle the batches randomly.
        num_partitions (int): Number of partitions.
        output_indices (torch.Tensor): A :obj:`torch.Tensor` vector containing a set of nodes as output nodes. Normally taken from train, validation or test split.
        return_edge_index_type (str): A string indicating the :obj:`edge_index` type. Should be either :obj:`adj` or :obj:`edge_index`. If :obj:`adj`, the edge_index of the batch will be a :obj:`SparseTensor`, otherwise :obj:`torch.Tensor`.
        batch_expand_ratio (float, optional): A :obj:`float` containing the ratio between the returned batch size and the original partition size. For example, the partition has 100 nodes, and you would like the batch to have 200 nodes, then set it to :obj:`2.`. (default: :obj:`1.`)
        metis_output_weight (float, optional): The weights on the output nodes for METIS graph partitioning algorithm. (default: :obj:`None`)
        num_outnodeset_per_batch (int, optional): An :obj:`int` indicating how many batches to run in parallel. We calculate the topic-sensitive pagerank scores with power iteration. If out-of-memory issue is encountered, set it to a smaller value. (default: :obj:`50`)
        alpha (float, optional): A :obj:`float` of the teleport probability of pagerank calculation.
        approximate_ppr_iterations (int, optional): An :obj:`int` of number of iterations of power iteration. (default: :obj:`50`)
        **kwargs (optional): Keywords of :obj:`torch.utils.data.DataLoader`. Specifically, :obj:`shuffle=False` is forced.
    """
    def __init__(self, graph: Data, batch_order: str, num_partitions: int,
                 output_indices: torch.Tensor, return_edge_index_type: str,
                 batch_expand_ratio: Optional[float] = 1.,
                 metis_output_weight: Optional[float] = None,
                 num_outnodeset_per_batch: Optional[int] = 50,
                 alpha: Optional[float] = 0.2,
                 approximate_ppr_iterations: Optional[int] = 50, **kwargs):

        self.subgraphs = []
        self.batch_wise_out_aux_pairs = []

        assert is_undirected(
            graph.edge_index,
            num_nodes=graph.num_nodes), "Assume the graph to be undirected"
        assert batch_order in ['rand', 'sample', 'order'
                               ], f"Unsupported batch order: {batch_order}"

        adj = self.create_adj_from_edge_index(graph.edge_index,
                                              graph.num_nodes,
                                              normalization='rw')

        self.cache_data = kwargs['batch_size'] == 1
        self.num_partitions = num_partitions
        self.output_indices = output_indices
        assert return_edge_index_type in ['adj', 'edge_index']
        self.return_edge_index_type = return_edge_index_type
        self.batch_expand_ratio = batch_expand_ratio
        self.metis_output_weight = metis_output_weight
        self.num_outnodeset_per_batch = num_outnodeset_per_batch
        self.alpha = alpha
        self.approximate_ppr_iterations = approximate_ppr_iterations

        self.create_loader(graph, adj)

        if len(self.batch_wise_out_aux_pairs) > 2:  # <= 2 order makes no sense
            ys = [
                graph.y[out].numpy()
                for out, _ in self.batch_wise_out_aux_pairs
            ]
            sampler = define_sampler(batch_order, ys, graph.y.max().item() + 1)
        else:
            sampler = kwargs['sampler']

        if not self.cache_data:
            cached_graph = graph  # need to cache the original graph
            if return_edge_index_type == 'adj':
                cached_adj = adj
            else:
                cached_adj = None
        else:
            cached_graph = None
            cached_adj = None

        super().__init__(
            self.subgraphs if self.cache_data else
            self.batch_wise_out_aux_pairs, cached_graph, cached_adj,
            return_edge_index_type, sampler=sampler, **kwargs)

    def create_loader(self, graph: Data, adj: SparseTensor):
        # graph partitioning
        partitions = get_partitions(adj, self.num_partitions,
                                    self.output_indices, graph.num_nodes,
                                    self.metis_output_weight)

        # get output - auxiliary node pairs
        topk = ceil(self.batch_expand_ratio * graph.num_nodes /
                    self.num_partitions)
        batch_wise_out_aux_pairs = create_batchwise_out_aux_pairs(
            adj, partitions, self.output_indices, topk,
            self.num_outnodeset_per_batch, self.alpha,
            self.approximate_ppr_iterations)

        indices_complete_check(batch_wise_out_aux_pairs, self.output_indices)
        self.batch_wise_out_aux_pairs = batch_wise_out_aux_pairs

        if self.cache_data:
            self.subgraphs = self.prepare_cache(graph,
                                                batch_wise_out_aux_pairs, adj,
                                                self.return_edge_index_type)

    def __getitem__(self, idx):
        return self.subgraphs[
            idx] if self.cache_data else self.batch_wise_out_aux_pairs[idx]

    def __len__(self):
        return self.num_partitions


class IBMBNodeLoader(IBMBBaseLoader):
    r"""A node-wise IBMB dataloader of `"Influence-Based Mini-Batching for Graph Neural Networks"<https://openreview.net/forum?id=b9g0vxzYa_>`_ paper.

    We first calculate the personalized pagerank (PPR) score for each output node, then take the :obj`k`: ones with the highest scores as its auxiliary nodes. We then merge them according to the pair wise PPR scores of each pair of output nodes.

    Similar to :class`~torch_geometric.loader.IBMBBatchLoader`:, we store the subgraphs in cache if :obj:`batch_size` is 1, otherwise we merge batches dynamically.

    Args:
        graph (torch_geometric.data.Data): The given graph data. Assumes to be undirected.
        batch_order (str): A string indicating the batch order type, should be one of :obj:`order`, :obj:`sample` and :obj:`rand`. If :obj:`order`, we calculate the pair wise KL divergence between every two batches, and organize an optimal order. If :obj:`sample`, we sample the next batch wrt the last one, a batch with higher KL divergence score is more likely to be sampled. If :obj:`rand`, we shuffle the batches randomly.
        output_indices (torch.Tensor): A :obj:`torch.Tensor` vector containing a set of nodes as output nodes. Normally taken from train, validation or test split.
        return_edge_index_type (str): A string indicating the :obj:`edge_index` type. Should be either :obj:`adj` or :obj:`edge_index`. If :obj:`adj`, the edge_index of the batch will be a :obj:`SparseTensor`, otherwise :obj:`torch.Tensor`.
        num_auxiliary_node_per_output (int): Number of auxiliary nodes per output node.
        num_output_nodes_per_batch: Number of output nodes per batch.
        alpha (float, optional): A :obj:`float` of the teleport probability of pagerank calculation.
        eps (float, optional): A :obj:`float` of error threshold of PPR calculation. The smaller :obj`eps`:, the more accurate results of PPR calculation, but it also takes longer.
        **kwargs (optional): Keywords of :obj:`torch.utils.data.DataLoader`. Specifically, :obj:`shuffle=False` is forced.
    """
    def __init__(self, graph: Data, batch_order: str,
                 output_indices: torch.LongTensor, return_edge_index_type: str,
                 num_auxiliary_node_per_output: int,
                 num_output_nodes_per_batch: int, alpha: float = 0.2,
                 eps: float = 1.e-4, **kwargs):

        self.subgraphs = []
        self.node_wise_out_aux_pairs = []

        assert is_undirected(
            graph.edge_index,
            num_nodes=graph.num_nodes), "Assume the graph to be undirected"
        assert batch_order in ['rand', 'sample', 'order'
                               ], f"Unsupported batch order: {batch_order}"

        if return_edge_index_type == 'adj':
            adj = self.create_adj_from_edge_index(graph.edge_index,
                                                  graph.num_nodes,
                                                  normalization='rw')
        else:
            adj = None

        self.cache_data = kwargs['batch_size'] == 1
        self._batchsize = kwargs['batch_size']
        self.output_indices = output_indices.numpy()
        assert return_edge_index_type in ['adj', 'edge_index']
        self.return_edge_index_type = return_edge_index_type
        self.num_auxiliary_node_per_output = num_auxiliary_node_per_output
        self.num_output_nodes_per_batch = num_output_nodes_per_batch
        self.alpha = alpha
        self.eps = eps

        self.create_loader(graph, adj)

        if len(self.node_wise_out_aux_pairs) > 2:  # <= 2 order makes no sense
            ys = [
                graph.y[out].numpy() for out, _ in self.node_wise_out_aux_pairs
            ]
            sampler = define_sampler(batch_order, ys, graph.y.max().item() + 1)
        else:
            sampler = kwargs['sampler']

        if not self.cache_data:
            cached_graph = graph  # need to cache the original graph
            cached_adj = adj
        else:
            cached_graph = None
            cached_adj = None

        super().__init__(
            self.subgraphs
            if self.cache_data else self.node_wise_out_aux_pairs, cached_graph,
            cached_adj, return_edge_index_type, sampler=sampler, **kwargs)

    def create_loader(self, graph: Data, adj: SparseTensor):
        logging.info("Start PPR calculation")
        ppr_matrix, neighbors = topk_ppr_matrix(
            graph.edge_index, graph.num_nodes, self.alpha, self.eps,
            self.output_indices, self.num_auxiliary_node_per_output)

        ppr_matrix = ppr_matrix[:, self.output_indices]

        logging.info("Getting PPR pairs")
        ppr_pairs = get_pairs(ppr_matrix)

        output_list = prime_orient_merge(ppr_pairs,
                                         self.num_output_nodes_per_batch,
                                         len(self.output_indices))
        output_list = prime_post_process(output_list,
                                         self.num_output_nodes_per_batch)
        node_wise_out_aux_pairs = []

        if isinstance(neighbors, list):
            neighbors = np.array(neighbors, dtype=object)

        _union = lambda inputs: np.unique(np.concatenate(inputs))
        for p in output_list:
            node_wise_out_aux_pairs.append(
                (self.output_indices[p],
                 _union(neighbors[p]).astype(np.int64)))

        indices_complete_check(node_wise_out_aux_pairs, self.output_indices)
        self.node_wise_out_aux_pairs = node_wise_out_aux_pairs

        if self.cache_data:
            self.subgraphs = self.prepare_cache(graph, node_wise_out_aux_pairs,
                                                adj,
                                                self.return_edge_index_type)

    def __getitem__(self, idx):
        return self.subgraphs[
            idx] if self.cache_data else self.node_wise_out_aux_pairs[idx]

    def __len__(self):
        return len(self.node_wise_out_aux_pairs)
