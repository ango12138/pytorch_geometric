import torch
from torch_geometric.utils import (to_undirected, is_undirected,
                                   contains_self_loops)
from torch_geometric.utils.negative_sampling import edge_index_to_vector
from torch_geometric.utils.negative_sampling import vector_to_edge_index
from torch_geometric.utils.negative_sampling import \
    is_neg_samp_feasible
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling,
                                   batched_negative_sampling)


def is_negative(edge_index, neg_edge_index, size, bipartite):
    adj = torch.zeros(size, dtype=torch.bool)
    neg_adj = torch.zeros(size, dtype=torch.bool)

    adj[edge_index[0], edge_index[1]] = True
    neg_adj[neg_edge_index[0], neg_edge_index[1]] = True

    if not bipartite:
        arange = torch.arange(size[0])
        assert neg_adj[arange, arange].sum() == 0

    return (adj & neg_adj).sum() == 0


def test_edge_index_to_vector_and_vice_versa():
    # Create a fully-connected graph:
    N = 10
    row = torch.arange(N).view(-1, 1).repeat(1, N).view(-1)
    col = torch.arange(N).view(1, -1).repeat(N, 1).view(-1)
    edge_index = torch.stack([row, col], dim=0)

    idx, population = edge_index_to_vector(edge_index, (N, N), bipartite=True)
    assert population == N * N
    assert idx.tolist() == list(range(population))
    edge_index2 = vector_to_edge_index(idx, (N, N), bipartite=True)
    assert is_undirected(edge_index2)
    assert edge_index.tolist() == edge_index2.tolist()

    idx, population = edge_index_to_vector(edge_index, (N, N), bipartite=False)
    assert population == N * N - N
    assert idx.tolist() == list(range(population))
    mask = edge_index[0] != edge_index[1]  # Remove self-loops.
    edge_index2 = vector_to_edge_index(idx, (N, N), bipartite=False)
    assert is_undirected(edge_index2)
    assert edge_index[:, mask].tolist() == edge_index2.tolist()

    idx, population = edge_index_to_vector(edge_index, (N, N), bipartite=False,
                                           force_undirected=True)
    assert population == (N * (N + 1)) / 2 - N
    assert idx.tolist() == list(range(population))
    mask = edge_index[0] != edge_index[1]  # Remove self-loops.
    edge_index2 = vector_to_edge_index(idx, (N, N), bipartite=False,
                                       force_undirected=True)
    assert is_undirected(edge_index2)
    assert edge_index[:, mask].tolist() == to_undirected(edge_index2).tolist()


def test_negative_sampling():
    edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])

    neg_edge_index = negative_sampling(edge_index)
    assert neg_edge_index.size(1) == edge_index.size(1)
    assert is_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)

    neg_edge_index = negative_sampling(edge_index, num_neg_samples=2)
    assert neg_edge_index.size(1) == 2
    assert is_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)

    edge_index = to_undirected(edge_index)
    neg_edge_index = negative_sampling(edge_index, force_undirected=True)
    assert neg_edge_index.size(1) == edge_index.size(1) - 1
    assert is_undirected(neg_edge_index)
    assert is_negative(edge_index, neg_edge_index, (4, 4), bipartite=False)


def test_bipartite_negative_sampling():
    edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])

    neg_edge_index = negative_sampling(edge_index, num_nodes=(3, 4))
    assert neg_edge_index.size(1) == edge_index.size(1)
    assert is_negative(edge_index, neg_edge_index, (3, 4), bipartite=True)

    neg_edge_index = negative_sampling(edge_index, num_nodes=(3, 4),
                                       num_neg_samples=2)
    assert neg_edge_index.size(1) == 2
    assert is_negative(edge_index, neg_edge_index, (3, 4), bipartite=True)


def test_structured_negative_sampling():
    edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])

    i, j, k = structured_negative_sampling(edge_index)
    assert i.size(0) == edge_index.size(1)
    assert j.size(0) == edge_index.size(1)
    assert k.size(0) == edge_index.size(1)

    adj = torch.zeros(4, 4, dtype=torch.bool)
    adj[i, j] = 1

    neg_adj = torch.zeros(4, 4, dtype=torch.bool)
    neg_adj[i, k] = 1
    assert (adj & neg_adj).sum() == 0


def test_structured_negative_sampling_without_self_loops():
    edge_index = torch.LongTensor([[0, 0, 1, 1, 2], [1, 2, 0, 2, 1]])
    i, j, k = structured_negative_sampling(edge_index, num_nodes=4,
                                           contains_neg_self_loops=False)
    neg_edge_index = torch.vstack([i, k])
    assert not contains_self_loops(neg_edge_index)


def test_check_feasibility_of_negative_sampling():
    edge_index = torch.LongTensor(
        [[0, 0, 1, 1, 2, 2, 2], [1, 2, 0, 2, 0, 1, 1]])
    assert not is_neg_samp_feasible(edge_index, 'structure', 3, False)
    assert is_neg_samp_feasible(edge_index, 'structure', 3, True)
    assert is_neg_samp_feasible(edge_index, 'structure', 4, False)
    assert not is_neg_samp_feasible(edge_index, 'common', 3, False)
    assert is_neg_samp_feasible(edge_index, 'common', 3, True)
    assert is_neg_samp_feasible(edge_index, 'common', 4, False)
    edge_index = torch.LongTensor([[0, 0, 1, 1, 2], [1, 2, 0, 2, 0]])
    assert is_neg_samp_feasible(edge_index, 'common', 3, False)


def test_batched_negative_sampling():
    edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
    edge_index = torch.cat([edge_index, edge_index + 4], dim=1)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    neg_edge_index = batched_negative_sampling(edge_index, batch)
    assert neg_edge_index.size(1) <= edge_index.size(1)

    adj = torch.zeros(8, 8, dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True

    neg_adj = torch.zeros(8, 8, dtype=torch.bool)
    neg_adj[neg_edge_index[0], neg_edge_index[1]] = True
    assert (adj & neg_adj).sum() == 0
    assert neg_adj[:4, 4:].sum() == 0
    assert neg_adj[4:, :4].sum() == 0
