import torch

from torch_geometric.nn import EdgePooling, maximal_matching_cluster
from torch_geometric.testing import is_full_test
from torch_geometric.utils import scatter


def test_compute_edge_score_softmax():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    raw = torch.randn(edge_index.size(1))
    e = EdgePooling.compute_edge_score_softmax(raw, edge_index, 6)
    assert torch.all(e >= 0) and torch.all(e <= 1)

    # Test whether all incoming edge scores sum up to one.
    assert torch.allclose(
        scatter(e, edge_index[1], reduce='sum'),
        torch.ones(6),
    )

    if is_full_test():
        jit = torch.jit.script(EdgePooling.compute_edge_score_softmax)
        assert torch.allclose(jit(raw, edge_index, 6), e)


def test_compute_edge_score_tanh():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    raw = torch.randn(edge_index.size(1))
    e = EdgePooling.compute_edge_score_tanh(raw, edge_index, 6)
    assert torch.all(e >= -1) and torch.all(e <= 1)
    assert torch.all(torch.argsort(raw) == torch.argsort(e))

    if is_full_test():
        jit = torch.jit.script(EdgePooling.compute_edge_score_tanh)
        assert torch.allclose(jit(raw, edge_index, 6), e)


def test_compute_edge_score_sigmoid():
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    raw = torch.randn(edge_index.size(1))
    e = EdgePooling.compute_edge_score_sigmoid(raw, edge_index, 6)
    assert torch.all(e >= 0) and torch.all(e <= 1)
    assert torch.all(torch.argsort(raw) == torch.argsort(e))

    if is_full_test():
        jit = torch.jit.script(EdgePooling.compute_edge_score_sigmoid)
        assert torch.allclose(jit(raw, edge_index, 6), e)


def test_edge_pooling():
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [-1.0]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4, 0]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 0])

    op = EdgePooling(in_channels=1)
    assert str(op) == 'EdgePooling(1)'

    # Setting parameters fixed so we can test the expected outcome:
    op.lin.weight.data.fill_(1.)
    op.lin.bias.data.fill_(0.)

    # Test pooling:
    new_x, new_edge_index, new_batch, unpool_info = op(x, edge_index, batch)

    assert new_x.size(0) == new_batch.size(0) == 4
    assert new_edge_index.tolist() == [[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 2, 0]]
    assert new_batch.tolist() == [0, 0, 1, 0]

    if is_full_test():
        jit = torch.jit.script(op)
        out = jit(x, edge_index, batch)
        assert torch.allclose(new_x, out[0])
        assert torch.equal(new_edge_index, out[1])
        assert torch.equal(new_batch, out[2])

    # Test unpooling:
    out = op.unpool(new_x, unpool_info)
    assert out[0].size() == x.size()
    assert out[0].tolist() == [[1], [1], [5], [5], [9], [9], [-1]]
    assert torch.equal(out[1], edge_index)
    assert torch.equal(out[2], batch)

    if is_full_test():
        jit = torch.jit.export(op)
        out = jit.unpool(new_x, unpool_info)
        assert out[0].size() == x.size()
        assert out[0].tolist() == [[1], [1], [5], [5], [9], [9], [-1]]
        assert torch.equal(out[1], edge_index)
        assert torch.equal(out[2], batch)

    # Test edge cases.
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])
    new_x, new_edge_index, new_batch, _ = op(x, edge_index, batch)

    assert new_x.size(0) == new_batch.size(0) == 3
    assert new_batch.tolist() == [0, 0, 1]
    assert new_edge_index.tolist() == [[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]]


def test_maximal_matching_cluster():
    n, m = 5, 12
    index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
                          [1, 4, 0, 2, 4, 1, 3, 2, 4, 0, 1, 3]]).long()

    perm1 = torch.arange(m)
    match1, clus1 = maximal_matching_cluster(edge_index=index, num_nodes=n,
                                             perm=perm1)
    exp_match1 = torch.tensor([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).bool()
    exp_clus1 = torch.tensor([0, 0, 1, 1, 2]).long()

    assert torch.equal(match1, exp_match1)
    assert torch.equal(clus1, exp_clus1)

    perm2 = perm1.roll(-1, 0)
    match2, clus2 = maximal_matching_cluster(edge_index=index, num_nodes=n,
                                             perm=perm2)
    exp_match2 = torch.tensor([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).bool()
    exp_clus2 = torch.tensor([0, 1, 1, 2, 0]).long()

    assert torch.equal(match2, exp_match2)
    assert torch.equal(clus2, exp_clus2)

    perm3 = torch.arange(m).flip(-1)
    match3, clus3 = maximal_matching_cluster(edge_index=index, num_nodes=n,
                                             perm=perm3)
    exp_match3 = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]).bool()
    exp_clus3 = torch.tensor([0, 1, 1, 2, 2])

    assert torch.equal(match3, exp_match3)
    assert torch.equal(clus3, exp_clus3)

    if is_full_test():
        jit = torch.jit.script(maximal_matching_cluster)
        assert exp_clus1.equal(jit(index, perm=perm1)[1])
        assert exp_clus2.equal(jit(index, perm=perm2)[1])
        assert exp_clus3.equal(jit(index, perm=perm3)[1])
