import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import TAGConv
from torch_geometric.testing import is_full_test


def test_tag_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    value = torch.rand(row.size(0))
    adj2 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    adj1 = adj2.set_value(None)

    conv = TAGConv(16, 32)
    assert conv.__repr__() == 'TAGConv(16, 32, K=3)'
    out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
<<<<<<< HEAD
    assert torch.allclose(conv(x, adj1.t()), out1)
    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 32)
    assert torch.allclose(conv(x, adj2.t()), out2)
=======
    assert torch.allclose(
        conv(x, adj1.t()),
        out1,
    )
    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 32)
    assert torch.allclose(
        conv(x, adj2.t()),
        out2,
    )
>>>>>>> 6d868987bf6879d01d72a714a92d5bb0bb30b049

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()
        assert jit(x, edge_index, value).tolist() == out2.tolist()

        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
<<<<<<< HEAD
        assert torch.allclose(jit(x, adj1.t()), out1)
        assert torch.allclose(jit(x, adj2.t()), out2)
=======
        assert torch.allclose(
            jit(x, adj1.t()),
            out1,
        )
        assert torch.allclose(
            jit(x, adj2.t()),
            out2,
        )
>>>>>>> 6d868987bf6879d01d72a714a92d5bb0bb30b049


def test_static_tag_conv():
    x = torch.randn(3, 4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = TAGConv(16, 32)
    out = conv(x, edge_index)
    assert out.size() == (3, 4, 32)
