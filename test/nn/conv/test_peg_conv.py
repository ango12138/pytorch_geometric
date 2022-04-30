import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import PEGConv
from torch_geometric.testing import is_full_test


def test_peg_conv():
    node_feature = torch.randn(4, 16)
    pos_encoding = torch.randn(4, 16)
    x = torch.cat((pos_encoding, node_feature), 1)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    value = torch.rand(row.size(0))
    adj2 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    adj1 = adj2.set_value(None)

    conv = PEGConv(in_feats_dim = 16, pos_dim = 16, out_feats_dim = 32)
    assert conv.__repr__() == 'PEGConv(16,16,32)'
    out1 = conv(x, edge_index)
    assert out1.size() == (4, 48)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)
    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 48)
    assert torch.allclose(conv(x, adj2.t()), out2, atol=1e-6)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert jit(x, edge_index).tolist() == out1.tolist()
        assert jit(x, edge_index, value).tolist() == out2.tolist()

        t = '(Tensor, SparseTensor, OptTensor) -> Tensor'
        jit = torch.jit.script(conv.jittable(t))
        assert torch.allclose(jit(x, adj1.t()), out1, atol=1e-6)
        assert torch.allclose(jit(x, adj2.t()), out2, atol=1e-6)

    conv.cached = True
    conv(x, edge_index)
    assert conv(x, edge_index).tolist() == out1.tolist()
    conv(x, adj1.t())
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)