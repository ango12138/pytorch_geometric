import pytest
import torch
import torch_scatter

from torch_geometric.utils import scatter

torch.manual_seed(12345)


@pytest.mark.parametrize('reduce', ['sum', 'add', 'mean', 'min', 'max', 'mul'])
def test_scatter(reduce):
    src = torch.randn(8, 100, 32)
    index = torch.randint(0, 10, (100, ), dtype=torch.long)

    out1 = scatter(src, index, dim=1, reduce=reduce)
    out2 = torch_scatter.scatter(src, index, dim=1, reduce=reduce)
    assert torch.allclose(out1, out2, atol=1e-6)


@pytest.mark.parametrize('reduce', ['sum', 'add', 'min', 'max', 'mul'])
def test_scatter_backward(reduce):
    src = torch.randn(8, 100, 32).requires_grad_(True)
    index = torch.randint(0, 10, (100, ), dtype=torch.long)
    out = torch.randn(8, 10, 32)

    out = scatter(src, index, dim=1, out=out.clone(), reduce=reduce)
    assert src.grad is None
    out.mean().backward()
    assert src.grad is not None


@pytest.mark.parametrize('reduce', ['sum', 'add', 'min', 'max', 'mul'])
def test_scatter_with_out(reduce):
    src = torch.randn(8, 100, 32)
    index = torch.randint(0, 10, (100, ), dtype=torch.long)
    out = torch.randn(8, 10, 32)

    out1 = scatter(src, index, dim=1, out=out.clone(), reduce=reduce)
    out2 = torch_scatter.scatter(src, index, dim=1, out=out.clone(),
                                 reduce=reduce)
    assert torch.allclose(out1, out2, atol=1e-6)
