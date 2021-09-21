import torch
from torch_geometric.transforms import RandomShear
from torch_geometric.data import Data


def test_random_shear():
    assert RandomShear(0.1).__repr__() == 'RandomShear(0.1)'

    pos = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])

    data = Data(pos=pos)
    data = RandomShear(0)(data)
    assert len(data) == 1
    assert data.pos.tolist() == pos.tolist()

    data = Data(pos=pos)
    data = RandomShear(0.1)(data)
    assert len(data) == 1
    assert data.pos.tolist() != pos.tolist()
