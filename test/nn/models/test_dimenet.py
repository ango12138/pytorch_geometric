import torch
import torch.nn.functional as F

from torch_geometric.nn import (
    BesselBasisLayer,
    DimeNet,
    DimeNetPlusPlus,
    Envelope,
)
from torch_geometric.testing import onlyFullTest


def test_envelope():
    env = Envelope(exponent=5)
    x = torch.randn(10, 3)

    assert env(x).size() == (10, 3)  # Isotonic Layer
    assert env(x).dtype == x.dtype


def test_bessel_basis_layer():
    bbl = BesselBasisLayer(5)
    bbl.reset_parameters()
    x = torch.randn(10, 3)

    assert bbl(x).size() != (10, 3)  # Non-isotonic Layer
    assert bbl(x).dtype == x.dtype


@onlyFullTest
def test_dimenet():
    z = torch.randint(1, 10, (20, ))
    pos = torch.randn(20, 3)

    model = DimeNet(hidden_channels=5, out_channels=1, num_blocks=5)
    model.reset_parameters()

    with torch.no_grad():
        out = model(z, pos)
        assert out.size() == (1, )

        jit = torch.jit.export(model)
        assert torch.allclose(jit(z, pos), out)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    min_loss = float("inf")
    for i in range(100):
        optimizer.zero_grad()
        out = model(z, pos)
        loss = F.l1_loss(out, torch.tensor([1.0]))
        loss.backward()
        optimizer.step()
        min_loss = min(float(loss), min_loss)
    assert min_loss < 2


@onlyFullTest
def test_dimenet_plus_plus():
    z = torch.randint(1, 10, (20, ))
    pos = torch.randn(20, 3)

    model = DimeNetPlusPlus(
        hidden_channels=5,
        out_channels=1,
        num_blocks=5,
        out_emb_channels=3,
        int_emb_size=5,
        basis_emb_size=5,
        num_spherical=5,
        num_radial=5,
        num_before_skip=2,
        num_after_skip=2,
    )
    model.reset_parameters()

    with torch.no_grad():
        out = model(z, pos)
        assert out.size() == (1, )

        jit = torch.jit.export(model)
        assert torch.allclose(jit(z, pos), out)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    min_loss = float("inf")
    for i in range(100):
        optimizer.zero_grad()
        out = model(z, pos)
        loss = F.l1_loss(out, torch.tensor([1.0]))
        loss.backward()
        optimizer.step()
        min_loss = min(float(loss), min_loss)
    assert min_loss < 2
