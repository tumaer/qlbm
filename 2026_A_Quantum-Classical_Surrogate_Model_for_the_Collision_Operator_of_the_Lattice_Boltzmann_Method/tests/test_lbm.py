"""Classical LBM building blocks, lattices and the training data."""

import pytest
import torch

from hqlbm import BGK, D2Q9, D3Q19, Streaming, equilibrium, moments, run, step
from hqlbm.data import generate

LATTICES = [D2Q9, D3Q19]


@pytest.mark.parametrize("lattice", LATTICES)
def test_lattice(lattice):
    assert torch.isclose(lattice.w.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(torch.einsum("q,qa,qb->ab", lattice.w, lattice.c, lattice.c),
                          lattice.cs2 * torch.eye(lattice.d, dtype=torch.float64))
    assert torch.equal(lattice.c[lattice.opposite], -lattice.c)
    assert len(lattice.group) == {2: 8, 3: 48}[lattice.d]
    for g, matrix in enumerate(lattice.group):                  # group[g] c_i = c_{sigma[g, i]}
        assert torch.equal(lattice.c @ matrix.T, lattice.c[lattice.sigma[g]])


@pytest.mark.parametrize("lattice", LATTICES)
def test_equilibrium_moments_and_bgk_conservation(lattice):
    shape = (4,) * lattice.d
    rho = 1 + 0.01 * torch.rand(shape, dtype=torch.float64)
    u = 0.05 * torch.randn((lattice.d,) + shape, dtype=torch.float64)
    f_eq = equilibrium(rho, u, lattice)
    rho_eq, u_eq = moments(f_eq, lattice)
    assert torch.allclose(rho_eq, rho) and torch.allclose(u_eq, u)
    f = f_eq + 1e-3 * torch.randn_like(f_eq)
    r, v = moments(f, lattice)
    r_post, v_post = moments(BGK(lattice, 1.5)(f, r, v), lattice)
    assert torch.allclose(r_post, r) and torch.allclose(r_post * v_post, r * v)


@pytest.mark.parametrize("lattice", LATTICES)
def test_periodic_streaming(lattice):
    shape = (5, 4, 3)[:lattice.d]
    f = torch.rand((lattice.q,) + shape, dtype=torch.float64)
    streamed = Streaming(lattice, shape)(f, f.sum(0))
    assert torch.isclose(streamed.sum(), f.sum())
    i = lattice.orbits["axis"][0]                                 # c_1 = (1, 0 (, 0)): shift by one cell in x
    assert torch.equal(streamed[i], torch.roll(f[i], 1, 0))


def test_closed_cavity_conserves_mass():
    n = 8
    walls = {face: (0.0, 0.0) for face in ("x-", "x+", "y-", "y+")}
    streaming = Streaming(D2Q9, (n, n), walls)
    u = 0.05 * torch.randn(2, n, n, dtype=torch.float64)
    f = equilibrium(torch.ones(n, n, dtype=torch.float64), u, D2Q9)
    mass = f.sum()
    for _ in range(20):
        f = step(f, BGK(D2Q9, 1.2), streaming)
    assert torch.isclose(f.sum(), mass, atol=1e-12, rtol=0)


def test_run():
    n = 6
    f = equilibrium(torch.ones(n, n, dtype=torch.float64), 0.05 * torch.randn(2, n, n, dtype=torch.float64), D2Q9)
    streaming, collision = Streaming(D2Q9, (n, n)), BGK(D2Q9, 1.2)
    reference = f
    for _ in range(7):
        reference = step(reference, collision, streaming)
    f_end, observed, completed = run(f, collision, streaming, 7, every=3, observe=lambda k, g: k)
    assert completed == 7 and observed == [0, 3, 6, 7] and torch.equal(f_end, reference)
    _, _, completed = run(f, BGK(D2Q9, float("nan")), streaming, 5)        # non-finite -> stops after step 1
    assert completed == 1


@pytest.mark.parametrize("lattice", LATTICES)
def test_training_data(lattice):
    data = generate(lattice, n_samples=500)
    assert torch.allclose(data["f"].sum(1), torch.ones(500, dtype=torch.float64))
    assert torch.allclose(data["f"] @ lattice.c, data["u"])
    assert torch.allclose(data["f_post"].sum(1), torch.ones(500, dtype=torch.float64))
    assert data["u"].norm(dim=1).max() <= 0.1 + 1e-12
