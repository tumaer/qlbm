"""Physical properties of the hybrid collision with the shipped example circuits."""

from pathlib import Path

import pytest
import torch

from hqlbm import BGK, D2Q9, D3Q19, QuantumCollision, equilibrium, load_circuits, moments

MODELS = Path(__file__).resolve().parents[1] / "models"
CASES = [(D2Q9, "Corgi", (4, 3)), (D3Q19, "LionDog", (2, 2, 1))]


def near_equilibrium(lattice, shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    rho = 1 + 0.02 * (torch.rand(shape, generator=gen, dtype=torch.float64) - 0.5)
    u = 0.1 / lattice.d ** 0.5 * (2 * torch.rand((lattice.d,) + shape, generator=gen, dtype=torch.float64) - 1)
    noise = 1e-3 * torch.randn((lattice.q,) + shape, generator=gen, dtype=torch.float64)
    return equilibrium(rho, u, lattice) + noise * lattice.w.view((-1,) + (1,) * lattice.d)


@pytest.mark.parametrize("lattice, tag, shape", CASES)
def test_example_circuits_cover_all_directions(lattice, tag, shape):
    circuits = load_circuits(MODELS, tag, lattice)
    assert len(circuits) == 3
    assert sorted(i for c in circuits for i in c.members) == list(range(lattice.q))


@pytest.mark.parametrize("lattice, tag, shape", CASES)
@pytest.mark.parametrize("symmetry", [False, True])
def test_conservation(lattice, tag, shape, symmetry):
    f = near_equilibrium(lattice, shape)
    rho, u = moments(f, lattice)
    f_post = QuantumCollision.from_folder(MODELS, tag, lattice, 1.3, symmetry, conservation=True)(f, rho, u)
    rho_post, u_post = moments(f_post, lattice)
    assert torch.allclose(rho_post, rho, atol=1e-14, rtol=0)
    assert torch.allclose(rho_post * u_post, rho * u, atol=1e-14, rtol=0)


@pytest.mark.parametrize("lattice, tag, shape", CASES)
def test_symmetry_is_exact(lattice, tag, shape):
    """collide(g f) = g collide(f) for the elements g of the hyperoctahedral group (node-local; all 8 elements
    in 2D, every 8th of the 48 elements in 3D)."""
    f = near_equilibrium(lattice, shape)
    collision = QuantumCollision.from_folder(MODELS, tag, lattice, 0.8, symmetry=True, conservation=False)
    rho, u = moments(f, lattice)
    f_post = collision(f, rho, u)
    for g in range(0, len(lattice.group), 1 if lattice.d == 2 else 8):
        perm = torch.empty(lattice.q, dtype=torch.long)
        perm[lattice.sigma[g]] = torch.arange(lattice.q)          # (g f)_{sigma_g(i)} = f_i
        fg = f[perm]
        rho_g, u_g = moments(fg, lattice)
        assert torch.allclose(collision(fg, rho_g, u_g), f_post[perm], atol=1e-14, rtol=0)


@pytest.mark.parametrize("lattice, tag, shape", CASES)
@pytest.mark.parametrize("symmetry, conservation", [(False, False), (True, True)])
def test_close_to_bgk(lattice, tag, shape, symmetry, conservation):
    f = near_equilibrium(lattice, shape)
    rho, u = moments(f, lattice)
    reference = BGK(lattice, 1.3)(f, rho, u)
    f_post = QuantumCollision.from_folder(MODELS, tag, lattice, 1.3, symmetry, conservation)(f, rho, u)
    # bound from the relative test errors of the trained circuits (order 1e-4)
    assert float((f_post - reference).abs().max()) < 3e-4 * float(f.abs().max())
