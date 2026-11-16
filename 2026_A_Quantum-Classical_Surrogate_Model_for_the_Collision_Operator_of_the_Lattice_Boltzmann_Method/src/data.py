"""Synthetic training data: random near-equilibrium states and their exact BGK collision.

Per sample: rho ~ U(rho_min, rho_max), |u| ~ U(0, u_max) with the direction given by uniformly distributed angles
(2D: theta in [0, 2 pi); 3D: theta in [0, 2 pi), phi in [0, pi], i.e. not uniform on the sphere), a Gaussian
non-equilibrium part f_neq ~ N(0, sigma^2) projected onto populations without mass and momentum, and
    f = f^eq(rho, u) + f_neq,   f* = f + omega (f^eq(rho', u') - f)    (rho', u' recomputed from f)
with the fixed relaxation rate omega = 1 (OMEGA). The relaxation is encoded exactly by the circuit (ancilla and
CSWAP); the trained register represents the equilibrium e(u) = f^eq / rho only, and the residual of the circuit
output is omega / 2 (e_theta(u) - e(u)). One fixed omega therefore suffices for the training, and the trained
circuit is used with any omega in (0, 2] in the simulation.
"""

import math

import torch

from hqlbm.lattice import Lattice
from hqlbm.lbm import equilibrium, moments

OMEGA = 1.0          # relaxation rate of the training data (enters the circuit output only as the exact weight)


def remove_mass_and_momentum(f_neq: torch.Tensor, lattice: Lattice) -> torch.Tensor:
    """Orthogonal projection of a perturbation f_neq (q, ...) onto populations without mass and momentum,
        f_neq_i <- f_neq_i - (1 / Q) sum_j f_neq_j - (1 / K) c_i . sum_j c_j f_neq_j,
    with the constants of the lattice: Q = number of velocities, K = sum_i c_ix^2 (D2Q9: Q = 9, K = 6;
    D3Q19: Q = 19, K = 10)."""
    Q, K = lattice.q, float(lattice.c[:, 0].pow(2).sum())
    mass = f_neq.sum(0)
    momentum = torch.einsum("qa,q...->a...", lattice.c, f_neq)
    return f_neq - mass / Q - torch.einsum("qa,a...->q...", lattice.c, momentum) / K


def generate(lattice: Lattice, n_samples: int = 100_000, u_max: float = 0.1, rho_range=(0.98, 1.02),
             sigma: float = 1e-3, seed: int = 42) -> dict:
    """{u (N, d), f (N, q), f_post (N, q), omega (N, 1) = OMEGA, settings}, populations divided by the density;
    settings records the arguments and omega (stored in the checkpoints by hqlbm.train)."""
    gen = torch.Generator().manual_seed(seed)
    rho = torch.rand(n_samples, dtype=torch.float64, generator=gen) * (rho_range[1] - rho_range[0]) + rho_range[0]
    speed = torch.rand(n_samples, dtype=torch.float64, generator=gen) * u_max
    theta = torch.rand(n_samples, dtype=torch.float64, generator=gen) * 2 * math.pi
    phi = torch.rand(n_samples, dtype=torch.float64, generator=gen) * math.pi
    if lattice.d == 2:
        u = torch.stack([speed * torch.cos(theta), speed * torch.sin(theta)])
    else:
        u = torch.stack([speed * torch.sin(phi) * torch.cos(theta), speed * torch.sin(phi) * torch.sin(theta),
                         speed * torch.cos(phi)])

    f_neq = remove_mass_and_momentum(
        torch.normal(0.0, sigma, size=(lattice.q, n_samples), dtype=torch.float64, generator=gen), lattice)

    f = equilibrium(rho, u, lattice) + f_neq
    rho, u = moments(f, lattice)
    omega = torch.full((n_samples, 1), OMEGA, dtype=torch.float64)
    f_post = f + omega.T * (equilibrium(rho, u, lattice) - f)
    settings = {"lattice": lattice.name, "num_samples": n_samples, "u_max": u_max, "rho_range": list(rho_range),
                "sigma": sigma, "omega": OMEGA, "seed": seed}
    return {"u": u.T.contiguous(), "f": (f / rho).T.contiguous(), "f_post": (f_post / rho).T.contiguous(),
            "omega": omega, "settings": settings}
