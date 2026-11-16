"""2D Taylor-Green vortex (D2Q9, periodic) with the 2D example circuits (Corgi) and the classical BGK reference.

Setup: periodic square of N x N cells (cell centres), rho = 1, exact solution with k = 2 pi / N,
F(t) = exp(-2 nu k^2 t):
    u = -U cos(kx) sin(ky) F,   v = U sin(kx) cos(ky) F,   p = -U^2 / 4 (cos 2kx + cos 2ky) F^2.
Diffusive scaling U = 0.08 * 16 / N, nu = U N / Re, run until F = exp(-1). Defaults N = 32, Re = 20. For N >= 13
the velocity stays inside the training range of the circuits (|u| <= 0.1).

Observed (final state): velocity field, velocity_error = relative L2 error of the velocity against the
exact solution.

    python examples/taylor_green_2d.py [--n 32] [--re 20]
"""

import math
import sys

import torch

from common import Setup, relative_l2, run_example
from hqlbm import D2Q9, equilibrium, moments

CASE, LATTICE, MODEL = "taylor_green_2d", D2Q9, "Corgi"


def arguments(p):
    p.add_argument("--n", type=int, default=32, help="cells per direction")
    p.add_argument("--re", type=float, default=20.0, help="Reynolds number U N / nu")


def setup(args) -> Setup:
    u0 = 0.08 * 16 / args.n
    nu = u0 * args.n / args.re
    return Setup(shape=(args.n, args.n), omega=1.0 / (3.0 * nu + 0.5),
                 steps=round(args.n ** 2 / (8 * math.pi ** 2 * nu)), flow=f"Re{args.re:g}",
                 parameters={"u0": u0, "nu": nu})


def exact(s: Setup, step: int):
    """Exact velocity (2, N, N) and density (N, N) at time step `step`."""
    n, u0, nu = s.shape[0], s.parameters["u0"], s.parameters["nu"]
    x = torch.arange(n, dtype=torch.float64) + 0.5
    X, Y = torch.meshgrid(x, x, indexing="ij")
    k = 2 * math.pi / n
    decay = math.exp(-2 * nu * k ** 2 * step)
    u = torch.stack([-torch.cos(k * X) * torch.sin(k * Y), torch.sin(k * X) * torch.cos(k * Y)]) * u0 * decay
    p = -0.25 * u0 ** 2 * (torch.cos(2 * k * X) + torch.cos(2 * k * Y)) * decay ** 2
    return u, 1.0 + 3.0 * p


def initial(s: Setup) -> torch.Tensor:
    u, rho = exact(s, 0)
    return equilibrium(rho, u, LATTICE)


def observe(step: int, f: torch.Tensor, s: Setup) -> dict:
    _, u = moments(f, LATTICE)
    return {"velocity_error": relative_l2(u, exact(s, step)[0]), "velocity": u}


if __name__ == "__main__":
    run_example(sys.modules[__name__])
