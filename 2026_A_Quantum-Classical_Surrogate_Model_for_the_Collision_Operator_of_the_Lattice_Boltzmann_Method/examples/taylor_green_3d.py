"""3D Taylor-Green vortex (D3Q19, periodic) with the 3D example circuits (LionDog) and the classical BGK reference.

Setup: periodic cube of N^3 cells (cell centres x in [0, 1)^3, k = 2 pi), rho = 1,
    u = u0 ( sin(kx) cos(ky) cos(kz), -cos(kx) sin(ky) cos(kz), 0 ),   u0 = 0.1 (lattice units),
relaxation rate omega, Re = u0 N / nu with nu = (1 / omega - 1/2) / 3. Defaults N = 20, omega = 1.3, 40 steps.

Observed (every step, from step 0): kinetic_energy = 0.5 <rho |u|^2> / u0^2, enstrophy = <|curl u|^2> / u0^2
(periodic central differences, cell size 1 / N), velocity field.

    python examples/taylor_green_3d.py [--omega 1.3] [--steps 40] [--n 20]
"""

import math
import sys

import torch

from common import Setup, run_example
from hqlbm import D3Q19, equilibrium, moments

CASE, LATTICE, MODEL = "taylor_green_3d", D3Q19, "LionDog"
U0 = 0.1


def arguments(p):
    p.add_argument("--omega", type=float, default=1.3, help="relaxation rate")
    p.add_argument("--steps", type=int, default=40, help="time steps")
    p.add_argument("--n", type=int, default=20, help="cells per direction")


def setup(args) -> Setup:
    nu = (1.0 / args.omega - 0.5) / 3.0
    return Setup(shape=(args.n,) * 3, omega=args.omega, steps=args.steps, flow=f"omega{args.omega:g}", every=1,
                 parameters={"u0": U0, "nu": nu, "re": U0 * args.n / nu})


def initial(s: Setup) -> torch.Tensor:
    x = (torch.arange(s.shape[0], dtype=torch.float64) + 0.5) / s.shape[0]
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    k = 2 * math.pi
    u = U0 * torch.stack([torch.sin(k * X) * torch.cos(k * Y) * torch.cos(k * Z),
                          -torch.cos(k * X) * torch.sin(k * Y) * torch.cos(k * Z), torch.zeros_like(X)])
    return equilibrium(torch.ones_like(X), u, LATTICE)


def observe(step: int, f: torch.Tensor, s: Setup) -> dict:
    rho, u = moments(f, LATTICE)
    dx = 1.0 / s.shape[0]

    def d(g, axis):
        return (torch.roll(g, -1, axis) - torch.roll(g, 1, axis)) / (2 * dx)

    curl = torch.stack([d(u[2], 1) - d(u[1], 2), d(u[0], 2) - d(u[2], 0), d(u[1], 0) - d(u[0], 1)])
    return {"kinetic_energy": float(0.5 * (rho * (u ** 2).sum(0)).mean() / U0 ** 2),
            "enstrophy": float((curl ** 2).sum(0).mean() / U0 ** 2),
            "velocity": u}


if __name__ == "__main__":
    run_example(sys.modules[__name__])
