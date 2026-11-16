"""Double shear layer (D2Q9, periodic) with the 2D example circuits (Corgi) and the classical BGK reference.

Setup: domain [0, 2] x [0, 1] with 200 x 100 cells (dx = 0.01, cell centres), u0 = 0.08 (lattice units), rho = 1,
    u_x = u0 tanh(80 (y - 1/4)) for y < 1/2,  u0 tanh(80 (3/4 - y)) otherwise,     u_y = u0 / 2 sin(2 pi (x + 1/4)),
nu = 100 u0 / Re (lattice units, length scale 1), dt = 0.01, i.e. t = step / 100. Defaults Re = 100, 1000 steps.

Observed (step 0 and every 50 steps): kinetic_energy = 0.5 <rho |u|^2> / u0^2, enstrophy = <omega_z^2> / u0^2
(periodic central differences, spacing dx), velocity field.

    python examples/double_shear_layer.py [--re 100] [--steps 1000]
"""

import math
import sys

import torch

from common import Setup, run_example
from hqlbm import D2Q9, equilibrium, moments

CASE, LATTICE, MODEL = "double_shear_layer", D2Q9, "Corgi"
U0, DX = 0.08, 0.01


def arguments(p):
    p.add_argument("--re", type=float, default=100.0, help="Reynolds number u0 L / nu")
    p.add_argument("--steps", type=int, default=1000, help="time steps")


def setup(args) -> Setup:
    nu = 100.0 * U0 / args.re
    return Setup(shape=(200, 100), omega=1.0 / (3.0 * nu + 0.5), steps=args.steps, flow=f"Re{args.re:g}", every=50,
                 parameters={"u0": U0, "nu": nu, "dx": DX})


def initial(s: Setup) -> torch.Tensor:
    x = (torch.arange(s.shape[0], dtype=torch.float64) + 0.5) * DX
    y = (torch.arange(s.shape[1], dtype=torch.float64) + 0.5) * DX
    X, Y = torch.meshgrid(x, y, indexing="ij")
    ux = torch.where(Y >= 0.5, U0 * torch.tanh(80.0 * (0.75 - Y)), U0 * torch.tanh(80.0 * (Y - 0.25)))
    uy = 0.5 * U0 * torch.sin(2.0 * math.pi * (X + 0.25))
    return equilibrium(torch.ones_like(X), torch.stack([ux, uy]), LATTICE)


def observe(step: int, f: torch.Tensor, s: Setup) -> dict:
    rho, u = moments(f, LATTICE)

    def d(g, axis):
        return (torch.roll(g, -1, axis) - torch.roll(g, 1, axis)) / (2 * DX)

    return {"kinetic_energy": float(0.5 * (rho * (u ** 2).sum(0)).mean() / U0 ** 2),
            "enstrophy": float(((d(u[1], 0) - d(u[0], 1)) ** 2).mean() / U0 ** 2),
            "velocity": u}


if __name__ == "__main__":
    run_example(sys.modules[__name__])
