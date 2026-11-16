"""Lid-driven cavity (D2Q9) with the 2D example circuits (Corgi) and the classical BGK reference.

Setup: square cavity of N x N cells with no-slip walls (half-way bounce-back) and a lid moving with u_lid = 0.08
(lattice units) at the top (y = N); rho = 1, u = 0 at t = 0; nu = u_lid N / Re. Defaults Re = 100, N = 32,
5000 steps (t = 12.5 N / u_lid).

Observed (final state): max_velocity = max |u| / u_lid, mass_change = <rho> - 1, velocity field and the
centreline profiles ux_centreline = u_x / u_lid on x = N/2 and uy_centreline = u_y / u_lid on y = N/2 (middle cell
for odd N, average of the two middle cells for even N; cell centres y = (j + 1/2) / N).

    python examples/lid_driven_cavity.py [--re 100] [--n 32] [--steps 5000]
"""

import sys

import torch

from common import Setup, run_example
from hqlbm import D2Q9, equilibrium, moments

CASE, LATTICE, MODEL = "lid_driven_cavity", D2Q9, "Corgi"
U_LID = 0.08


def arguments(p):
    p.add_argument("--re", type=float, default=100.0, help="Reynolds number u_lid N / nu")
    p.add_argument("--n", type=int, default=32, help="cells per direction")
    p.add_argument("--steps", type=int, default=5000, help="time steps")


def setup(args) -> Setup:
    nu = U_LID * args.n / args.re
    walls = {"x-": (0.0, 0.0), "x+": (0.0, 0.0), "y-": (0.0, 0.0), "y+": (U_LID, 0.0)}
    return Setup(shape=(args.n, args.n), omega=1.0 / (3.0 * nu + 0.5), steps=args.steps, flow=f"Re{args.re:g}",
                 walls=walls, parameters={"u_lid": U_LID, "nu": nu})


def initial(s: Setup) -> torch.Tensor:
    return equilibrium(torch.ones(s.shape, dtype=torch.float64), torch.zeros((2,) + s.shape, dtype=torch.float64),
                       LATTICE)


def observe(step: int, f: torch.Tensor, s: Setup) -> dict:
    rho, u = moments(f, LATTICE)
    n = s.shape[0]
    middle = [n // 2 - 1, n // 2] if n % 2 == 0 else [n // 2]          # cell(s) on the centreline
    return {"max_velocity": float(u.norm(dim=0).max()) / U_LID,
            "mass_change": float(rho.mean()) - 1.0,
            "velocity": u,
            "ux_centreline": u[0, middle].mean(0) / U_LID,
            "uy_centreline": u[1][:, middle].mean(1) / U_LID}


if __name__ == "__main__":
    run_example(sys.modules[__name__])
