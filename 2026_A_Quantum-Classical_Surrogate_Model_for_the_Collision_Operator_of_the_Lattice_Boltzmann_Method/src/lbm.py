"""Classical lattice Boltzmann building blocks (lattice units, float64).

Populations are stored as f (q, *grid), e.g. (9, nx, ny) for D2Q9 or (19, nx, ny, nz) for D3Q19, the velocity as
u (d, *grid). One time step is collide -> stream (with the boundary conditions):

    rho, u = moments(f)
    f      = streaming(collision(f, rho, u), rho)

Collision operators share the call signature collision(f, rho, u) -> f*. BGK below is the classical reference;
hqlbm.collision.QuantumCollision is the hybrid quantum-classical operator.
"""

import torch

from hqlbm.lattice import Lattice

# order in which the wall faces are applied; at edges and corners the later face sets the wall velocity
_FACES = ("x+", "x-", "y+", "y-", "z+", "z-")


def moments(f: torch.Tensor, lattice: Lattice):
    """Density rho (*grid) and velocity u (d, *grid)."""
    rho = f.sum(0)
    u = torch.einsum("qa,q...->a...", lattice.c, f) / rho
    return rho, u


def equilibrium(rho: torch.Tensor, u: torch.Tensor, lattice: Lattice) -> torch.Tensor:
    """Second-order equilibrium f_i^eq = w_i rho (1 + 3 c_i.u + 9/2 (c_i.u)^2 - 3/2 |u|^2), shape (q, *grid)."""
    cu = torch.einsum("qa,a...->q...", lattice.c, u)
    w = lattice.w.view((-1,) + (1,) * rho.dim())
    return rho * w * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * (u ** 2).sum(0))


class BGK:
    """Classical BGK collision f* = f + omega (f^eq(rho, u) - f), the reference for the quantum collision."""

    def __init__(self, lattice: Lattice, omega: float):
        self.lattice, self.omega = lattice, omega

    def __call__(self, f, rho, u):
        return f + self.omega * (equilibrium(rho, u, self.lattice) - f)


class Streaming:
    """Streaming on a box of cells (cell centres), periodic unless a face is a wall.

    walls: {face: wall velocity} with face in "x-", "x+", "y-", "y+" (, "z-", "z+"), both faces of an axis or none
    (axes without walls are periodic), e.g. a lid-driven cavity
    {"x-": (0, 0), "x+": (0, 0), "y-": (0, 0), "y+": (u_lid, 0)}. Walls use the half-way bounce-back with a
    moving-wall correction,
        f_opp(i)(x, t + 1) = f*_i(x, t) - 6 w_i rho(x, t) c_i . u_wall,
    for every cell x whose neighbour x + c_i lies behind the wall.
    """

    def __init__(self, lattice: Lattice, shape, walls: dict = None):
        self.lattice = lattice
        self.shape = tuple(shape)
        self.shifts = [tuple(int(v) for v in ci) for ci in lattice.c]
        self.dims = tuple(range(lattice.d))
        self.mask = None
        if walls:
            self._build_walls(walls)

    def _build_walls(self, walls: dict):
        lat = self.lattice
        unknown = set(walls) - set(_FACES[:2 * lat.d])
        if unknown:
            raise ValueError(f"Unknown wall faces {sorted(unknown)} for {lat.name}.")
        unpaired = [face for face in walls if face[0] + ("-" if face[1] == "+" else "+") not in walls]
        if unpaired:
            raise ValueError(f"Wall faces {sorted(unpaired)} without the opposite face: an axis is either periodic "
                             f"or bounded by walls on both sides.")
        index = torch.meshgrid(*[torch.arange(n) for n in self.shape], indexing="ij")
        crosses = torch.zeros((lat.q,) + self.shape, dtype=torch.bool)       # x + c_i behind a wall
        wall_term = torch.zeros((lat.q,) + self.shape, dtype=torch.float64)  # 6 w_i c_i . u_wall
        for face in _FACES:
            if face not in walls:
                continue
            axis, side = "xyz".index(face[0]), face[1]
            u_wall = torch.as_tensor(walls[face], dtype=torch.float64)
            for i, ci in enumerate(lat.c):
                step = int(ci[axis])
                if (side == "+" and step <= 0) or (side == "-" and step >= 0):
                    continue
                target = index[axis] + step
                hit = target >= self.shape[axis] if side == "+" else target < 0
                crosses[i] |= hit
                wall_term[i] = torch.where(hit, 6.0 * lat.w[i] * (ci @ u_wall), wall_term[i])
        # incoming population j = opp(i) is replaced at the cells where c_i crosses a wall
        self.mask = crosses[lat.opposite]
        self.wall_term = wall_term[lat.opposite]
        self.source = lat.opposite

    def __call__(self, f_post: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        f = torch.stack([torch.roll(f_post[i], self.shifts[i], self.dims) for i in range(self.lattice.q)])
        if self.mask is not None:
            f = torch.where(self.mask, f_post[self.source] - self.wall_term * rho, f)
        return f


def step(f: torch.Tensor, collision, streaming: Streaming) -> torch.Tensor:
    """One time step: moments -> collision -> streaming (with boundary conditions)."""
    rho, u = moments(f, streaming.lattice)
    return streaming(collision(f, rho, u), rho)


def run(f: torch.Tensor, collision, streaming: Streaming, steps: int, every: int = 0, observe=None):
    """Advance f by `steps` time steps. If given, observe(n, f) is called after the last step and, for every > 0, at
    n = 0 and every `every` steps; its return values are collected. The run stops early when the populations become
    non-finite; the stopped state is then observed as the last one.

    Returns (f, observations, completed), with completed < steps if the run was stopped."""
    observations = []
    for n in range(steps + 1):
        if observe and (n == steps or (every and n % every == 0)):
            observations.append(observe(n, f))
        if n == steps:
            break
        f = step(f, collision, streaming)
        if not torch.isfinite(f).all():
            if observe:
                observations.append(observe(n + 1, f))
            return f, observations, n + 1
    return f, observations, steps
