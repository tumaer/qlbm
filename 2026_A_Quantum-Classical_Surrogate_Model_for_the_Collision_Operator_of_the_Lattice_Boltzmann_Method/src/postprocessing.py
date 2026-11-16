"""Optional post-processing of the circuit output: symmetry (SYM) and conservation (CONS).

Both steps are switched off by default; QuantumCollision(..., symmetry=True, conservation=True) applies them
(CONS+SYM = both). The circuit output of a node is z_i = <Z_pre> for the direction i, and the post-collision
populations are f*_i = 2 rho z_i - f_i.

SYM  symmetrisation over the hyperoctahedral group G of the lattice (signed permutations of the axes, 8 elements in
     2D, 48 in 3D). For every g in G the node state is transformed (u -> g u, populations permuted with g), the
     circuits are evaluated, the result is transformed back, and the |G| results are averaged:
         z_j = 1/|G| sum_g z_{sigma_g(j)}(g u, f_j),
     so the collision commutes exactly with every lattice rotation and reflection. Transforming the populations
     with g and back means that direction j receives the circuit of direction s = sigma_g(j) with the velocity
     R_s^T g u (R_s: orbit rotation of s) and its own population f_j. symmetry_plan lists these inputs.

CONS orthogonal projection of the circuit output onto the states with the mass and momentum of the input,
         z' = P z + (I - P) f / rho,    P = I - C^T (C C^T)^-1 C,    C = [1; c_x; c_y (; c_z)]   ((1 + d) x q),
     i.e. the smallest change of z (least squares) for which sum_i z'_i = sum_i f_i / rho and
     sum_i c_i z'_i = sum_i c_i f_i / rho, so f* conserves mass and momentum exactly.

CONS+SYM applies CONS to the symmetrised output. P commutes with the group permutations, so this equals applying CONS
to every transformed state before averaging.
"""

import torch

from hqlbm.lattice import Lattice


def symmetry_plan(lattice: Lattice, circuits: list, symmetry: bool) -> tuple:
    """Inputs to evaluate per circuit, as [(directions j (T,), velocity transformations (T, d, d))] with one entry per
    (group element, direction) served by that circuit, and the number of group elements to average over. Without
    symmetry the group is {identity}, i.e. every direction is evaluated once with its orbit rotation R_j^T u."""
    owner = {i: (k, c.rotations[i]) for k, c in enumerate(circuits) for i in c.members}
    group = range(len(lattice.group)) if symmetry else [None]
    directions, transforms = [[] for _ in circuits], [[] for _ in circuits]
    for g in group:
        matrix = lattice.group[g] if symmetry else torch.eye(lattice.d, dtype=torch.float64)
        for j in range(lattice.q):
            s = int(lattice.sigma[g, j]) if symmetry else j          # direction whose circuit serves j
            k, rotation = owner[s]
            directions[k].append(j)
            transforms[k].append(rotation.T @ matrix)
    plan = [(torch.tensor(j), torch.stack(t)) for j, t in zip(directions, transforms)]
    return plan, len(group)


def conservation_projector(lattice: Lattice) -> torch.Tensor:
    """P = I - C^T (C C^T)^-1 C, the orthogonal projector onto population vectors without mass and momentum."""
    C = torch.cat([torch.ones(1, lattice.q, dtype=torch.float64), lattice.c.T])
    return torch.eye(lattice.q, dtype=torch.float64) - C.T @ torch.linalg.solve(C @ C.T, C)


def conserve(z: torch.Tensor, f_rho: torch.Tensor, projector: torch.Tensor) -> torch.Tensor:
    """CONS: z' = P z + (I - P) f / rho = f / rho + P (z - f / rho) for z, f_rho of shape (q, ...)."""
    return f_rho + torch.einsum("ij,j...->i...", projector, z - f_rho)
