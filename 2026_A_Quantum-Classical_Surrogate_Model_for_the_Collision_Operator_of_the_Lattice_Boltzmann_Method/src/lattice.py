"""DdQq velocity sets and their hyperoctahedral symmetry group.

Ordering of the discrete velocities (the direction indices used throughout, e.g. in the checkpoints):

    D2Q9   c_0 = 0, c_1..4 = axis directions, c_5..8 = diagonals
    D3Q19  c_0 = 0, c_1..6 = axis directions, c_7..18 = face diagonals

The velocities of one lattice split into orbits of equal length (|c|^2 = 0, 1, 2), named "rest", "axis" and
"diag". The hyperoctahedral group (all signed permutations of the coordinate axes, 2^d d! elements) maps each orbit
onto itself; it is used for the orbit compression (one circuit per orbit) and for the symmetrisation of the
learned equilibrium.
"""

import itertools

import torch

_VELOCITIES = {
    "D2Q9": [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
    "D3Q19": [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
              (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (-1, 1, 0), (-1, -1, 0), (-1, 0, 1), (-1, 0, -1),
              (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)],
}
_WEIGHTS = {  # weight per |c|^2 = 0, 1, 2
    "D2Q9": (4 / 9, 1 / 9, 1 / 36),
    "D3Q19": (1 / 3, 1 / 18, 1 / 36),
}
ORBIT_NAMES = ("rest", "axis", "diag")


class Lattice:
    """A DdQq velocity set (float64 tensors).

    Attributes:
        name      "D2Q9" or "D3Q19"
        d, q      spatial dimension, number of velocities
        c         (q, d) velocities, integer valued
        w         (q,) weights
        opposite  (q,) index of -c_i
        orbits    {"rest": [...], "axis": [...], "diag": [...]} velocity indices per orbit
        group     (G, d, d) signed permutation matrices, G = 2^d d!
        sigma     (G, q) velocity permutation of the group: group[g] @ c_i = c_{sigma[g, i]}
    """

    cs2 = 1.0 / 3.0

    def __init__(self, name: str):
        if name not in _VELOCITIES:
            raise ValueError(f"Unknown lattice '{name}', choose from {sorted(_VELOCITIES)}.")
        self.name = name
        self.c = torch.tensor(_VELOCITIES[name], dtype=torch.float64)
        self.q, self.d = self.c.shape
        norm2 = self.c.pow(2).sum(1).long()
        self.w = torch.tensor([_WEIGHTS[name][int(n)] for n in norm2], dtype=torch.float64)
        self.orbits = {ORBIT_NAMES[n]: torch.nonzero(norm2 == n).flatten().tolist() for n in range(3)}
        self.opposite = torch.tensor([self.index(-ci) for ci in self.c])

        perms = torch.tensor(list(itertools.permutations(range(self.d))))
        P = torch.zeros(len(perms), self.d, self.d, dtype=torch.float64).scatter_(2, perms.unsqueeze(-1), 1.0)
        S = torch.diag_embed(torch.tensor(list(itertools.product([1.0, -1.0], repeat=self.d)), dtype=torch.float64))
        self.group = (P[:, None] @ S[None, :]).reshape(-1, self.d, self.d)
        self._group_index = {self._key(m): g for g, m in enumerate(self.group)}
        self.sigma = torch.tensor([[self.index(m @ ci) for ci in self.c] for m in self.group])

    def __repr__(self) -> str:
        return f"Lattice({self.name})"

    @staticmethod
    def _key(matrix: torch.Tensor) -> tuple:
        return tuple(int(v) for v in matrix.round().flatten())

    def index(self, velocity: torch.Tensor) -> int:
        """Index i with c_i == velocity."""
        hits = torch.nonzero((self.c == velocity).all(1)).flatten()
        if len(hits) != 1:
            raise ValueError(f"{velocity.tolist()} is not a velocity of {self.name}.")
        return int(hits[0])

    def group_index(self, matrix: torch.Tensor) -> int:
        """Index g with group[g] == matrix (a signed permutation matrix)."""
        return self._group_index[self._key(matrix)]

    def rotation(self, i_ref: int, i: int) -> torch.Tensor:
        """Signed permutation R with R c_ref = c_i that maps the representative of an orbit onto its member i
        (orbit compression); the identity for the rest velocity. Of the several such R, the first one in the
        order permutations x signs (+ before -) is taken; training and collision use the same R."""
        if not self.c[i_ref].any():
            return torch.eye(self.d, dtype=torch.float64)
        for perm in itertools.permutations(range(self.d)):
            for signs in itertools.product([1.0, -1.0], repeat=self.d):
                R = torch.zeros(self.d, self.d, dtype=torch.float64)
                R[range(self.d), perm] = torch.tensor(signs, dtype=torch.float64)
                if torch.equal(R @ self.c[i_ref], self.c[i]):
                    return R
        raise ValueError(f"No signed permutation maps c_{i_ref} to c_{i}.")


D2Q9 = Lattice("D2Q9")
D3Q19 = Lattice("D3Q19")
LATTICES = {"D2Q9": D2Q9, "D3Q19": D3Q19}
