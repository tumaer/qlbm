"""Hybrid quantum-classical BGK collision with the trained circuits.

For every node and direction i the collision circuit (hqlbm.circuit) returns z_i = <Z_pre> for the input
[u, f_i / rho, omega / 2], and the post-collision population is f*_i = 2 rho z_i - f_i (plain circuit output, the
default). The optional post-processing steps SYM and CONS are implemented in hqlbm.postprocessing.

Orbit compression: one circuit per orbit (rest / axis / diag) is trained on the representative direction c_ref of
the orbit; member i uses the same circuit with the rotated input R_i^T u (R_i c_ref = c_i). Per-direction models
(one circuit per direction) are supported as well.
"""

from pathlib import Path

import torch

from hqlbm.circuit import CircuitConfig, collision_circuit
from hqlbm.lattice import Lattice
from hqlbm.postprocessing import conservation_projector, conserve, symmetry_plan
from hqlbm.train import checkpoint_name


class TrainedCircuit:
    """One trained circuit (PennyLane QNode with its weights) and the directions it serves."""

    def __init__(self, checkpoint: dict, lattice: Lattice, name: str = ""):
        self.name = name
        self.config = CircuitConfig.from_checkpoint(checkpoint)
        self.circuit_weights = checkpoint["circuit_weights"].to(torch.float64)
        self.scaling_weights = checkpoint["scaling_weights"].to(torch.float64)
        self.qnode = collision_circuit(self.config)
        meta = checkpoint["lattice"]
        if meta["name"] != lattice.name:
            raise ValueError(f"{name}: trained for {meta['name']}, not {lattice.name}.")
        self.members = list(meta["directions"])
        self.rotations = {i: lattice.rotation(meta["reference_direction"], i) for i in self.members}

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """<Z_pre> for the inputs x (m, d + 2) -> (m,), evaluated in batches of max(1, 2^19 / 2^qubits) inputs."""
        batch_size = max(1, 2 ** 19 // 2 ** (self.config.n_wires + 2))
        return torch.cat([self.qnode(self.circuit_weights, self.scaling_weights, part).view(-1)
                          for part in x.split(batch_size)])


def load_circuits(folder, tag: str, lattice: Lattice, orbit_compression: bool = True) -> list:
    """The trained circuits <tag>_<circuit>_<target>.pth in folder (hqlbm.train.checkpoint_name): one per orbit
    (orbit_compression) or one per direction."""
    circuits = []
    for path in sorted(Path(folder).glob(f"{tag}_*.pth")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint_name(tag, checkpoint) != path.name:       # another tag that starts with <tag>_
            continue
        if checkpoint["lattice"]["orbit_compression"] == orbit_compression:
            circuits.append(TrainedCircuit(checkpoint, lattice, path.name))
    covered = sorted(i for circuit in circuits for i in circuit.members)
    if covered != list(range(lattice.q)):
        kind = "orbit" if orbit_compression else "per-direction"
        raise FileNotFoundError(f"The {kind} circuits '{tag}' in {folder} do not cover the {lattice.q} directions "
                                f"of {lattice.name} exactly once (found {[c.name for c in circuits]}).")
    return circuits


class QuantumCollision:
    """Hybrid collision operator, collision(f, rho, u) -> f* (same interface as hqlbm.lbm.BGK).

    symmetry, conservation: the optional post-processing steps SYM and CONS (hqlbm.postprocessing), off by default."""

    def __init__(self, lattice: Lattice, omega: float, circuits: list, symmetry: bool = False,
                 conservation: bool = False):
        if not 0.0 < omega <= 2.0:
            raise ValueError(f"omega = {omega} outside (0, 2]: the ancilla encodes omega / 2 as a probability.")
        self.lattice, self.omega, self.circuits = lattice, omega, circuits
        self.symmetry, self.conservation = symmetry, conservation
        self.plan, self.group_size = symmetry_plan(lattice, circuits, symmetry)       # SYM (or plain evaluation)
        self.projector = conservation_projector(lattice) if conservation else None   # CONS

    @classmethod
    def from_folder(cls, folder, tag: str, lattice: Lattice, omega: float, symmetry: bool = False,
                    conservation: bool = False, orbit_compression: bool = True) -> "QuantumCollision":
        return cls(lattice, omega, load_circuits(folder, tag, lattice, orbit_compression), symmetry, conservation)

    def __repr__(self) -> str:
        constraints = "+".join(name for name, on in (("SYM", self.symmetry), ("CONS", self.conservation)) if on)
        return (f"QuantumCollision({self.lattice.name}, omega={self.omega:.4f}, {len(self.circuits)} circuits "
                f"[{self.circuits[0].config.label()}], {constraints or 'no post-processing'}, "
                f"{self.group_size * self.lattice.q} circuit evaluations per node)")

    @torch.no_grad()
    def __call__(self, f: torch.Tensor, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        q, d = self.lattice.q, self.lattice.d
        f_rho = (f / rho).reshape(q, -1)
        u = u.reshape(d, -1).T
        n = u.shape[0]
        # circuit outputs z = <Z_pre>, summed over the group elements of the plan (one element without SYM)
        z = torch.zeros(q, n, dtype=torch.float64)
        for circuit, (directions, transforms) in zip(self.circuits, self.plan):
            x = torch.cat([torch.einsum("tab,nb->tna", transforms, u), f_rho[directions].unsqueeze(-1),
                           torch.full((len(directions), n, 1), 0.5 * self.omega, dtype=torch.float64)], dim=-1)
            z.index_add_(0, directions, circuit(x.reshape(-1, d + 2)).view(len(directions), n))
        z /= self.group_size                                       # SYM: group average
        if self.conservation:
            z = conserve(z, f_rho, self.projector)                 # CONS: mass and momentum
        return (2.0 * z * rho.reshape(1, -1)).view(f.shape) - f
