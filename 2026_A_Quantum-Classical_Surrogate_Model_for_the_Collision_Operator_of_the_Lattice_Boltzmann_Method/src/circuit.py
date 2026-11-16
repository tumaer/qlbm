"""Variational quantum circuit of the hybrid BGK collision (PennyLane, state-vector simulation on default.qubit).

Per lattice node and velocity direction i the circuit takes x = [u (d), f_i / rho, p] with p = omega / 2:

    re-uploading register (n qubits)   W(theta_0) prod_l [ S(u s_l) W(theta_l) ]     -> e_i(u) = <Z_out>
    pre-collision qubit                R_Y(arccos(f_i / rho))                        -> <Z> = f_i / rho
    ancilla                            R_Y(2 arcsin(sqrt(p)))                        -> P(|1>) = p
    CSWAP(ancilla; out, pre)           measured: <Z_pre> = (1 - p) f_i / rho + p e_i(u)

so that f*_i = 2 rho <Z_pre> - f_i = f_i + omega (rho e_i(u) - f_i) is the BGK collision with the learned
equilibrium e_i(u) ~ f_i^eq / rho. Only the re-uploading register is trained; it depends on u only.

Re-uploading ansaetze (encoding S = rotations of x_k = u_k s_l, one trainable scale s_l per layer; R_Y except Line+):
    Line           1 qubit,      S(u_1) W, S(u_2) W, ... per layer       -> 1 + L d weight blocks
    Line+          as Line with R_X, R_Y, R_Z for the features 1, 2, 3
    Parallel       d qubits,     (x) S(u_k) then W per layer            -> 1 + L weight blocks
    SuperParallel  P d qubits,   u repeated P times, then W per layer   -> 1 + L weight blocks
Processing W:  SEL  StronglyEntanglingLayers, weights (L_W, n, 3)
               BEL  BasicEntanglerLayers,      weights (L_W, n)
The output qubit is the last qubit of the re-uploading register.
"""

from dataclasses import dataclass

import pennylane as qml
import torch

ANSAETZE = ("Line", "Line+", "Parallel", "SuperParallel")
PROCESSINGS = ("SEL", "BEL")
_LINE_AXES = {"Line": "YYY", "Line+": "XYZ"}


@dataclass(frozen=True)
class CircuitConfig:
    """Architecture of the re-uploading register."""

    ansatz: str = "Parallel"        # Line, Line+, Parallel, SuperParallel
    processing: str = "SEL"         # SEL, BEL
    entangle_layers: int = 1        # L_W, entangling sub-layers per processing block
    layers: int = 6                 # L, re-uploading layers
    blocks: int = 2                 # P, copies of the input (SuperParallel only; 1 for the other ansaetze)
    features: int = 2               # d, spatial dimension

    def __post_init__(self):
        if self.ansatz not in ANSAETZE or self.processing not in PROCESSINGS:
            raise ValueError(f"Unsupported circuit {self.ansatz}/{self.processing}.")
        if self.ansatz == "Line+" and self.features > 3:
            raise ValueError("Line+ supports at most 3 features.")
        if min(self.entangle_layers, self.layers, self.blocks, self.features) < 1:
            raise ValueError("entangle_layers, layers, blocks and features must be >= 1.")
        if self.ansatz != "SuperParallel":
            object.__setattr__(self, "blocks", 1)      # input copies exist for SuperParallel only

    @property
    def line(self) -> bool:
        return self.ansatz in ("Line", "Line+")

    @property
    def repeat(self) -> int:
        return self.blocks if self.ansatz == "SuperParallel" else 1

    @property
    def n_wires(self) -> int:
        """Qubits of the re-uploading register (the full circuit has two more)."""
        return 1 if self.line else self.features * self.repeat

    @property
    def circuit_weight_shape(self) -> tuple:
        n_blocks = 1 + self.layers * (self.features if self.line else 1)
        single = (self.entangle_layers, self.n_wires) + ((3,) if self.processing == "SEL" else ())
        return (n_blocks,) + single

    @property
    def scaling_weight_shape(self) -> tuple:
        return (self.layers, 1)

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "CircuitConfig":
        c = checkpoint["circuit"]
        return cls(c["ansatz"], c["processing"], c["entangle_layers"], c["layers"], c["blocks"],
                   features=int(checkpoint["lattice"]["name"][1]))

    def label(self) -> str:
        p = f" P={self.blocks}" if self.ansatz == "SuperParallel" else ""
        return f"{self.ansatz}{p} {self.processing}x{self.entangle_layers} L={self.layers} ({self.n_wires + 2} qubits)"


def collision_circuit(config: CircuitConfig):
    """The collision circuit as a PennyLane QNode on n + 2 qubits (state-vector simulator default.qubit, torch
    interface, backpropagation): qnode(circuit_weights, scaling_weights, x) -> <Z_pre> (m,) for a batch x (m, d + 2)
    of inputs [u, f_i / rho, omega / 2]."""
    n, d = config.n_wires, config.features
    out_wire, pre_wire, ancilla = n - 1, n, n + 1
    wires = list(range(n))

    def process(weights):
        if config.processing == "SEL":
            qml.StronglyEntanglingLayers(weights, wires=wires)
        else:
            qml.BasicEntanglerLayers(weights, wires=wires)

    @qml.qnode(qml.device("default.qubit", wires=n + 2), interface="torch", diff_method="backprop")
    def circuit(circuit_weights, scaling_weights, x):
        u = x[:, :d]
        process(circuit_weights[0])
        if config.line:
            for layer in range(config.layers):
                for k in range(d):
                    qml.AngleEmbedding(u[:, k:k + 1] * scaling_weights[layer], wires=[0],
                                       rotation=_LINE_AXES[config.ansatz][k])
                    process(circuit_weights[1 + layer * d + k])
        else:
            for layer in range(config.layers):
                qml.AngleEmbedding((u * scaling_weights[layer]).repeat(1, config.repeat), wires=wires, rotation="Y")
                process(circuit_weights[1 + layer])
        qml.RY(torch.acos(x[:, d]), wires=pre_wire)                            # pre-collision population
        qml.RY(2 * torch.asin(torch.sqrt(x[:, d + 1])), wires=ancilla)        # relaxation weight p = omega / 2
        qml.CSWAP(wires=[ancilla, out_wire, pre_wire])                         # weighted sum
        return qml.expval(qml.PauliZ(pre_wire))

    return circuit
