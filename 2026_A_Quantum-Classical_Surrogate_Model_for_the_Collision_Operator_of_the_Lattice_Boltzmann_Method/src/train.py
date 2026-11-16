"""Training of one collision circuit (one orbit, or one direction).

Input  x = [u (d), f_i / rho, omega / 2],   target  z_i = (f*_i + f_i) / (2 rho)   (= <Z_pre> of the exact BGK),
loss MSE, Adam with a step learning-rate schedule, 80/10/10 random split (fixed seed), uniform initialisation
theta ~ U(-pi, pi), s ~ U(-1, 1); the weights with the lowest validation loss are kept. For an orbit the circuit is
trained on the representative direction (rest c_0, axis c_1, diag: first diagonal) and serves all members.
"""

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from hqlbm.circuit import CircuitConfig, collision_circuit
from hqlbm.lattice import Lattice


@dataclass
class TrainingConfig:
    epochs: int = 9
    batch_size: int = 32
    learning_rate: float = 1e-3
    lr_decay: float = 0.1
    lr_step: int = 3
    split: tuple = (0.8, 0.1, 0.1)
    split_seed: int = 42


def train_circuit(lattice: Lattice, target, circuit: CircuitConfig, data: dict, training: TrainingConfig = None,
                  seed: int = None, log=print) -> dict:
    """Train the circuit of an orbit (target "rest" / "axis" / "diag") or of a direction (target int).
    Returns the checkpoint dict (load it with hqlbm.collision.load_circuits after saving it with save_checkpoint)."""
    training = training or TrainingConfig()
    seed = int(np.random.default_rng().integers(0, 2 ** 31 - 1)) if seed is None else seed
    orbit = isinstance(target, str)
    direction = lattice.orbits[target][0] if orbit else int(target)

    x = torch.cat([data["u"], data["f"][:, direction:direction + 1], 0.5 * data["omega"]], dim=1)
    z = 0.5 * (data["f_post"][:, direction:direction + 1] + data["f"][:, direction:direction + 1])
    train_set, val_set, test_set = random_split(TensorDataset(x, z), training.split,
                                                generator=torch.Generator().manual_seed(training.split_seed))
    loader = DataLoader(train_set, batch_size=training.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    (val_x, val_z), (test_x, test_z) = val_set[:], test_set[:]

    model = collision_circuit(circuit)
    rng = np.random.default_rng(seed)
    circuit_weights = torch.nn.Parameter(torch.tensor(rng.uniform(-np.pi, np.pi, circuit.circuit_weight_shape)))
    scaling_weights = torch.nn.Parameter(torch.tensor(rng.uniform(-1, 1, circuit.scaling_weight_shape)))
    optimiser = torch.optim.Adam([circuit_weights, scaling_weights], lr=training.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=training.lr_step, gamma=training.lr_decay)
    mse = torch.nn.MSELoss()

    name = target if orbit else f"d{lattice.d}q{direction}"
    best, best_cw, best_sw, tic = math.inf, None, None, time.time()
    for epoch in range(1, training.epochs + 1):
        running = 0.0
        for xb, zb in loader:
            optimiser.zero_grad()
            loss = mse(model(circuit_weights, scaling_weights, xb).view(-1, 1), zb)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(xb)
        scheduler.step()
        with torch.no_grad():
            val_loss = mse(model(circuit_weights, scaling_weights, val_x).view(-1, 1), val_z).item()
            if val_loss < best:
                best, best_cw, best_sw = val_loss, circuit_weights.detach().clone(), scaling_weights.detach().clone()
            test_rel = ((test_z - model(circuit_weights, scaling_weights, test_x).view(-1, 1)).abs()
                        / test_z.abs()).mean().item()
        log(f"[{name}] epoch {epoch}/{training.epochs}  train {running / len(train_set):.3e}  val {val_loss:.3e}  "
            f"test rel {test_rel:.3e}  ({time.time() - tic:.0f} s)")

    if best_cw is None:
        raise RuntimeError(f"[{name}] no finite validation loss in {training.epochs} epochs.")

    # relative test error of every direction the circuit serves (orbit members with their rotated inputs)
    members = lattice.orbits[target] if orbit else [direction]
    test_rows = torch.as_tensor(test_set.indices)
    member_errors = {}
    with torch.no_grad():
        for m in members:
            u_m = data["u"][test_rows] @ lattice.rotation(direction, m)                     # rows: R^T u
            x_m = torch.cat([u_m, data["f"][test_rows, m:m + 1], 0.5 * data["omega"][test_rows]], dim=1)
            z_m = 0.5 * (data["f_post"][test_rows, m:m + 1] + data["f"][test_rows, m:m + 1])
            member_errors[m] = ((z_m - model(best_cw, best_sw, x_m).view(-1, 1)).abs() / z_m.abs()).mean().item()

    return make_checkpoint(best_cw, best_sw, circuit, lattice, target, data["settings"],
                           {**asdict(training), "split": list(training.split), "seed": seed,
                            "best_validation_loss": best, "test_rel_errors": member_errors})


def make_checkpoint(circuit_weights, scaling_weights, circuit: CircuitConfig, lattice: Lattice, target,
                    dataset: dict, training: dict) -> dict:
    """The checkpoint of one trained circuit (the format of every .pth file of this package):

        circuit_weights, scaling_weights   the trained parameters
        circuit    ansatz, processing, entangle_layers, layers, blocks (= CircuitConfig), qubits, parameters, backend
        lattice    name, orbit_compression, target (orbit name or direction index), reference_direction, directions
        dataset    training data: lattice, num_samples, u_max, rho_range, sigma, omega, seed (hqlbm.data)
        training   epochs, batch_size, learning_rate, lr_decay, lr_step, split, split_seed, seed (initialisation and
                   shuffling), best_validation_loss, test_rel_errors (mean relative test error per direction)"""
    orbit = isinstance(target, str)
    target = target if orbit else int(target)
    reference = lattice.orbits[target][0] if orbit else target
    return {
        "circuit_weights": circuit_weights,
        "scaling_weights": scaling_weights,
        "circuit": {"ansatz": circuit.ansatz, "processing": circuit.processing,
                    "entangle_layers": circuit.entangle_layers, "layers": circuit.layers, "blocks": circuit.blocks,
                    "qubits": circuit.n_wires + 2, "parameters": circuit_weights.numel() + scaling_weights.numel(),
                    "backend": "PennyLane default.qubit"},
        "lattice": {"name": lattice.name, "orbit_compression": orbit, "target": target,
                    "reference_direction": reference,
                    "directions": lattice.orbits[target] if orbit else [reference]},
        "dataset": dataset,
        "training": training,
    }


def checkpoint_name(tag: str, checkpoint: dict) -> str:
    """<tag>_<circuit>_<target>.pth, with circuit = <ansatz>[P<blocks>]_<processing><L_W>_L<L> and target = the
    orbit (rest / axis / diag) or the direction (d<d>q<i>), e.g. Corgi_Parallel_SEL1_L6_axis.pth."""
    c, lat = checkpoint["circuit"], checkpoint["lattice"]
    blocks = f"P{c['blocks']}" if c["ansatz"] == "SuperParallel" else ""
    target = lat["target"] if lat["orbit_compression"] else f"d{lat['name'][1]}q{lat['target']}"
    return f"{tag}_{c['ansatz']}{blocks}_{c['processing']}{c['entangle_layers']}_L{c['layers']}_{target}.pth"


def save_checkpoint(checkpoint: dict, folder, tag: str) -> Path:
    path = Path(folder) / checkpoint_name(tag, checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path
