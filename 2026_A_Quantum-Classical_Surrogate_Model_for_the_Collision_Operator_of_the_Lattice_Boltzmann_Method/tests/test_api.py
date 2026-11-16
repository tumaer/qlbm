"""Configuration checks, error handling, lattice symmetry tools, training data and wall boundaries."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from hqlbm import (BGK, D2Q9, D3Q19, CircuitConfig, QuantumCollision, Streaming, equilibrium, load_circuits, moments,
                   run, step)
from hqlbm.collision import TrainedCircuit
from hqlbm.data import generate, remove_mass_and_momentum
from hqlbm.postprocessing import conservation_projector, conserve
from hqlbm.train import TrainingConfig, make_checkpoint, train_circuit

MODELS = Path(__file__).resolve().parents[1] / "models"


# --------------------------------------------------------------------------------------------------- configuration

@pytest.mark.parametrize("kwargs", [{"ansatz": "Ring"}, {"processing": "XYZ"}, {"ansatz": "Line+", "features": 4}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CircuitConfig(**kwargs)


@pytest.mark.parametrize("ansatz, wires", [("Line", 1), ("Line+", 1), ("Parallel", 3), ("SuperParallel", 6)])
def test_register_size(ansatz, wires):
    config = CircuitConfig(ansatz, "SEL", 2, 4, 2, 3)
    assert config.n_wires == wires and str(config.n_wires + 2) in config.label()


def test_shipped_models():
    """Example circuits: architecture as documented, every direction covered once."""
    corgi = load_circuits(MODELS, "Corgi", D2Q9)
    liondog = load_circuits(MODELS, "LionDog", D3Q19)
    assert {c.config for c in corgi} == {CircuitConfig("Parallel", "SEL", 1, 6, 2, 2)}
    assert {c.config for c in liondog} == {CircuitConfig("SuperParallel", "BEL", 3, 6, 2, 3)}
    for circuit in corgi + liondog:
        checkpoint = torch.load(MODELS / circuit.name, weights_only=True)
        assert checkpoint["circuit"]["backend"] == "PennyLane default.qubit"


def test_loader_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_circuits(tmp_path, "Corgi", D2Q9)                                 # empty folder
    with pytest.raises(ValueError):
        load_circuits(MODELS, "Corgi", D3Q19)                                  # 2D circuits on a 3D lattice
    (tmp_path / "Corgi_Parallel_SEL1_L6_axis.pth").write_bytes((MODELS / "Corgi_Parallel_SEL1_L6_axis.pth").read_bytes())
    with pytest.raises(FileNotFoundError):
        load_circuits(tmp_path, "Corgi", D2Q9)                                 # rest and diag missing


# --------------------------------------------------------------------------------------------------------- lattices

@pytest.mark.parametrize("lattice", [D2Q9, D3Q19])
def test_orbits_and_rotations(lattice):
    assert [len(lattice.orbits[o]) for o in ("rest", "axis", "diag")] == ({9: [1, 4, 4], 19: [1, 6, 12]}[lattice.q])
    for members in lattice.orbits.values():
        for i in members:
            R = lattice.rotation(members[0], i)
            assert torch.equal(R @ lattice.c[members[0]], lattice.c[i])
            assert lattice.group_index(R) in range(len(lattice.group))
    for g in range(len(lattice.group)):
        assert lattice.group_index(lattice.group[g]) == g


# ----------------------------------------------------------------------------------------------- post-processing

@pytest.mark.parametrize("lattice", [D2Q9, D3Q19])
def test_conservation_projector(lattice):
    P = conservation_projector(lattice)
    C = torch.cat([torch.ones(1, lattice.q, dtype=torch.float64), lattice.c.T])
    assert torch.allclose(P @ P, P) and torch.allclose(P, P.T) and torch.allclose(C @ P, torch.zeros_like(C), atol=1e-14)
    f, z = torch.rand(lattice.q, 5, dtype=torch.float64), torch.rand(lattice.q, 5, dtype=torch.float64)
    assert torch.allclose(C @ conserve(z, f, P), C @ f)


# ------------------------------------------------------------------------------------------------------------- data

@pytest.mark.parametrize("lattice", [D2Q9, D3Q19])
def test_projection_constants(lattice):
    f_neq = remove_mass_and_momentum(torch.randn(lattice.q, 50, dtype=torch.float64), lattice)
    assert torch.allclose(f_neq.sum(0), torch.zeros(50, dtype=torch.float64), atol=1e-14)
    assert torch.allclose(lattice.c.T @ f_neq, torch.zeros(lattice.d, 50, dtype=torch.float64), atol=1e-14)


@pytest.mark.parametrize("lattice", [D2Q9, D3Q19])
def test_data_ranges_and_seed(lattice):
    data = generate(lattice, n_samples=400, u_max=0.05, rho_range=(0.99, 1.01), seed=3)
    assert data["u"].norm(dim=1).max() <= 0.05 + 1e-12
    assert torch.equal(data["omega"], torch.ones(400, 1, dtype=torch.float64)) and data["settings"]["omega"] == 1.0
    e = equilibrium(torch.ones(400, dtype=torch.float64), data["u"].T, lattice).T             # f* = f^eq for omega = 1
    assert torch.allclose(data["f_post"], e, atol=1e-15, rtol=0)
    assert data["settings"]["lattice"] == lattice.name and data["settings"]["seed"] == 3
    assert torch.equal(generate(lattice, n_samples=400, seed=3)["f"], generate(lattice, n_samples=400, seed=3)["f"])
    assert not torch.equal(generate(lattice, n_samples=400, seed=3)["f"], generate(lattice, n_samples=400, seed=4)["f"])


# ------------------------------------------------------------------------------------------------------------ walls

def test_closed_box_3d_conserves_mass():
    n = 5
    walls = {face: (0.0, 0.0, 0.0) for face in ("x-", "x+", "y-", "y+", "z-", "z+")}
    streaming = Streaming(D3Q19, (n, n, n), walls)
    f = equilibrium(torch.ones(n, n, n, dtype=torch.float64), 0.03 * torch.randn(3, n, n, n, dtype=torch.float64), D3Q19)
    mass = f.sum()
    for _ in range(10):
        f = step(f, BGK(D3Q19, 1.1), streaming)
    assert torch.isclose(f.sum(), mass, atol=1e-12, rtol=0)


def test_moving_lid_drives_the_flow():
    n = 8
    streaming = Streaming(D2Q9, (n, n), {"x-": (0, 0), "x+": (0, 0), "y-": (0, 0), "y+": (0.05, 0.0)})
    f = equilibrium(torch.ones(n, n, dtype=torch.float64), torch.zeros(2, n, n, dtype=torch.float64), D2Q9)
    for _ in range(30):
        f = step(f, BGK(D2Q9, 1.5), streaming)
    _, u = moments(f, D2Q9)
    assert u[0, :, -1].mean() > 0.01 and u[0, :, -1].mean() > u[0, :, 0].mean()   # fluid below the lid moves along it


def test_invalid_wall_face():
    with pytest.raises(ValueError):
        Streaming(D2Q9, (4, 4), {"z+": (0.0, 0.0)})


@pytest.mark.parametrize("omega", [0.0, -0.5, 2.1])
def test_invalid_omega(omega):
    with pytest.raises(ValueError, match="omega"):
        QuantumCollision.from_folder(MODELS, "Corgi", D2Q9, omega)


# ----------------------------------------------------------------------------------------- review regressions

def test_config_ranges_and_blocks():
    for kwargs in ({"layers": 0}, {"entangle_layers": 0}, {"ansatz": "SuperParallel", "blocks": 0}):
        with pytest.raises(ValueError):
            CircuitConfig(**kwargs)
    assert CircuitConfig("Parallel", blocks=3) == CircuitConfig("Parallel", blocks=1)      # P only for SuperParallel
    assert CircuitConfig("SuperParallel", blocks=3).n_wires == 6


def test_single_wall_face_is_rejected():
    with pytest.raises(ValueError, match="opposite face"):
        Streaming(D2Q9, (4, 4), {"y+": (0.1, 0.0)})


def test_run_observes_the_stopped_state():
    f = equilibrium(torch.ones(4, 4, dtype=torch.float64), torch.zeros(2, 4, 4, dtype=torch.float64), D2Q9)
    diverging = lambda f, rho, u: f * float("nan")                                        # noqa: E731
    _, observations, completed = run(f, diverging, Streaming(D2Q9, (4, 4)), 5, 0, lambda n, f: n)
    assert completed == 1 and observations == [1]


def test_large_register_is_evaluated():
    """Registers with more than 17 qubits (batch size 2^19 / 2^qubits < 1) are evaluated one input at a time."""
    config = CircuitConfig("SuperParallel", "BEL", 1, 1, blocks=6, features=3)                # 18 + 2 qubits
    weights = torch.zeros(config.circuit_weight_shape, dtype=torch.float64)
    checkpoint = make_checkpoint(weights, torch.ones(config.scaling_weight_shape, dtype=torch.float64), config, D3Q19,
                                 "rest", {}, {})
    x = torch.tensor([[0.01, 0.02, 0.0, 0.3, 0.0]] * 2, dtype=torch.float64)                # p = 0: returns f / rho
    assert torch.allclose(TrainedCircuit(checkpoint, D3Q19)(x), x[:, 3], atol=1e-14)


def test_loader_ignores_other_tags(tmp_path):
    for path in MODELS.glob("Corgi_*.pth"):
        shutil.copy(path, tmp_path / path.name)
        shutil.copy(path, tmp_path / path.name.replace("Corgi_", "Corgi_v2_"))               # tag Corgi_v2
    assert len(load_circuits(tmp_path, "Corgi", D2Q9)) == 3


def test_training_without_finite_validation_loss():
    data = generate(D2Q9, n_samples=40)
    with pytest.raises(RuntimeError, match="no finite validation loss"):
        train_circuit(D2Q9, "rest", CircuitConfig(layers=1), data, TrainingConfig(epochs=0), seed=1, log=lambda m: None)


def test_direction_target_is_stored_as_int():
    config = CircuitConfig(layers=1)
    checkpoint = make_checkpoint(torch.zeros(config.circuit_weight_shape, dtype=torch.float64),
                                 torch.zeros(config.scaling_weight_shape, dtype=torch.float64), config, D2Q9,
                                 np.int64(3), {}, {})
    assert type(checkpoint["lattice"]["target"]) is int and checkpoint["lattice"]["directions"] == [3]
