"""Every circuit configuration through the whole pipeline: data -> training -> checkpoint -> loading -> collision with
every post-processing variant (tiny data sets, one epoch)."""

import itertools
from pathlib import Path

import pytest
import torch

from hqlbm import D2Q9, D3Q19, CircuitConfig, QuantumCollision, equilibrium, load_circuits, moments
from hqlbm.circuit import ANSAETZE, PROCESSINGS
from hqlbm.data import generate
from hqlbm.train import TrainingConfig, checkpoint_name, save_checkpoint, train_circuit

TINY = TrainingConfig(epochs=1, batch_size=16)
VARIANTS = [(False, False), (False, True), (True, False), (True, True)]           # plain, CONS, SYM, CONS-SYM
CONFIGS = [(lattice, ansatz, processing) for lattice, ansatz, processing
           in itertools.product((D2Q9, D3Q19), ANSAETZE, PROCESSINGS)]


def config_of(lattice, ansatz, processing):
    return CircuitConfig(ansatz, processing, entangle_layers=2, layers=2, blocks=2, features=lattice.d)


@pytest.fixture(scope="module")
def data():
    return {lattice.name: generate(lattice, n_samples=96, seed=7) for lattice in (D2Q9, D3Q19)}


@pytest.fixture(scope="module")
def orbit_models(tmp_path_factory, data):
    """Orbit circuits (rest, axis, diag) of every configuration, saved with save_checkpoint."""
    folder = tmp_path_factory.mktemp("models")
    for lattice, ansatz, processing in CONFIGS:
        for orbit in ("rest", "axis", "diag"):
            checkpoint = train_circuit(lattice, orbit, config_of(lattice, ansatz, processing), data[lattice.name],
                                       TINY, seed=3, log=lambda message: None)
            save_checkpoint(checkpoint, folder / lattice.name / ansatz / processing, "Test")
    return folder


def near_equilibrium(lattice, n_cells=2):
    gen = torch.Generator().manual_seed(1)
    shape = (n_cells,) + (1,) * (lattice.d - 1)
    u = 0.05 * torch.randn((lattice.d,) + shape, generator=gen, dtype=torch.float64)
    f = equilibrium(torch.ones(shape, dtype=torch.float64), u, lattice)
    return f + 1e-4 * torch.randn(f.shape, generator=gen, dtype=torch.float64) * lattice.w.view((-1,) + (1,) * lattice.d)


@pytest.mark.parametrize("lattice, ansatz, processing", CONFIGS, ids=lambda v: getattr(v, "name", v))
def test_pipeline(orbit_models, lattice, ansatz, processing):
    folder = orbit_models / lattice.name / ansatz / processing
    config = config_of(lattice, ansatz, processing)
    files = sorted(p.name for p in folder.glob("*.pth"))
    assert files == sorted(f"Test_{config.ansatz}{'P2' if ansatz == 'SuperParallel' else ''}_{processing}2_L2_{o}.pth"
                           for o in ("rest", "axis", "diag"))

    circuits = load_circuits(folder, "Test", lattice)
    assert all(c.config == config for c in circuits)

    f = near_equilibrium(lattice)
    rho, u = moments(f, lattice)
    for symmetry, conservation in VARIANTS:
        collision = QuantumCollision(lattice, 1.2, circuits, symmetry, conservation)
        f_post = collision(f, rho, u)
        assert f_post.shape == f.shape and torch.isfinite(f_post).all()
        if conservation:
            rho_post, u_post = moments(f_post, lattice)
            assert torch.allclose(rho_post, rho, atol=1e-14, rtol=0)
            assert torch.allclose(rho_post * u_post, rho * u, atol=1e-14, rtol=0)
        if symmetry:                                    # equivariance under one non-trivial group element
            g = len(lattice.group) - 1
            perm = torch.empty(lattice.q, dtype=torch.long)
            perm[lattice.sigma[g]] = torch.arange(lattice.q)
            rho_g, u_g = moments(f[perm], lattice)
            assert torch.allclose(collision(f[perm], rho_g, u_g), f_post[perm], atol=1e-14, rtol=0)


def test_per_direction_models(tmp_path, data):
    """One circuit per direction (no orbit compression), loaded with orbit_compression=False."""
    config = config_of(D2Q9, "Line", "SEL")
    for i in range(D2Q9.q):
        save_checkpoint(train_circuit(D2Q9, i, config, data["D2Q9"], TINY, seed=i, log=lambda m: None), tmp_path, "Dir")
    assert sorted(p.name for p in tmp_path.glob("*.pth")) == sorted(f"Dir_Line_SEL2_L2_d2q{i}.pth" for i in range(9))
    circuits = load_circuits(tmp_path, "Dir", D2Q9, orbit_compression=False)
    assert [c.members for c in circuits] == sorted([[i] for i in range(9)])
    f = near_equilibrium(D2Q9)
    rho, u = moments(f, D2Q9)
    for symmetry, conservation in VARIANTS:
        assert torch.isfinite(QuantumCollision(D2Q9, 1.2, circuits, symmetry, conservation)(f, rho, u)).all()
    with pytest.raises(FileNotFoundError):
        load_circuits(tmp_path, "Dir", D2Q9)                                 # no orbit circuits with this tag


def test_training_is_reproducible(data):
    config = config_of(D2Q9, "Parallel", "SEL")
    a, b = (train_circuit(D2Q9, "axis", config, data["D2Q9"], TINY, seed=5, log=lambda m: None) for _ in range(2))
    c = train_circuit(D2Q9, "axis", config, data["D2Q9"], TINY, seed=6, log=lambda m: None)
    assert torch.equal(a["circuit_weights"], b["circuit_weights"]) and torch.equal(a["scaling_weights"], b["scaling_weights"])
    assert not torch.equal(a["circuit_weights"], c["circuit_weights"])


def test_checkpoint_contents(data):
    config = config_of(D3Q19, "SuperParallel", "BEL")
    checkpoint = train_circuit(D3Q19, "diag", config, data["D3Q19"], TINY, seed=1, log=lambda m: None)
    assert checkpoint_name("X", checkpoint) == "X_SuperParallelP2_BEL2_L2_diag.pth"
    assert CircuitConfig.from_checkpoint(checkpoint) == config
    assert checkpoint["circuit"]["backend"] == "PennyLane default.qubit"
    assert checkpoint["circuit"]["qubits"] == 8 and checkpoint["circuit"]["parameters"] == 3 * 2 * 6 + 2
    assert checkpoint["dataset"] == data["D3Q19"]["settings"]
    assert checkpoint["lattice"] == {"name": "D3Q19", "orbit_compression": True, "target": "diag",
                                     "reference_direction": 7, "directions": D3Q19.orbits["diag"]}
    assert sorted(checkpoint["training"]["test_rel_errors"]) == D3Q19.orbits["diag"]


def structure(value):
    """Keys and value types of a checkpoint (nested), to compare file formats."""
    if isinstance(value, dict):
        return {key: structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return ["list", sorted({type(item).__name__ for item in value})]
    return type(value).__name__


def test_shipped_files_have_the_training_format(data):
    """The example circuits in models/ are stored in exactly the format train_circuit / save_checkpoint write,
    under the name checkpoint_name gives them."""
    models = Path(__file__).resolve().parents[1] / "models"
    for lattice, tag in ((D2Q9, "Corgi"), (D3Q19, "LionDog")):
        for path in sorted(models.glob(f"{tag}_*.pth")):
            shipped = torch.load(path, weights_only=True)
            fresh = train_circuit(lattice, shipped["lattice"]["target"], CircuitConfig.from_checkpoint(shipped),
                                  data[lattice.name], TINY, seed=1, log=lambda m: None)
            assert structure(shipped) == structure(fresh), path.name
            assert shipped["dataset"] == data[lattice.name]["settings"] | {"num_samples": 100_000, "seed": 42}
            assert checkpoint_name(tag, shipped) == path.name
