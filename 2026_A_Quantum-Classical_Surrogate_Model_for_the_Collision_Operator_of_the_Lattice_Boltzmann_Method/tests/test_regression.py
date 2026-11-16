"""Regression tests: the accuracy of the shipped circuits and of the example cases must not degrade.

1. Shipped circuits: the training metrics stored in models/*.pth (validation loss, relative test error per
   direction) recompute from regenerated training data (hqlbm.data.generate with the stored settings).
2. Collision output: fingerprints of the post-collision populations of both models in all four post-processing
   variants, recorded with the version of the code verified against the implementation used for the publication.
3. Example cases: errors against the exact solution and differences to BGK of short runs of the example scripts,
   at most 5 % above the recorded values.
4. Training: a short training run reproduces its recorded validation loss.

Slow tier (pytest -m slow): the test errors of all LionDog directions, and the retraining of the Corgi axis circuit
from its stored seed, which reaches the accuracy of the shipped circuit.
"""

import contextlib
import io
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import random_split

from hqlbm import BGK, D2Q9, D3Q19, CircuitConfig, QuantumCollision, equilibrium, moments
from hqlbm.collision import TrainedCircuit
from hqlbm.data import generate
from hqlbm.train import TrainingConfig, train_circuit

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LATTICES = {"Corgi": D2Q9, "LionDog": D3Q19}
FILES = sorted(MODELS.glob("*.pth"))
slow = pytest.mark.slow

sys.path.insert(0, str(ROOT / "examples"))
import common  # noqa: E402  (example driver)


@pytest.fixture(scope="module")
def training_data():
    """The training data of the shipped circuits, regenerated from the settings stored in the checkpoints."""
    cache = {}

    def get(checkpoint):
        settings = dict(checkpoint["dataset"])
        lattice = {"D2Q9": D2Q9, "D3Q19": D3Q19}[settings.pop("lattice")]
        settings.pop("omega")                                            # fixed in hqlbm.data (OMEGA)
        if lattice.name not in cache:
            cache[lattice.name] = generate(lattice, n_samples=settings.pop("num_samples"), **settings)
        return lattice, cache[lattice.name]
    return get


def evaluate(checkpoint, lattice, data, rows, member):
    """(circuit output, target) of direction `member` for the samples `rows`, with the rotated input R^T u."""
    rows = torch.as_tensor(rows)
    reference = checkpoint["lattice"]["reference_direction"]
    x = torch.cat([data["u"][rows] @ lattice.rotation(reference, member), data["f"][rows, member:member + 1],
                   0.5 * data["omega"][rows]], dim=1)
    z = 0.5 * (data["f_post"][rows, member:member + 1] + data["f"][rows, member:member + 1])
    return TrainedCircuit(checkpoint, lattice)(x).view(-1, 1), z          # batched evaluation, as in a simulation


def split(checkpoint, n):
    training = checkpoint["training"]
    _, validation, test = random_split(range(n), training["split"],
                                       generator=torch.Generator().manual_seed(training["split_seed"]))
    return validation.indices, test.indices


# ---------------------------------------------------------------------------------------------- shipped circuits

@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_stored_validation_loss_and_test_error(path, training_data):
    """Validation MSE and the relative test error of the trained direction recompute from regenerated data."""
    checkpoint = torch.load(path, weights_only=True)
    lattice, data = training_data(checkpoint)
    validation, test = split(checkpoint, len(data["u"]))
    reference = checkpoint["lattice"]["reference_direction"]

    output, target = evaluate(checkpoint, lattice, data, validation, reference)
    assert float(((output - target) ** 2).mean()) == pytest.approx(checkpoint["training"]["best_validation_loss"],
                                                                   rel=1e-9)
    output, target = evaluate(checkpoint, lattice, data, test, reference)
    error = float(((target - output).abs() / target.abs()).mean())
    assert error == pytest.approx(checkpoint["training"]["test_rel_errors"][reference], rel=1e-9)
    assert checkpoint["training"]["best_validation_loss"] < 1e-9 and error < 5e-4


@pytest.mark.parametrize("path", [pytest.param(p, marks=[slow] if p.name.startswith("LionDog") else [])
                                  for p in FILES], ids=lambda p: p.stem)
def test_stored_test_errors_of_all_directions(path, training_data):
    """The relative test error of every direction a circuit serves (rotated inputs) recomputes."""
    checkpoint = torch.load(path, weights_only=True)
    lattice, data = training_data(checkpoint)
    _, test = split(checkpoint, len(data["u"]))
    for member in checkpoint["lattice"]["directions"]:
        output, target = evaluate(checkpoint, lattice, data, test, member)
        error = float(((target - output).abs() / target.abs()).mean())
        assert error == pytest.approx(checkpoint["training"]["test_rel_errors"][member], rel=1e-9), member
        assert error < 5e-4


# ---------------------------------------------------------------------------------------------- collision output

# (model, symmetry, conservation): (sum_k v_k d_k, |d|) of d = f*_quantum - f*_BGK on a fixed state, v = linspace(-1, 1)
FINGERPRINTS = {
    ("Corgi", False, False): (-0.00023090116866969729, 0.00011201268861089843),
    ("Corgi", False, True): (-0.00022997470286726263, 0.00010344084792126167),
    ("Corgi", True, False): (-0.00020844894642047405, 0.00010451800014711307),
    ("Corgi", True, True): (-0.00021165178624844418, 9.341311285496453e-05),
    ("LionDog", False, False): (-5.5519827933340994e-05, 9.53713891245259e-05),
    ("LionDog", False, True): (-4.9368724495024675e-05, 7.59319446807606e-05),
    ("LionDog", True, False): (-3.478403132350683e-05, 7.17835860041886e-05),
    ("LionDog", True, True): (-2.2423612826890788e-05, 3.7978480828184385e-05),
}
SHAPES = {"Corgi": (3, 2), "LionDog": (2, 1, 1)}


def near_equilibrium(lattice, shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    rho = 1 + 0.02 * (torch.rand(shape, generator=gen, dtype=torch.float64) - 0.5)
    u = 0.1 / lattice.d ** 0.5 * (2 * torch.rand((lattice.d,) + shape, generator=gen, dtype=torch.float64) - 1)
    noise = 1e-3 * torch.randn((lattice.q,) + shape, generator=gen, dtype=torch.float64)
    return equilibrium(rho, u, lattice) + noise * lattice.w.view((-1,) + (1,) * lattice.d)


@pytest.mark.parametrize("model, symmetry, conservation", list(FINGERPRINTS))
def test_collision_fingerprint(model, symmetry, conservation):
    lattice = LATTICES[model]
    f = near_equilibrium(lattice, SHAPES[model])
    rho, u = moments(f, lattice)
    quantum = QuantumCollision.from_folder(MODELS, model, lattice, 1.3, symmetry, conservation)
    d = quantum(f, rho, u) - BGK(lattice, 1.3)(f, rho, u)
    v = torch.linspace(-1, 1, d.numel(), dtype=torch.float64)
    weighted, norm = FINGERPRINTS[model, symmetry, conservation]
    assert float(d.flatten() @ v) == pytest.approx(weighted, rel=1e-9)
    assert float(d.norm()) == pytest.approx(norm, rel=1e-9)


# ---------------------------------------------------------------------------------------------- example cases

def velocity_difference(a):
    return common.relative_l2(torch.as_tensor(a["velocity_quantum"]), torch.as_tensor(a["velocity_bgk"]))


QUANTITIES = {
    "bgk error": lambda a: a["velocity_error_bgk"][-1],                   # against the exact solution
    "quantum error": lambda a: a["velocity_error_quantum"][-1],
    "velocity difference": velocity_difference,                          # relative L2, quantum vs BGK
    "mass change": lambda a: abs(a["mass_change_quantum"][-1]),
    "centreline difference": lambda a: common.relative_l2(torch.as_tensor(a["ux_centreline_quantum"]),
                                                          torch.as_tensor(a["ux_centreline_bgk"])),
    "kinetic energy difference": lambda a: abs(a["kinetic_energy_quantum"][-1] / a["kinetic_energy_bgk"][-1] - 1),
}
POST = {"plain": [], "CONS": ["--conservation"], "SYM": ["--symmetry"], "CONS-SYM": ["--conservation", "--symmetry"]}

# script, arguments, post-processing, recorded values
EXAMPLES = [
    ("taylor_green_2d", ["--n", "16"], "plain",
     {"bgk error": 2.2081e-2, "quantum error": 3.4936e-2, "velocity difference": 5.7702e-2}),
    ("taylor_green_2d", ["--n", "16"], "CONS",
     {"bgk error": 2.2081e-2, "quantum error": 2.4217e-2, "velocity difference": 6.8545e-3}),
    ("taylor_green_2d", ["--n", "16"], "SYM",
     {"bgk error": 2.2081e-2, "quantum error": 3.3941e-2, "velocity difference": 5.7080e-2}),
    ("taylor_green_2d", ["--n", "16"], "CONS-SYM",
     {"bgk error": 2.2081e-2, "quantum error": 2.3281e-2, "velocity difference": 1.2423e-3}),
    ("lid_driven_cavity", ["--n", "8", "--steps", "200"], "plain",
     {"mass change": 1.6308e-2, "velocity difference": 5.8782e-2, "centreline difference": 7.4081e-2}),
    ("lid_driven_cavity", ["--n", "8", "--steps", "200"], "CONS",
     {"mass change": 1e-13, "velocity difference": 4.8406e-3, "centreline difference": 3.3462e-3}),
    ("double_shear_layer", ["--steps", "10"], "CONS",
     {"velocity difference": 3.9587e-4, "kinetic energy difference": 1.5624e-6}),
    ("taylor_green_3d", ["--n", "6", "--steps", "10"], "plain",
     {"velocity difference": 3.9396e-2, "kinetic energy difference": 6.9455e-2}),
    ("taylor_green_3d", ["--n", "6", "--steps", "10"], "CONS",
     {"velocity difference": 1.6661e-2, "kinetic energy difference": 1.2471e-3}),
]


@pytest.mark.parametrize("script, arguments, post, recorded", EXAMPLES, ids=[f"{e[0]}-{e[2]}" for e in EXAMPLES])
def test_example_accuracy(tmp_path, script, arguments, post, recorded):
    case = __import__(script)
    with contextlib.redirect_stdout(io.StringIO()):
        arrays = common.run_example(case, [*arguments, *POST[post], "--output", str(tmp_path)])
    for quantity, value in recorded.items():
        measured = QUANTITIES[quantity](arrays)
        if quantity == "bgk error":                                   # classical reference: unchanged
            assert measured == pytest.approx(value, rel=1e-4), quantity
        else:                                                         # quantum collision: not worse
            assert measured <= 1.05 * value, f"{quantity}: {measured:.4e} > recorded {value:.4e}"


# ---------------------------------------------------------------------------------------------- training

SHORT_RUN_VALIDATION_LOSS = 0.031158768568890893


def test_short_training_run():
    """1 epoch on 400 samples with the Corgi architecture (axis orbit, seed 1)."""
    data = generate(D2Q9, n_samples=400)
    checkpoint = train_circuit(D2Q9, "axis", CircuitConfig("Parallel", "SEL", 1, 6, features=2), data,
                               TrainingConfig(epochs=1), seed=1, log=lambda message: None)
    assert checkpoint["training"]["best_validation_loss"] == pytest.approx(SHORT_RUN_VALIDATION_LOSS, rel=1e-6)



def equilibrium_error(checkpoint, lattice, data, rows):
    """mean (e_theta(u) - e(u))^2 of the reference direction: the error of the learned equilibrium (omega / 2 = 1)."""
    rows = torch.as_tensor(rows)
    i = checkpoint["lattice"]["reference_direction"]
    u = data["u"][rows]
    ones = torch.ones(len(rows), 1, dtype=torch.float64)
    e = equilibrium(ones.view(-1), u.T, lattice)[i]
    return float(((TrainedCircuit(checkpoint, lattice)(torch.cat([u, 0 * ones, ones], 1)) - e) ** 2).mean())


@slow
def test_retraining_reaches_the_shipped_accuracy(training_data):
    """Training the Corgi axis circuit with its stored seed and settings gives an equivalent circuit: the error of the
    learned equilibrium within 5 % of the shipped circuit, weights within 1e-2 (measured: 0.3 %, 2.9e-3)."""
    shipped = torch.load(MODELS / "Corgi_Parallel_SEL1_L6_axis.pth", weights_only=True)
    lattice, data = training_data(shipped)
    stored = shipped["training"]
    training = TrainingConfig(**{k: stored[k] for k in ("epochs", "batch_size", "learning_rate", "lr_decay", "lr_step",
                                                        "split_seed")}, split=tuple(stored["split"]))
    retrained = train_circuit(lattice, "axis", CircuitConfig.from_checkpoint(shipped), data, training,
                              seed=stored["seed"], log=lambda message: None)
    validation, _ = split(shipped, len(data["u"]))
    assert equilibrium_error(retrained, lattice, data, validation) == pytest.approx(
        equilibrium_error(shipped, lattice, data, validation), rel=0.05)
    assert torch.allclose(retrained["circuit_weights"], shipped["circuit_weights"], atol=1e-2, rtol=0)
    assert torch.allclose(retrained["scaling_weights"], shipped["scaling_weights"], atol=1e-2, rtol=0)
