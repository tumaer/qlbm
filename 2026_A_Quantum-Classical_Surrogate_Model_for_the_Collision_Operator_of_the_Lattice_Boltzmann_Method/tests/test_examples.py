"""Every example script follows the structure of examples/common.py, runs (in miniature) from the command line and
writes its results under the naming convention of examples/common.py."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = ["taylor_green_2d", "double_shear_layer", "lid_driven_cavity", "taylor_green_3d"]
FUNCTIONS = ["arguments", "setup", "initial", "observe"]          # after CASE, LATTICE, MODEL = ...
SETUP = {"omega", "steps"}                    # arrays without the _bgk / _quantum suffix (besides case parameters)

CASES = [
    ("taylor_green_2d.py", ["--n", "8"], "taylor_green_2d_Corgi_8x8_Re20_plain.npz"),
    ("taylor_green_2d.py", ["--n", "8", "--conservation"], "taylor_green_2d_Corgi_8x8_Re20_CONS.npz"),
    ("taylor_green_2d.py", ["--n", "8", "--symmetry"], "taylor_green_2d_Corgi_8x8_Re20_SYM.npz"),
    ("taylor_green_2d.py", ["--n", "8", "--conservation", "--symmetry"], "taylor_green_2d_Corgi_8x8_Re20_CONS-SYM.npz"),
    ("double_shear_layer.py", ["--steps", "2"], "double_shear_layer_Corgi_200x100_Re100_plain.npz"),
    ("lid_driven_cavity.py", ["--n", "8", "--steps", "5"], "lid_driven_cavity_Corgi_8x8_Re100_plain.npz"),
    ("taylor_green_3d.py", ["--n", "4", "--steps", "2"], "taylor_green_3d_LionDog_4x4x4_omega1.3_plain.npz"),
]


def run_script(script, arguments):
    result = subprocess.run([sys.executable, "-W", "ignore", str(EXAMPLES / script), *arguments],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("script, arguments, expected", CASES, ids=[c[2] for c in CASES])
def test_example(tmp_path, script, arguments, expected):
    run_script(script, [*arguments, "--output", str(tmp_path)])
    assert [p.name for p in tmp_path.iterdir()] == [expected]
    with np.load(tmp_path / expected) as results:
        keys = set(results.files)
        runs = {k.rsplit("_", 1)[0] for k in keys if k.endswith(("_bgk", "_quantum"))}
        assert {"step", "velocity"} <= runs and SETUP <= keys
        assert all(f"{k}_bgk" in keys and f"{k}_quantum" in keys for k in runs)          # both runs, same arrays
        assert all(np.isfinite(results[key]).all() for key in keys)


@pytest.mark.parametrize("script", SCRIPTS)
def test_structure(script):
    """Every case defines the names of the common interface, in the same order, and runs through run_example."""
    source = (EXAMPLES / f"{script}.py").read_text()
    positions = [source.find(f'CASE, LATTICE, MODEL = "{script}"')]
    positions += [source.find(f"\ndef {name}(") for name in FUNCTIONS]
    assert -1 not in positions and positions == sorted(positions), dict(zip(["CASE", *FUNCTIONS], positions))
    assert source.rstrip().endswith('if __name__ == "__main__":\n    run_example(sys.modules[__name__])')


def test_train_example(tmp_path):
    run_script("train.py", ["--orbits", "rest", "--samples", "64", "--epochs", "1", "--layers", "1", "--seed", "1",
                            "--out", str(tmp_path)])
    assert [p.name for p in tmp_path.iterdir()] == ["MyCorgi_Parallel_SEL1_L1_rest.pth"]


@pytest.mark.parametrize("n", [8, 9])
def test_cavity_centrelines(n):
    """The centreline profiles lie on x = N/2 and y = N/2 for even and odd N: with u_x linear in x and u_y linear in
    y, both vanishing on the centrelines, the profiles are zero."""
    sys.path.insert(0, str(EXAMPLES))
    import lid_driven_cavity as case

    s = case.setup(argparse.Namespace(re=100.0, n=n, steps=1))
    x = (torch.arange(n, dtype=torch.float64) + 0.5) / n - 0.5
    X, Y = torch.meshgrid(x, x, indexing="ij")
    f = case.equilibrium(torch.ones(n, n, dtype=torch.float64), 0.01 * torch.stack([X, Y]), case.LATTICE)
    observed = case.observe(0, f, s)
    assert observed["ux_centreline"].shape == (n,) and observed["uy_centreline"].shape == (n,)
    assert observed["ux_centreline"].abs().max() < 1e-15 and observed["uy_centreline"].abs().max() < 1e-15
