"""Shared driver of the example cases: command line, both collision operators, time loop, comparison, output file.

Every case script defines the same names, in this order:

    CASE, LATTICE, MODEL               case name (output files), lattice, tag of the trained circuits in models/
    arguments(p)                       adds the case options to the argparse parser p
    setup(args) -> Setup               grid, relaxation rate, number of steps, walls and case parameters
    initial(s) -> f                    initial populations (q, *grid)
    observe(step, f, s) -> dict        diagnostics of a state: floats (stored as time series) and tensors (fields,
                                       stored at the last observation)

and ends with run_example(sys.modules[__name__]). run_example runs the case with the classical BGK reference and
the hybrid quantum collision on the same setup, prints the final diagnostics of both and their difference, and
writes one .npz file to --output (default examples/output/):

    <case>_<model>_<grid>_<flow>_<post-processing>.npz

    case             taylor_green_2d, double_shear_layer, lid_driven_cavity, taylor_green_3d
    model            tag of the trained circuits, e.g. Corgi, LionDog
    grid             cells per direction, e.g. 32x32, 200x100, 20x20x20
    flow             the flow parameter of the case, e.g. Re100 or omega1.3
    post-processing  plain (circuit output only), CONS, SYM or CONS-SYM

e.g. taylor_green_2d_Corgi_32x32_Re20_plain.npz. Inside, every diagnostic of a run is stored as <name>_bgk
(classical reference) and <name>_quantum (hybrid collision), e.g. step_bgk, velocity_quantum, kinetic_energy_bgk;
the setup values (omega, steps and the case parameters) have no suffix. Trained circuits follow
<tag>_<circuit>_<target>.pth (hqlbm.train.checkpoint_name).
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from hqlbm import BGK, QuantumCollision, Streaming, run

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUTPUT = Path(__file__).resolve().parent / "output"
RUNS = ("bgk", "quantum")


@dataclass
class Setup:
    """Parameters of one run of a case."""
    shape: tuple                 # cells per direction
    omega: float                 # relaxation rate
    steps: int                   # number of time steps
    flow: str                    # flow parameter in the file name, e.g. Re20 or omega1.3
    every: int = 0               # observe at step 0 and every `every` steps (0: final state only)
    walls: dict = None           # wall faces and velocities (hqlbm.Streaming); None: periodic
    parameters: dict = field(default_factory=dict)   # case parameters (velocity scale, viscosity, ...)


def parser(description: str) -> argparse.ArgumentParser:
    """Common options. The quantum collision is the plain circuit output unless a post-processing step is chosen."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symmetry", action="store_true",
                   help="post-processing: symmetrisation over the hyperoctahedral group (SYM)")
    p.add_argument("--conservation", action="store_true",
                   help="post-processing: mass and momentum conservation correction (CONS)")
    p.add_argument("--models", type=Path, default=MODELS, help="folder of the trained circuits")
    p.add_argument("--output", type=Path, default=OUTPUT, help="folder of the result files")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    return p


def collisions(lattice, tag: str, omega: float, args) -> dict:
    """{"bgk": classical reference, "quantum": hybrid collision with the trained circuits <tag>}."""
    quantum = QuantumCollision.from_folder(args.models, tag, lattice, omega, symmetry=args.symmetry,
                                           conservation=args.conservation)
    print(quantum)
    return {"bgk": BGK(lattice, omega), "quantum": quantum}


def relative_l2(a, b) -> float:
    return float(torch.sqrt(((a - b) ** 2).sum() / (b ** 2).sum()))


def output_name(case: str, tag: str, shape, flow: str, args) -> str:
    """<case>_<model>_<grid>_<flow>_<post-processing> (see the module docstring)."""
    post = "-".join(name for name, on in (("CONS", args.conservation), ("SYM", args.symmetry)) if on) or "plain"
    return f"{case}_{tag}_{'x'.join(str(n) for n in shape)}_{flow}_{post}"


def collect(records: list):
    """Observations of one run -> (series, fields): floats as time series, tensors at the last observation."""
    last = records[-1]
    series = {k: np.asarray([r[k] for r in records]) for k, v in last.items() if not torch.is_tensor(v)}
    fields = {k: v.numpy() for k, v in last.items() if torch.is_tensor(v)}
    return series, fields


def save(args, name: str, **arrays) -> Path:
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{name}.npz"
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print(f"results -> {path}")
    return path


def run_example(case, argv=None) -> dict:
    """Run a case module (see the module docstring) with both collisions, print the comparison and save the results.
    argv: command line arguments (default sys.argv). Returns the saved arrays."""
    p = parser(case.__doc__)
    case.arguments(p)
    args = p.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    s = case.setup(args)
    values = ", ".join(f"{k} = {v:.4g}" for k, v in {"omega": s.omega, **s.parameters}.items())
    print(f"{case.CASE} ({case.LATTICE.name}, {'x'.join(map(str, s.shape))}, {s.flow}): {values}, {s.steps} steps")
    streaming = Streaming(case.LATTICE, s.shape, s.walls)

    series, fields = {}, {}
    for name, collision in collisions(case.LATTICE, case.MODEL, s.omega, args).items():
        tic = time.time()
        _, records, completed = run(case.initial(s), collision, streaming, s.steps, s.every,
                                    lambda n, f: {"step": n, **case.observe(n, f, s)})
        if completed < s.steps:
            print(f"  non-finite populations after step {completed} - stopped")
        series[name], fields[name] = collect(records)
        final = ", ".join(f"{k} {v[-1]:.6g}" for k, v in series[name].items() if k != "step")
        print(f"{name:8s} step {series[name]['step'][-1]}: {final}   ({time.time() - tic:.1f} s)")

    (sb, sq), (fb, fq) = (series[n] for n in RUNS), (fields[n] for n in RUNS)
    differences = [f"{k} rel. L2 {relative_l2(torch.as_tensor(fq[k]), torch.as_tensor(v)):.3e}" for k, v in fb.items()]
    differences += [f"{k} rel. diff. {abs(sq[k][-1] - v[-1]) / abs(v[-1]):.2e}" for k, v in sb.items()
                    if k != "step" and abs(v[-1]) > 1e-10]       # skips values that vanish for BGK
    print("quantum vs BGK: " + ", ".join(differences))

    arrays = {f"{k}_{name}": v for name in RUNS for k, v in {**series[name], **fields[name]}.items()}
    arrays.update(omega=s.omega, steps=s.steps, **s.parameters)
    save(args, output_name(case.CASE, case.MODEL, s.shape, s.flow, args), **arrays)
    return arrays
