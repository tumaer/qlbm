"""Train new collision circuits (one per orbit: rest, axis, diag) on synthetic BGK data.

Defaults: the architecture and training data of the 2D example circuits (Corgi: D2Q9, Parallel, SEL x 1, L = 6,
100 000 samples, |u| <= 0.1, 9 epochs), with the fixed relaxation rate omega = 1 of hqlbm.data. The 3D example
circuits (LionDog) use
    --lattice D3Q19 --ansatz SuperParallel --blocks 2 --processing BEL --entangle-layers 3
--seed applies to every orbit of the run, and a run is repeated exactly with the same seed. The seed of a circuit is
stored in its checkpoint; to train with it, train that orbit alone (--orbits axis --seed <seed>). New circuits are
written to examples/output/models/ as <tag>_<circuit>_<orbit>.pth, in the format of the files in models/, and are
used with QuantumCollision.from_folder(<folder>, <tag>, ...).

    python examples/train.py [--tag MyCorgi] [--orbits rest axis diag] [--epochs 9] [--seed S]
"""

import argparse
from pathlib import Path

from hqlbm import LATTICES, CircuitConfig
from hqlbm.data import generate
from hqlbm.train import TrainingConfig, save_checkpoint, train_circuit


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lattice", choices=sorted(LATTICES), default="D2Q9")
    p.add_argument("--tag", default="MyCorgi", help="name of the new circuits")
    p.add_argument("--orbits", nargs="+", choices=["rest", "axis", "diag"], default=["rest", "axis", "diag"])
    p.add_argument("--ansatz", choices=["Line", "Line+", "Parallel", "SuperParallel"], default="Parallel")
    p.add_argument("--processing", choices=["SEL", "BEL"], default="SEL")
    p.add_argument("--entangle-layers", type=int, default=1)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--blocks", type=int, default=2, help="input copies P (SuperParallel)")
    p.add_argument("--samples", type=int, default=100_000)
    p.add_argument("--epochs", type=int, default=9)
    p.add_argument("--seed", type=int, default=None, help="initialisation / shuffling seed (default: random)")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output" / "models")
    args = p.parse_args()

    lattice = LATTICES[args.lattice]
    circuit = CircuitConfig(args.ansatz, args.processing, args.entangle_layers, args.layers, args.blocks, lattice.d)
    print(f"{lattice.name}: {circuit.label()}, {args.samples} samples")
    data = generate(lattice, n_samples=args.samples)
    for orbit in args.orbits:
        checkpoint = train_circuit(lattice, orbit, circuit, data, TrainingConfig(epochs=args.epochs), seed=args.seed)
        path = save_checkpoint(checkpoint, args.out, args.tag)
        print(f"[{orbit}] seed {checkpoint['training']['seed']}, best validation MSE "
              f"{checkpoint['training']['best_validation_loss']:.3e} -> {path}")


if __name__ == "__main__":
    main()
