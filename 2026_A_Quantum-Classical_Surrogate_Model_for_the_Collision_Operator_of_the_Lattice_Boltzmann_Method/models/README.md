# Pre-trained example circuits

The files in this folder are **already trained** circuit parameters, provided as ready-to-use examples. They let you
run the example cases without training first. They are the two models used for the 2D and 3D cases in the
publication. To train your own circuits, use `examples/train.py`; it writes to `examples/output/models/` and never
overwrites these files.

Each model is a set of three circuits, one per velocity orbit (orbit compression). The circuit of an orbit is trained
on the orbit's representative direction and serves all members with rotated inputs.

| Model (tag) | Lattice | Circuit | Qubits per circuit | Parameters per circuit | Used in |
|---|---|---|---|---|---|
| `Corgi` | D2Q9 | Parallel, SEL x 1, L = 6 | 4 (2 re-uploading + pre-collision + ancilla) | 48 (42 + 6 scaling) | `taylor_green_2d.py`, `double_shear_layer.py`, `lid_driven_cavity.py` |
| `LionDog` | D3Q19 | SuperParallel (P = 2), BEL x 3, L = 6 | 8 (6 re-uploading + pre-collision + ancilla) | 132 (126 + 6 scaling) | `taylor_green_3d.py` |

| File | Directions | Validation MSE | Relative test error (mean per direction, max. over directions) |
|---|---|---|---|
| `Corgi_Parallel_SEL1_L6_rest.pth` | 0 | 3.4e-10 | 2.8e-5 |
| `Corgi_Parallel_SEL1_L6_axis.pth` | 1-4 | 7.1e-11 | 5.8e-5 |
| `Corgi_Parallel_SEL1_L6_diag.pth` | 5-8 | 5.7e-11 | 2.4e-4 |
| `LionDog_SuperParallelP2_BEL3_L6_rest.pth` | 0 | 1.7e-10 | 2.6e-5 |
| `LionDog_SuperParallelP2_BEL3_L6_axis.pth` | 1-6 | 4.6e-11 | 1.2e-4 |
| `LionDog_SuperParallelP2_BEL3_L6_diag.pth` | 7-18 | 1.0e-10 | 2.7e-4 |

## Training data

Both models were trained on 100 000 synthetic samples (seed 42) with these ranges:

- density: rho in [0.98, 1.02]
- velocity: |u| <= 0.1 (lattice units)
- non-equilibrium amplitude: sigma = 1e-3
- relaxation rate: omega = 1

The training used the 80/10/10 split, Adam (learning rate 1e-3, decayed by a factor of 0.1 every 3 epochs), batch
size 32 and 9 epochs. Use the models for lattice velocities up to about 0.1 and any omega in (0, 2]: the circuit
encodes the relaxation exactly, and the trained part represents the equilibrium e(u) = f^eq / rho only. The residual
of the circuit output is omega / 2 (e_theta(u) - e(u)), so the relaxation rate of the training data only weights the
samples in the loss, and one fixed value suffices.

The validation MSE and the test errors are those of the stored weights on this data (`generate(lattice)` with the
default arguments).

## File names and contents

The files are named `<tag>_<ansatz>[P<blocks>]_<processing><L_W>_L<L>_<target>.pth`, where the target is the orbit
(`rest`, `axis`, `diag`) or, for per-direction models, the direction (`d<d>q<i>`). `examples/train.py` writes new
circuits with the same names and in the same format.

Each file is a `torch.save` dictionary:

| Entry | Contents |
|---|---|
| `circuit_weights`, `scaling_weights` | the trained parameters |
| `circuit` | `ansatz`, `processing`, `entangle_layers`, `layers`, `blocks` (the `CircuitConfig`), `qubits`, `parameters`, `backend` |
| `lattice` | `name` (D2Q9 / D3Q19), `orbit_compression`, `target`, `reference_direction` (the direction trained on), `directions` (the directions served) |
| `dataset` | the training data: `lattice`, `num_samples`, `u_max`, `rho_range`, `sigma`, `seed` (arguments of `hqlbm.data.generate`) and `omega` (fixed relaxation rate, 1) |
| `training` | `epochs`, `batch_size`, `learning_rate`, `lr_decay`, `lr_step`, `split`, `split_seed`, `seed` (initialisation and shuffling), `best_validation_loss`, `test_rel_errors` (mean relative test error per direction) |

Load the files with `hqlbm.load_circuits(folder, tag, lattice)` or `QuantumCollision.from_folder(...)`.

## Reproducing the training

Every circuit has its own seed, stored in `checkpoint["training"]["seed"]`. Training its orbit alone with that
seed uses the same samples, initialisation and batch order, e.g. for `Corgi_Parallel_SEL1_L6_axis.pth`:

```bash
python examples/train.py --tag Corgi --orbits axis --seed 772397179
```

and for the LionDog circuits additionally `--lattice D3Q19 --ansatz SuperParallel --blocks 2 --processing BEL
--entangle-layers 3`. This gives a circuit of the same accuracy (checked for the Corgi axis circuit by the slow
regression test).
