# hqlbm: hybrid quantum-classical lattice Boltzmann method

`hqlbm` is the code accompanying the [publication](https://doi.org/10.1016/j.cpc.2026.110513) on a variational
quantum circuit as the collision operator of the lattice Boltzmann method (LBM). A parametrised quantum circuit,
implemented in PennyLane and simulated with its state-vector simulator `default.qubit`, computes the BGK collision
of every node and lattice direction. Streaming and boundary conditions stay classical. The package contains:

- the collision circuit (`hqlbm.circuit`), configurable as described in [CIRCUIT.md](CIRCUIT.md);
- the hybrid collision operator with orbit compression and two optional post-processing steps, an exact symmetry
  constraint and exact mass and momentum conservation (`hqlbm.collision`);
- a minimal D2Q9 / D3Q19 LBM solver with the classical BGK collision as reference, periodic boundaries and moving
  walls (`hqlbm.lbm`);
- data generation and training of new circuits (`hqlbm.data`, `hqlbm.train`);
- pre-trained example circuits for 2D and 3D (see [models/README.md](models/README.md));
- four flow cases: 2D and 3D Taylor-Green vortex, double shear layer and lid-driven cavity.

## Installation

```bash
pip install -e .            # torch, numpy, pennylane
pip install -e .[test]      # + pytest
```

Python >= 3.14, torch >= 2.14, PennyLane >= 0.45.1, NumPy >= 2.5 (the versions the package is tested with).
Everything runs on the CPU in double precision.

## Quick start

The collision circuit of one orbit, with its trained weights:

```python
import torch, pennylane as qml
from hqlbm import CircuitConfig, collision_circuit

checkpoint = torch.load("models/Corgi_Parallel_SEL1_L6_axis.pth", weights_only=True)
config = CircuitConfig.from_checkpoint(checkpoint)          # Parallel, SEL x 1, L = 6, 2D: 4 qubits
circuit = collision_circuit(config)                         # PennyLane QNode on default.qubit

x = torch.tensor([[0.05, -0.02, 0.11, 0.65]], dtype=torch.float64)   # [u_x, u_y, f_i / rho, omega / 2]
print(qml.draw(circuit, decimals=None)(checkpoint["circuit_weights"], checkpoint["scaling_weights"], x))
z = circuit(checkpoint["circuit_weights"], checkpoint["scaling_weights"], x)   # f*_i = 2 rho z - f_i
```

A simulation with the hybrid collision and the classical reference:

```python
from hqlbm import D2Q9, BGK, QuantumCollision, Streaming, equilibrium, run, step

n, omega = 32, 1.4
u0 = 0.05 * (2 * torch.rand(2, n, n, dtype=torch.float64) - 1)        # |u| <= 0.07, inside the training range
f = equilibrium(torch.ones(n, n, dtype=torch.float64), u0, D2Q9)

quantum = QuantumCollision.from_folder("models", "Corgi", D2Q9, omega)   # pre-trained 2D example circuits,
                                                                        # plain circuit output (no post-processing)
classical = BGK(D2Q9, omega)                                            # reference
streaming = Streaming(D2Q9, (n, n))                                     # periodic box

for _ in range(100):
    f = step(f, quantum, streaming)

# or: f, observations, completed = run(f, quantum, streaming, steps=100, every=10, observe=my_function)
```

## Method

**Circuit.** For node `x` and direction `i`, the circuit takes the input `[u, f_i/rho, omega/2]` and has three parts:

- A data re-uploading register (`n` qubits) encodes `u` with R_Y rotations and trainable scales in `L` layers,
  interleaved with trainable entangling blocks (StronglyEntanglingLayers `SEL` or BasicEntanglerLayers `BEL`). Its
  output qubit gives `e_i(u) = <Z>`, the learned equilibrium `f_i^eq / rho`.
- A pre-collision qubit encodes `f_i / rho` with `R_Y(arccos(f_i/rho))`.
- An ancilla encodes `p = omega/2` with `R_Y(2 arcsin(sqrt(p)))`.

A CSWAP controlled by the ancilla mixes the output qubit and the pre-collision qubit, so that

```
<Z_pre> = (1 - omega/2) f_i/rho + (omega/2) e_i(u),        f*_i = 2 rho <Z_pre> - f_i = f_i + omega (rho e_i(u) - f_i),
```

which is the BGK collision with the learned equilibrium. Only the re-uploading register is trained, and it depends
on `u` only. One model therefore covers all relaxation rates in its training range.

The ansaetze differ in how the input is encoded:

| Ansatz | Re-uploading qubits | Encoding |
|---|---|---|
| `Line` / `Line+` | 1 | features one after another (`Line+`: R_X, R_Y, R_Z) |
| `Parallel` | d | one feature per qubit |
| `SuperParallel` | P·d | the input repeated P times |

**Orbit compression.** The lattice velocities split into orbits of equal length: rest, axis and diagonal. One circuit
per orbit is trained on the representative direction `c_ref`. Member `i` reuses it with the rotated input `R_i^T u`,
where `R_i c_ref = c_i`. This needs 3 circuits instead of 9 (D2Q9) or 19 (D3Q19).

**Post-processing (optional, off by default).** By default the post-collision populations are the plain circuit
output, `f*_i = 2 rho <Z_pre> - f_i`. Two post-processing steps can be switched on with
`QuantumCollision(..., symmetry=True, conservation=True)`; both are implemented in
`src/postprocessing.py`:

- **Symmetry (SYM).** The circuits are evaluated for every element `g` of the hyperoctahedral group (8 elements in
  2D, 48 in 3D) on the transformed state. The results are transformed back and averaged, which makes the collision
  exactly equivariant under lattice rotations and reflections.
- **Conservation (CONS).** The circuit output is corrected by the orthogonal projection onto the populations that
  carry neither mass nor momentum: `z' = P z + (I - P) f/rho` with `P = I - C^T (C C^T)^-1 C` and `C = [1; c]`.
  This is the smallest correction, in the least-squares sense, that conserves mass and momentum exactly.

## Modifying the circuit

How to change the ansatz, the entangling blocks and the number of layers, and how to train and use a modified
circuit, is described in [CIRCUIT.md](CIRCUIT.md).

## Examples

All cases run the quantum collision and the classical BGK reference on the same setup and report the difference.
The quantum collision uses the plain circuit output; add `--symmetry` and/or `--conservation` for the
post-processing. Results are written as `.npz` files to `examples/output/` (or `--output`).

Every node needs `q` circuit evaluations per time step, and `|G| q` with SYM (72 for D2Q9, 912 for D3Q19); each
evaluation is a state-vector simulation of the full circuit.

| Script | Case (defaults) | Circuits |
|---|---|---|
| `examples/taylor_green_2d.py` | 2D Taylor-Green vortex against the exact solution (32^2, Re = 20) | Corgi |
| `examples/double_shear_layer.py` | double shear layer (200 x 100, Re = 100, 1000 steps) | Corgi |
| `examples/lid_driven_cavity.py` | lid-driven cavity (32^2, Re = 100, 5000 steps) | Corgi |
| `examples/taylor_green_3d.py` | 3D Taylor-Green vortex (20^3, omega = 1.3, 40 steps) | LionDog |
| `examples/train.py` | training of new circuits (default: the Corgi architecture) | - |

Install the package first (`pip install -e .`), then run the examples from the package folder, for example
`python examples/taylor_green_3d.py --omega 0.7`. `--help` lists the options of a script.

All case scripts have the same structure and are run by one driver, `run_example` in `examples/common.py`:

| Name | Purpose |
|---|---|
| `CASE, LATTICE, MODEL` | case name (output file), lattice, tag of the trained circuits in `models/` |
| `arguments(p)` | case options of the command line |
| `setup(args) -> Setup` | grid, relaxation rate, number of steps, observation interval, walls, case parameters |
| `initial(s) -> f` | initial populations |
| `observe(step, f, s) -> dict` | diagnostics of a state: numbers (stored as time series) and fields (stored at the end) |

`run_example` runs the case with BGK and with the quantum collision, prints the final diagnostics of both and their
difference, and saves the results. A new case is a new script with these five definitions.

## Output files

All files the package writes follow one scheme: what it is, then the configuration, then the variant.

| Files | Name | Example |
|---|---|---|
| trained circuits (`.pth`) | `<tag>_<ansatz>[P<blocks>]_<processing><L_W>_L<L>_<target>.pth`, target = orbit or `d<d>q<i>` | `Corgi_Parallel_SEL1_L6_axis.pth` |
| example results (`.npz`) | `<case>_<model>_<grid>_<flow>_<post-processing>.npz`, post-processing = `plain`, `CONS`, `SYM` or `CONS-SYM` | `taylor_green_2d_Corgi_32x32_Re20_plain.npz` |

The circuits in `models/` have exactly the name and format that `examples/train.py` produces (format in
[models/README.md](models/README.md)). In the result files, the arrays of the two runs end in `_bgk` (classical
reference) and `_quantum` (hybrid collision), e.g. `step_bgk`, `velocity_quantum`, `kinetic_energy_quantum`; the
setup values (`omega`, `steps` and the case parameters) have no suffix.

## Tests

```bash
pytest              # all tests except the slow tier
pytest -m slow      # slow tier: test errors of all LionDog directions, retraining of a shipped circuit
```

The tests check:

- the weighted-sum structure of the circuit (`p = 0` returns `f_i/rho`, `p = 1` returns `e_i(u)`, linear in
  between) for 32 architectures, and the gradients;
- every circuit configuration through training, saving, loading and the collision with all post-processing variants;
- exact mass and momentum conservation with the CONS post-processing;
- exact equivariance under the group elements with the SYM post-processing;
- the lattices, streaming, walls and training data;
- the structure, command line and output files of the example scripts.

Regression tests (`tests/test_regression.py`) keep the accuracy of the shipped circuits and the examples:

- the validation loss and test errors stored in `models/*.pth` recompute from regenerated training data;
- the collision output of both models in all four post-processing variants matches recorded values;
- short runs of the example cases stay within 5 % of the recorded errors against the exact solution and differences
  to BGK;
- a short training run reproduces its recorded validation loss, and (slow) retraining the Corgi axis circuit with
  its stored seed reaches the accuracy of the shipped circuit.

If a change of the code alters these values on purpose, record the new values in `tests/test_regression.py`.

## Package layout

```
src/                    the package hqlbm (import hqlbm)
    lattice.py          D2Q9 / D3Q19 velocities, weights, orbits, hyperoctahedral group
    lbm.py              moments, equilibrium, BGK reference, streaming (periodic, moving walls), time loop
    circuit.py          CircuitConfig and the collision circuit (PennyLane QNode)
    collision.py        QuantumCollision (orbit compression), loading of trained circuits
    postprocessing.py   optional post-processing: symmetry (SYM) and conservation (CONS)
    data.py             synthetic training data
    train.py            training of one circuit
models/                 pre-trained example circuits (2D: Corgi, 3D: LionDog)
examples/               flow cases and training script
tests/                  verification
```

## License

This code is licensed under the GNU General Public License v3.0, see [LICENSE](../LICENSE) in the root of the
repository.

## Citation

If you use this code, please cite the accompanying publication in Computer Physics Communications:
doi:[10.1016/j.cpc.2026.110513](https://doi.org/10.1016/j.cpc.2026.110513).
