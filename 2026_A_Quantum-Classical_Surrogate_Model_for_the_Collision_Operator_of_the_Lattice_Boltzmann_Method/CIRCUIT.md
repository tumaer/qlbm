# Modifying the collision circuit

This guide explains which parts of the collision circuit can be changed, how to train a modified circuit, and how
to use it in a simulation. The circuit is defined in `src/circuit.py` as a PennyLane QNode and simulated with
the state-vector simulator `default.qubit`.

## Structure of the circuit

For one lattice node and one direction `i`, the circuit takes the input `x = [u, f_i/rho, omega/2]` and acts on
`n + 2` qubits:

```
re-uploading register (n qubits):   W(θ₀) ─ S(u·s₁) ─ W(θ₁) ─ S(u·s₂) ─ W(θ₂) ─ … ─ S(u·s_L) ─ W(θ_L) ─┐
                                                                                                   CSWAP ── 
pre-collision qubit:                R_Y(arccos(f_i/rho)) ──────────────────────────────────────────┤      <Z>
ancilla:                            R_Y(2 arcsin(sqrt(omega/2))) ──────────────────────────────────●
```

- **Encoding `S(u·s_l)`** writes the velocity into the register as angle rotations (R_Y; R_X, R_Y, R_Z for
  `Line+`). The angles are `u_k · s_l`, with one trainable scale `s_l` per re-uploading layer.
- **Processing `W(θ)`** is a block of trainable entangling layers. Every block has its own weights.
- **Readout.** The CSWAP, controlled by the ancilla, mixes the output qubit of the register with the pre-collision
  qubit. The measured value is `<Z_pre> = (1 - omega/2) f_i/rho + (omega/2) e_i(u)`. The post-collision population
  is `f*_i = 2 rho <Z_pre> - f_i`, which is the BGK collision with the learned equilibrium `e_i(u)`.

The re-uploading register is configurable. The pre-collision qubit, the ancilla and the CSWAP are fixed, because
together they implement the BGK update.

## Configuration options

The architecture is described by `hqlbm.CircuitConfig`:

| Field | `examples/train.py` flag | Options | Meaning |
|---|---|---|---|
| `ansatz` | `--ansatz` | `Line`, `Line+`, `Parallel`, `SuperParallel` | how the velocity is encoded (see below) |
| `blocks` | `--blocks` | integer >= 1 | copies P of the input; used by `SuperParallel` only (set to 1 otherwise) |
| `processing` | `--processing` | `SEL`, `BEL` | type of the trainable blocks W |
| `entangle_layers` | `--entangle-layers` | integer >= 1 | L_W, sub-layers inside every block W |
| `layers` | `--layers` | integer >= 1 | L, number of re-uploading layers (how often u is encoded) |
| `features` | set by `--lattice` | 2 (D2Q9), 3 (D3Q19) | d, number of velocity components |

All options can be combined freely.

### Ansatz: how the velocity is encoded

| Ansatz | Register qubits n | Encoding per layer | Number of blocks W |
|---|---|---|---|
| `Line` | 1 | the d components one after another on the same qubit, each followed by its own block W | 1 + L·d |
| `Line+` | 1 | as `Line`, with R_X, R_Y, R_Z for the components 1, 2, 3 (non-commuting axes) | 1 + L·d |
| `Parallel` | d | component k on qubit k, all at once, then one block W | 1 + L |
| `SuperParallel` | P·d | the input repeated P times (qubit k gets component k mod d), then one block W | 1 + L |

`Parallel` and `SuperParallel` encode all components at once on several qubits; `Line` and `Line+` use a single
register qubit and encode the components one after another.

### Processing: the trainable blocks

Each block W consists of `entangle_layers` (L_W) sub-layers:

| Processing | One sub-layer | Weights per block |
|---|---|---|
| `SEL` (`qml.StronglyEntanglingLayers`) | a general rotation `Rot(φ, θ, ω)` on every qubit, then CNOTs from qubit k to qubit k + r; the range r changes from sub-layer to sub-layer | (L_W, n, 3) |
| `BEL` (`qml.BasicEntanglerLayers`) | an R_X rotation on every qubit, then a ring of CNOTs | (L_W, n) |

`SEL` has three parameters per qubit and sub-layer, `BEL` one. For a single register qubit (`Line`, `Line+`) there
are no CNOTs, so a block consists of rotations only.

### Qubits and parameters

- qubits: `n + 2`
- circuit weights: `(number of blocks) × L_W × n × 3` for `SEL`, without the factor 3 for `BEL`
- scaling weights: `L`

`CircuitConfig` reports them as `config.label()`, `config.circuit_weight_shape` and `config.scaling_weight_shape`.

The table lists some configurations with L = 6. Gate counts are from `qml.specs`; each SEL or BEL sub-layer and
each encoding layer counts as several gates.

| Configuration | Lattice | Qubits | Parameters | Gates |
|---|---|---|---|---|
| `Line`, SEL, L_W = 1 | D2Q9 | 3 | 39 + 6 = 45 | 28 |
| `Line+`, SEL, L_W = 1 | D3Q19 | 3 | 57 + 6 = 63 | 40 |
| `Parallel`, SEL, L_W = 1 (Corgi) | D2Q9 | 4 | 42 + 6 = 48 | 43 |
| `Parallel`, SEL, L_W = 1 | D3Q19 | 5 | 63 + 6 = 69 | 63 |
| `Parallel`, BEL, L_W = 3 | D3Q19 | 5 | 63 + 6 = 69 | 147 |
| `SuperParallel`, P = 2, SEL, L_W = 1 | D3Q19 | 8 | 126 + 6 = 132 | 123 |
| `SuperParallel`, P = 2, BEL, L_W = 3 (LionDog) | D3Q19 | 8 | 126 + 6 = 132 | 291 |

The state vector simulated for one evaluation has `2^(n+2)` amplitudes.

## Inspecting a circuit

```python
import torch, pennylane as qml
from hqlbm import CircuitConfig, collision_circuit

config = CircuitConfig(ansatz="SuperParallel", blocks=2, processing="BEL", entangle_layers=3, layers=6, features=3)
circuit = collision_circuit(config)
print(config.label(), config.circuit_weight_shape, config.scaling_weight_shape)

weights = torch.zeros(config.circuit_weight_shape, dtype=torch.float64)
scales = torch.ones(config.scaling_weight_shape, dtype=torch.float64)
x = torch.tensor([[0.05, -0.02, 0.01, 0.06, 0.65]], dtype=torch.float64)    # [u_x, u_y, u_z, f_i/rho, omega/2]

print(qml.draw(circuit, decimals=None)(weights, scales, x))                  # circuit diagram
print(qml.specs(circuit, level="device")(weights, scales, x))                # gate counts and depth
```

On Windows, set `PYTHONIOENCODING=utf-8` if the diagram fails to print to a redirected output.

## Training a modified circuit

From the command line, one circuit per orbit (rest, axis, diag):

```bash
# a 3D Parallel circuit with BEL blocks and 4 re-uploading layers
python examples/train.py --lattice D3Q19 --tag MyModel --ansatz Parallel --processing BEL --entangle-layers 3 --layers 4
```

Further options are `--orbits` (train only some orbits), `--samples`, `--epochs` and `--seed`. The circuits are
saved to `examples/output/models/` (or `--out`) as `<tag>_<ansatz>[P<blocks>]_<processing><L_W>_L<L>_<orbit>.pth`,
for example `MyModel_Parallel_BEL3_L4_axis.pth`. Each file records the architecture, the lattice and the training
range, so nothing else needs to be stored; the format is described in [models/README.md](models/README.md).

In Python, the training data and the optimiser can be changed as well:

```python
from hqlbm import D3Q19, CircuitConfig
from hqlbm.data import generate
from hqlbm.train import TrainingConfig, train_circuit, save_checkpoint

config = CircuitConfig("Parallel", "BEL", entangle_layers=3, layers=4, features=3)
data = generate(D3Q19, n_samples=100_000, u_max=0.1, rho_range=(0.98, 1.02), sigma=1e-3)
training = TrainingConfig(epochs=9, batch_size=32, learning_rate=1e-3, lr_decay=0.1, lr_step=3)

for orbit in ("rest", "axis", "diag"):
    checkpoint = train_circuit(D3Q19, orbit, config, data, training, seed=1)
    save_checkpoint(checkpoint, "my_models", "MyModel")
```

- `generate` sets the training range: velocity magnitude, density and non-equilibrium amplitude. Use the circuits
  only inside that range. The relaxation rate is fixed to omega = 1 in the data: the circuit encodes the relaxation
  exactly, so the trained circuit is used with any omega in (0, 2].
- `train_circuit(lattice, target, ...)` trains one orbit when `target` is `"rest"`, `"axis"` or `"diag"`, or one
  direction when `target` is a direction index (per-direction models instead of orbit compression).
- The weights with the lowest validation loss are kept. Pass the same `seed` to repeat a training run exactly.

## Using a modified circuit

```python
from hqlbm import D3Q19, QuantumCollision

collision = QuantumCollision.from_folder("my_models", "MyModel", D3Q19, omega=1.3)
```

- The architecture is read from the files, so no `CircuitConfig` is needed.
- For per-direction models (one file per direction, `<tag>_<circuit>_d3q<i>.pth`), pass `orbit_compression=False`.
- By default the collision is the plain circuit output, with no post-processing. Switch on the optional steps with
  `symmetry=True` (average over the lattice symmetry group) and `conservation=True` (exact mass and momentum).

## Extending the code

All changes are made in `src/circuit.py`.

**A new processing block.**
1. Add its name to `PROCESSINGS`.
2. Apply it in `process()` inside `collision_circuit`.
3. Give its weight shape in `CircuitConfig.circuit_weight_shape`.

For example, `qml.SimplifiedTwoDesign` or a hardware-efficient layer of your own fits here.

**A new encoding.**
1. Add the name to `ANSAETZE`.
2. Define the register size in `CircuitConfig.n_wires` and the number of blocks in `circuit_weight_shape`.
3. Write the encoding loop in `circuit()`.

The rotation axis of `Parallel` and `SuperParallel` is set by `rotation="Y"` in the `AngleEmbedding` call; for
`Line` and `Line+` it is set by `_LINE_AXES`.

**What must stay unchanged** for the circuit to remain a BGK collision:
- the pre-collision qubit with `R_Y(arccos(f_i/rho))`;
- the ancilla with `R_Y(2 arcsin(sqrt(omega/2)))`;
- the CSWAP between the ancilla, the last register qubit and the pre-collision qubit;
- the measurement `<Z>` on the pre-collision qubit.

`tests/test_circuit.py` checks this structure for every ansatz and processing option: `omega/2 = 0` must return
`f_i/rho`, `omega/2 = 1` must return the learned equilibrium, and the values in between must be the weighted sum.
Add a new option to `CONFIGS` in that file and run `pytest` to check it.

A new option must also be readable from a checkpoint. `CircuitConfig.from_checkpoint` reads the `circuit` entry that
`train_circuit` writes, so extend both if the option adds a new setting.
