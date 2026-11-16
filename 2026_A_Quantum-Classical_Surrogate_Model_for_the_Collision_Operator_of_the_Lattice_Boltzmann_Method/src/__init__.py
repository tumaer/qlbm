"""hqlbm - hybrid quantum-classical lattice Boltzmann method with a variational quantum BGK collision.

    lattice        D2Q9 / D3Q19 velocity sets, orbits and hyperoctahedral symmetry group
    lbm            moments, equilibrium, classical BGK reference, streaming (periodic, moving walls), time loop
    circuit        the collision circuit (PennyLane, state-vector simulator default.qubit) and its configuration
    collision      QuantumCollision: trained circuits as an LBM collision operator (orbit compression)
    postprocessing optional post-processing of the circuit output: symmetry (SYM), conservation (CONS)
    data           synthetic training data
    train          training of one circuit (orbit or direction)
"""

from hqlbm.circuit import CircuitConfig, collision_circuit
from hqlbm.collision import QuantumCollision, load_circuits
from hqlbm.lattice import D2Q9, D3Q19, LATTICES, Lattice
from hqlbm.lbm import BGK, Streaming, equilibrium, moments, run, step

__version__ = "1.0.0"
__all__ = ["BGK", "CircuitConfig", "D2Q9", "D3Q19", "LATTICES", "Lattice", "QuantumCollision", "collision_circuit",
           "Streaming", "equilibrium", "load_circuits", "moments", "run", "step"]
