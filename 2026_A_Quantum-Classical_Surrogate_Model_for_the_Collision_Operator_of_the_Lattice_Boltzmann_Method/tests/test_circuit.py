"""The collision circuit (PennyLane): weighted-sum structure of the CSWAP / ancilla construction and gradients."""

import itertools
import math

import pytest
import torch

from hqlbm.circuit import CircuitConfig, collision_circuit

CONFIGS = [CircuitConfig(a, p, lw, 2, 2, d)
           for a, p, lw, d in itertools.product(("Line", "Line+", "Parallel", "SuperParallel"), ("SEL", "BEL"),
                                                (1, 2), (2, 3))]


def random_inputs(config, m=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    cw = (2 * torch.rand(config.circuit_weight_shape, generator=gen, dtype=torch.float64) - 1) * math.pi
    sw = 2 * torch.rand(config.scaling_weight_shape, generator=gen, dtype=torch.float64) - 1
    u = 0.1 * (2 * torch.rand(m, config.features, generator=gen, dtype=torch.float64) - 1)
    f = 0.02 + 0.4 * torch.rand(m, 1, generator=gen, dtype=torch.float64)
    return cw, sw, u, f


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.label())
def test_weighted_sum(config):
    """<Z_pre> = (1 - p) f_i / rho + p e(u): p = 0 gives the input population, p = 1 the learned equilibrium e(u)
    (independent of f), and every p in between the weighted sum - the BGK collision."""
    circuit = collision_circuit(config)
    cw, sw, u, f = random_inputs(config)

    def z(f_rho, p):
        return circuit(cw, sw, torch.cat([u, f_rho, torch.full_like(f_rho, p)], dim=1)).view(-1)

    e = z(f, 1.0)
    assert torch.allclose(z(f, 0.0), f.view(-1), atol=1e-14)
    assert torch.allclose(z(0.5 * f, 1.0), e, atol=1e-14)
    for p in (0.3, 0.8):
        assert torch.allclose(z(f, p), (1 - p) * f.view(-1) + p * e, atol=1e-14)


@pytest.mark.parametrize("config", CONFIGS[::5], ids=lambda c: c.label())
def test_gradients(config):
    circuit = collision_circuit(config)
    cw, sw, u, f = random_inputs(config)
    cw.requires_grad_(), sw.requires_grad_()
    circuit(cw, sw, torch.cat([u, f, torch.full_like(f, 0.6)], dim=1)).sum().backward()
    assert torch.isfinite(cw.grad).all() and cw.grad.abs().sum() > 0
    assert torch.isfinite(sw.grad).all() and sw.grad.abs().sum() > 0


def test_weight_shapes():
    assert CircuitConfig("Parallel", "SEL", 1, 6, 2, 2).circuit_weight_shape == (7, 1, 2, 3)      # Corgi
    assert CircuitConfig("SuperParallel", "BEL", 3, 6, 2, 3).circuit_weight_shape == (7, 3, 6)    # LionDog
    assert CircuitConfig("Line", "SEL", 1, 6, 2, 3).circuit_weight_shape == (19, 1, 1, 3)
