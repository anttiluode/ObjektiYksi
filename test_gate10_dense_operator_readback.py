import numpy as np

from gate10_dense_operator_readback import (
    dense_probes,
    effective_matrix,
    hidden_teacher_material,
    matrix_ports,
    solve_probe_block,
    train_one,
)
from self_carving import make_grid


def test_effective_matrix_matches_dense_probe_superposition():
    grid = make_grid(7)
    g = np.ones(len(grid.edges), dtype=float)
    x = dense_probes(5, seed=7)
    m = effective_matrix(grid, g)
    y_direct = solve_probe_block(grid, g, x)
    assert np.allclose(y_direct, m @ x, rtol=1e-11, atol=1e-12)


def test_training_probes_are_dense_and_span_three_inputs():
    x = dense_probes(3, seed=10000)
    assert x.shape == (3, 3)
    assert np.linalg.matrix_rank(x) == 3
    assert np.all(np.abs(x) > 1e-6)


def test_hidden_teacher_uses_same_ports_and_conserves_material():
    grid = make_grid(7)
    inputs, outputs = matrix_ports(grid)
    assert len(inputs) == len(outputs) == 3
    assert set(inputs).isdisjoint(outputs)
    g = hidden_teacher_material(grid, steps=40, seed=3)
    assert np.isclose(np.sum(g), len(g), rtol=0.0, atol=1e-12)
    assert np.count_nonzero(np.abs(g - 1.0) > 1e-12) > 0


def test_small_consequence_run_improves_training_loss_without_basis_training():
    run = train_one(
        "reachable_teacher",
        "consequence",
        seed=0,
        n=7,
        attempts=40,
        train_probes=3,
    )["result"]
    assert run["basis_vectors_used_during_training"] is False
    assert run["train_probe_rank"] == 3
    assert run["final_train_loss"] < run["initial_train_loss"]
    assert run["material_sum_relative_error"] < 1e-12
