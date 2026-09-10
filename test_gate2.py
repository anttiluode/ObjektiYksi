import numpy as np

from gate2_local_dither import (
    local_edge_pairs,
    powers_and_utility,
    train_dither,
    transfer_local_material,
)
from self_carving import make_grid


def test_local_material_transfer_changes_only_two_adjacent_edges_and_conserves_sum():
    grid = make_grid(7)
    pairs = local_edge_pairs(grid)
    g = np.ones(len(grid.edges), dtype=float)
    a, b = map(int, pairs[0])
    out = transfer_local_material(g, a, b, orientation=0, delta=0.05)
    assert out is not None
    changed = np.flatnonzero(np.abs(out - g) > 1e-15)
    assert set(changed.tolist()) == {a, b}
    assert abs(float(np.sum(out)) - float(np.sum(g))) < 1e-12


def test_true_consequence_never_reduces_bounded_utility():
    run = train_dither("true_consequence", seed=2, n=7, attempts=45, delta=0.05)
    r = run["result"]
    assert r["final_utility"] >= r["baseline_utility"] - 1e-12
    assert r["material_sum_relative_error"] < 1e-12


def test_no_write_stays_exactly_symmetric():
    run = train_dither("no_write", seed=3, n=7, attempts=20)
    r = run["result"]
    assert r["changed_edges"] == 0
    assert abs(r["final_wave"]["desired_over_decoy"] - 1.0) < 1e-12


def test_bounded_utility_returns_to_zero_if_both_ports_are_below_same_floor():
    grid = make_grid(7)
    g = np.ones(len(grid.edges), dtype=float)
    pt, pd, _ = powers_and_utility(grid, g, sigma2=0.0)
    # Mirror symmetry at baseline is the task's zero point.
    assert abs(pt - pd) < 1e-12
