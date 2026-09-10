import numpy as np

from self_carving import (
    finite_difference_gradient_check,
    make_grid,
    material_step,
    query_metrics,
    receiver_power_gradient_logg,
    train,
)


def test_baseline_is_mirror_symmetric():
    grid = make_grid()
    g = np.ones(len(grid.edges), dtype=float)
    m = query_metrics(grid, g)
    assert abs(m["desired_over_decoy"] - 1.0) < 1e-10


def test_analytic_receiver_gradient_matches_finite_difference():
    check = finite_difference_gradient_check()
    assert check["relative_error"] < 2e-5


def test_one_phase_reference_step_increases_target_power():
    grid = make_grid()
    g = np.ones(len(grid.edges), dtype=float)
    permutation = np.arange(len(g))[::-1]
    before = query_metrics(grid, g)["desired_power"]
    g2, _ = material_step(grid, g, "phase_reference", permutation, eta=0.01)
    after = query_metrics(grid, g2)["desired_power"]
    assert after > before
    assert abs(np.mean(g2) - 1.0) < 1e-12


def test_persistent_write_breaks_original_symmetry_after_state_erase():
    run = train("phase_reference", epochs=35, eta=0.035)
    ratio = run["summary"]["final"]["desired_over_decoy"]
    assert ratio > 1.001
    assert run["summary"]["material_std"] > 1e-3


def test_no_write_stays_symmetric():
    run = train("no_write", epochs=8)
    ratio = run["summary"]["final"]["desired_over_decoy"]
    assert abs(ratio - 1.0) < 1e-10
