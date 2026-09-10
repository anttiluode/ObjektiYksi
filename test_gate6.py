import numpy as np

from gate6_learned_addressing import (
    addresses,
    exact_matrix,
    oracle_choices,
    run_mode,
    task_port_indices,
)
from self_carving import make_grid


def test_gate6_has_three_tasks_and_nine_addresses():
    grid = make_grid(7)
    addr = addresses()
    tasks = task_port_indices(grid)
    assert len(addr) == 9
    assert len(tasks) == 3
    assert len({(a.source_dr, a.omega) for a in addr}) == 9


def test_exact_matrix_and_oracle_have_expected_shape():
    grid = make_grid(7)
    g = np.ones(len(grid.edges), dtype=float)
    addr = addresses()
    tasks = task_port_indices(grid)
    matrix = exact_matrix(grid, g, tasks, addr, sigma2=1e-6)
    choices = oracle_choices(matrix)
    assert len(matrix) == 3
    assert all(len(row) == 9 for row in matrix)
    assert choices.shape == (3,)
    assert np.all((choices >= 0) & (choices < 9))


def test_selector_only_never_changes_material():
    run = run_mode(
        "selector_only",
        seed=0,
        n=7,
        epochs=2,
        probes_per_task=2,
        slow_proposals=2,
    )
    r = run["result"]
    assert r["changed_edges"] == 0
    assert r["accepted"] == 0
    assert r["material_sum_relative_error"] == 0.0


def test_coadaptive_local_writes_conserve_total_material():
    run = run_mode(
        "coadaptive",
        seed=1,
        n=7,
        epochs=2,
        probes_per_task=1,
        slow_proposals=5,
    )
    r = run["result"]
    assert r["material_sum_relative_error"] < 1e-12
    assert 0 <= r["accepted"] <= r["proposed"]
    assert 0.0 <= r["final_selected"]["rank1_fraction"] <= 1.0
    assert 0.0 <= r["selector_oracle_exact_fraction"] <= 1.0
