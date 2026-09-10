import numpy as np

from gate3_multiport_dither import multiport_measure, right_ports, train_multiport
from self_carving import make_grid


def test_right_port_strip_contains_target_and_decoy():
    grid = make_grid(7)
    ports = right_ports(grid)
    assert grid.target in ports
    assert grid.decoy in ports
    assert len(ports) == grid.n - 2


def test_baseline_multiport_measure_is_finite_and_target_not_rank_one():
    grid = make_grid(11)
    g = np.ones(len(grid.edges), dtype=float)
    m = multiport_measure(grid, g, sigma2=1e-8)
    assert np.isfinite(m["utility"])
    assert m["target_rank"] >= 1
    assert m["target_over_strongest_other"] <= 1.0 + 1e-12


def test_true_multiport_never_reduces_its_scalar_utility():
    run = train_multiport("true_multiport", seed=1, n=7, attempts=50, delta=0.05)
    r = run["result"]
    assert float(r["final"]["utility"]) >= float(r["baseline"]["utility"]) - 1e-12
    assert r["material_sum_relative_error"] < 1e-12


def test_no_write_stays_exactly_at_baseline_material():
    run = train_multiport("no_write", seed=0, n=7, attempts=20)
    r = run["result"]
    assert r["changed_edges"] == 0
    assert r["material_sum_relative_error"] == 0.0
