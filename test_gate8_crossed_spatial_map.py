import numpy as np

from gate8_crossed_spatial_map import (
    common_sigma2,
    mapping_measure,
    relation_defs,
    train_mode,
)
from self_carving import make_grid


def test_crossed_mapping_is_a_true_three_cycle():
    grid = make_grid(11)
    identity = relation_defs(grid, "identity")
    crossed = relation_defs(grid, "crossed")

    id_targets = [r["target_port_index"] for r in identity]
    crossed_targets = [r["target_port_index"] for r in crossed]

    assert crossed_targets == [id_targets[2], id_targets[0], id_targets[1]]
    assert all(a != b for a, b in zip(id_targets, crossed_targets))


def test_common_floor_does_not_depend_on_mapping():
    grid = make_grid(9)
    sigma2 = common_sigma2(grid)
    assert sigma2 > 0

    g = np.ones(len(grid.edges), dtype=float)
    identity = mapping_measure(grid, g, "identity", sigma2)
    crossed = mapping_measure(grid, g, "crossed", sigma2)
    assert identity["rows"] and crossed["rows"]


def test_crossed_learning_conserves_material():
    run = train_mode("crossed", seed=0, n=7, attempts=12)
    r = run["result"]
    assert r["material_sum_relative_error"] < 1e-12
    assert 0 <= r["accepted"] <= r["proposed"]
    assert len(r["final"]["rows"]) == 3


def test_no_write_keeps_blank_crossed_map():
    run = train_mode("no_write_crossed", seed=0, n=7, attempts=12)
    r = run["result"]
    assert r["accepted"] == 0
    assert r["changed_edges"] == 0
    assert abs(r["baseline"]["min_ratio"] - r["final"]["min_ratio"]) < 1e-15
