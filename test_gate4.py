import numpy as np

from gate4_routing_basin import (
    TRAIN_OMEGA,
    contiguous_rank1_band,
    evaluate_material,
    nearby_sources,
)
from self_carving import make_grid


def test_nearby_source_patch_contains_training_source():
    grid = make_grid(11)
    defs = nearby_sources(grid)
    assert len(defs) == 9
    assert any(d["dr"] == 0 and d["dc"] == 0 and d["node"] == grid.source for d in defs)


def test_uncarved_evaluation_contains_frequency_and_source_controls():
    grid = make_grid(11)
    g = np.ones(len(grid.edges), dtype=float)
    ev = evaluate_material(grid, g)
    assert ev["frequency_all"]["samples"] == 17
    assert ev["frequency_heldout"]["samples"] == 16
    assert ev["source_all"]["samples"] == 9
    assert ev["source_heldout"]["samples"] == 8
    assert ev["joint_all"]["samples"] == 45
    assert ev["joint_heldout"]["samples"] == 44


def test_contiguous_rank1_band_zero_if_training_point_not_rank_one():
    rows = [
        {"omega": 1.50, "target_rank": 1},
        {"omega": TRAIN_OMEGA, "target_rank": 2},
        {"omega": 1.60, "target_rank": 1},
    ]
    band = contiguous_rank1_band(rows)
    assert band["width"] == 0.0
    assert band["samples"] == 0
