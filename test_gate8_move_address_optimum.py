import numpy as np

from gate8_move_address_optimum import (
    ALIGNED_CHOICES,
    FORCED_CHOICES,
    POSITION_ADDRESSES,
    run_mode,
)


def test_gate8_forced_map_is_cyclic_and_changes_every_task():
    assert [a.source_dr for a in POSITION_ADDRESSES] == [-1, 0, 1]
    assert np.array_equal(ALIGNED_CHOICES, np.asarray([0, 1, 2]))
    assert np.array_equal(FORCED_CHOICES, np.asarray([1, 2, 0]))
    assert np.all(FORCED_CHOICES != ALIGNED_CHOICES)


def test_gate8_small_consequence_run_conserves_material():
    run = run_mode(
        "forced_consequence",
        seed=0,
        n=7,
        epochs=3,
        slow_proposals=5,
    )
    r = run["result"]
    assert r["material_sum_relative_error"] < 1e-12
    assert 0 <= r["accepted"] <= r["proposed"]
    assert 0.0 <= r["final_forced_is_oracle_fraction"] <= 1.0


def test_gate8_no_write_keeps_material_blank():
    run = run_mode(
        "forced_no_write",
        seed=0,
        n=7,
        epochs=2,
        slow_proposals=3,
    )
    r = run["result"]
    assert r["accepted"] == 0
    assert r["changed_edges"] == 0
    assert r["material_sum_relative_error"] == 0.0
