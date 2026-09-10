import numpy as np

from gate5_multisource_basin import source_defs, train
from self_carving import make_grid


def test_source_split_is_five_train_four_heldout():
    grid = make_grid(11)
    train_defs, heldout, all_defs = source_defs(grid)
    assert len(train_defs) == 5
    assert len(heldout) == 4
    assert len(all_defs) == 9
    assert any(d["dr"] == 0 and d["dc"] == 0 for d in train_defs)
    assert all(abs(d["dr"]) == 1 and abs(d["dc"]) == 1 for d in heldout)


def test_worst_case_training_never_reduces_training_worst_utility():
    run = train("worst_case_multi_source", seed=0, n=7, attempts=30, delta=0.05)
    r = run["result"]
    assert r["train"]["worst_utility"] >= r["baseline_train"]["worst_utility"] - 1e-12
    assert r["material_sum_relative_error"] < 1e-12


def test_no_write_keeps_material_and_baseline_training_measure():
    run = train("no_write", seed=0, n=7, attempts=10)
    r = run["result"]
    assert r["changed_edges"] == 0
    assert r["material_sum_relative_error"] == 0.0
    assert np.isclose(r["train"]["worst_utility"], r["baseline_train"]["worst_utility"])
