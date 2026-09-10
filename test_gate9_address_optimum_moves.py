import numpy as np

from gate8_crossed_spatial_map import common_sigma2, train_mode
from gate9_address_optimum_moves import audit_target_sources, target_source_maps
from self_carving import make_grid


def test_blank_native_and_crossed_assignments_differ_for_all_targets():
    grid = make_grid(11)
    native, assigned = target_source_maps(grid)
    assert set(native) == set(assigned)
    assert all(native[t] != assigned[t] for t in native)


def test_blank_audit_has_three_targets_and_three_sources_each():
    grid = make_grid(9)
    g = np.ones(len(grid.edges), dtype=float)
    audit = audit_target_sources(grid, g, common_sigma2(grid))
    assert len(audit["rows"]) == 3
    assert all(len(row["by_source"]) == 3 for row in audit["rows"])


def test_gate9_audit_does_not_change_gate8_material():
    trained = train_mode("crossed", seed=0, n=7, attempts=12)
    g_before = trained["g"].copy()
    _ = audit_target_sources(trained["grid"], trained["g"], common_sigma2(trained["grid"]))
    assert np.array_equal(g_before, trained["g"])
