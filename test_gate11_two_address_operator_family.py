import numpy as np

from gate11_two_address_operator_family import (
    OMEGAS,
    audit_family,
    probe_sets,
    scalar_consequence,
    target_family,
    train_joint,
)
from self_carving import make_grid


def test_gate11_targets_are_two_dense_3x3_matrices():
    grid = make_grid(7)
    for condition in ("compatible_shared_teacher", "conflicting_independent_teachers"):
        targets, teachers = target_family(grid, condition)
        assert len(targets) == len(teachers) == len(OMEGAS) == 2
        assert all(t.shape == (3, 3) for t in targets)
        assert all(np.iscomplexobj(t) for t in targets)


def test_compatible_condition_reuses_teacher_but_conflict_does_not():
    grid = make_grid(7)
    _, compatible = target_family(grid, "compatible_shared_teacher")
    _, conflict = target_family(grid, "conflicting_independent_teachers")
    assert np.array_equal(compatible[0], compatible[1])
    assert not np.array_equal(conflict[0], conflict[1])


def test_training_probes_are_dense_full_rank_and_not_basis():
    x_train, _ = probe_sets("compatible_shared_teacher", seed=0)
    for x in x_train:
        assert x.shape == (3, 3)
        assert np.linalg.matrix_rank(x) == 3
        assert np.min(np.abs(x)) > 0.0
        assert not np.array_equal(x, np.eye(3))


def test_scalar_consequence_penalizes_worst_address():
    balanced = scalar_consequence(np.asarray([0.2, 0.2]))
    sacrificed = scalar_consequence(np.asarray([0.01, 0.5]))
    assert sacrificed > balanced


def test_gate11_small_run_conserves_material_and_audit_is_read_only():
    trained = train_joint(
        "compatible_shared_teacher",
        "consequence",
        seed=0,
        n=7,
        attempts=6,
    )
    result = trained["result"]
    assert result["basis_vectors_used_during_training"] is False
    assert result["material_sum_relative_error"] < 1e-12
    g_before = trained["g"].copy()
    targets, _ = target_family(trained["grid"], "compatible_shared_teacher")
    _, x_heldout = probe_sets("compatible_shared_teacher", seed=0)
    audit = audit_family(trained["grid"], trained["g"], targets, x_heldout)
    assert len(audit["matrix_errors"]) == 2
    assert np.array_equal(g_before, trained["g"])
