import numpy as np

from gate10_dense_operator_readback import effective_matrix
from gate11_two_address_operator_family import target_family
from gate12_collision_or_optimization import (
    complex_operator_jacobian,
    feasible_alpha,
    tangent_audit,
    tangent_projector,
)
from self_carving import make_grid


def test_exact_operator_jacobian_matches_central_difference():
    grid = make_grid(7)
    g = np.ones(len(grid.edges), dtype=float)
    # Move away from exact blank symmetry while preserving total material.
    g[3] += 0.07
    g[4] -= 0.07
    omega = 1.20
    edge = 9
    analytic = complex_operator_jacobian(grid, g, omega)[:, edge].reshape(3, 3)

    eps = 1e-6
    gp = g.copy()
    gm = g.copy()
    gp[edge] += eps
    gm[edge] -= eps
    numeric = (effective_matrix(grid, gp, omega=omega) - effective_matrix(grid, gm, omega=omega)) / (2.0 * eps)

    rel = np.linalg.norm(analytic - numeric) / max(np.linalg.norm(numeric), 1e-30)
    assert rel < 2e-6


def test_tangent_projector_preserves_only_fixed_sum_directions():
    P = tangent_projector(12)
    one = np.ones(12)
    assert np.linalg.norm(P @ one) < 1e-12
    assert np.linalg.norm(P @ P - P) < 1e-12
    assert np.linalg.norm(P.T - P) < 1e-12


def test_tangent_audit_direction_conserves_first_order_material():
    grid = make_grid(7)
    g = np.ones(len(grid.edges), dtype=float)
    targets, _ = target_family(grid, "conflicting_independent_teachers")
    audit = tangent_audit(grid, g, targets)
    d = np.asarray(audit["joint_least_squares_direction"])
    assert abs(float(np.sum(d))) < 1e-10
    assert 0 <= audit["stacked_tangent_rank"] <= 36
    assert audit["best_linearized_residual_ratio"] >= 0.0


def test_feasible_alpha_keeps_box_constraints():
    g = np.asarray([1.0, 1.0, 0.09, 3.9])
    d = np.asarray([1.0, -1.0, -1.0, 1.0])
    alpha = feasible_alpha(g, d)
    cand = g + alpha * d
    assert alpha >= 0.0
    assert np.min(cand) >= 0.08 - 1e-12
    assert np.max(cand) <= 4.0 + 1e-12
