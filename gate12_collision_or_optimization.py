from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate2_local_dither import local_edge_pairs, make_schedule, transfer_local_material
from gate10_dense_operator_readback import effective_matrix, matrix_error, matrix_ports
from gate11_two_address_operator_family import (
    OMEGAS,
    audit_family,
    normalized_loss_at_address,
    probe_sets,
    scalar_consequence,
    target_family,
    train_joint,
)
from self_carving import operator_matrix


GATE11_ATTEMPTS = 2400
EXTRA_LOCAL_ATTEMPTS = 7200
DELTA = 0.055
GN_STEPS = 5

# These keep the Gate-11 bar fixed rather than moving it after seeing Gate 11.
RESCUE_MATRIX_ERROR = 0.10
RESCUE_HELDOUT_ERROR = 0.10
FULL_STACK_REAL_RANK = 36  # two 3x3 complex matrices = 36 real output coordinates
LINEARIZED_RESIDUAL_RATIO = 0.10


def complex_operator_jacobian(grid, g: np.ndarray, omega: float) -> np.ndarray:
    """Exact d vec(M_g(omega)) / d g as a complex (9 x n_edges) Jacobian.

    With A(g)=K(g)-omega^2 I+i omega Gamma and H=A^-1,

        M = C H B
        dM/dg_e = -(C H b_e)(b_e^T H B),

    because dK/dg_e=b_e b_e^T for an edge incidence vector b_e.
    """
    inputs, outputs = matrix_ports(grid)
    n_nodes = grid.n * grid.n
    rhs_in = np.zeros((n_nodes, len(inputs)), dtype=complex)
    rhs_out = np.zeros((n_nodes, len(outputs)), dtype=complex)
    for k, src in enumerate(inputs):
        rhs_in[src, k] = 1.0
    for k, dst in enumerate(outputs):
        rhs_out[dst, k] = 1.0

    A = operator_matrix(grid, g, omega=omega)
    u = np.linalg.solve(A, rhs_in)   # H B
    r = np.linalg.solve(A, rhs_out)  # H C^T; A is complex symmetric
    i = grid.edges[:, 0]
    j = grid.edges[:, 1]
    du = u[i, :] - u[j, :]
    dr = r[i, :] - r[j, :]

    jac = np.empty((9, len(grid.edges)), dtype=complex)
    for e in range(len(grid.edges)):
        jac[:, e] = (-np.outer(dr[e, :], du[e, :])).reshape(-1)
    return jac


def real_normalized_residual_and_jacobian(grid, g, target, omega):
    current = effective_matrix(grid, g, omega=omega)
    scale = max(float(np.linalg.norm(target)), 1e-30)
    rc = (current - target).reshape(-1) / scale
    jc = complex_operator_jacobian(grid, g, omega) / scale
    return np.concatenate([rc.real, rc.imag]), np.vstack([jc.real, jc.imag])


def tangent_projector(n_edges: int) -> np.ndarray:
    one = np.ones((n_edges, 1), dtype=float)
    return np.eye(n_edges, dtype=float) - (one @ one.T) / float(n_edges)


def svd_rank(s: np.ndarray, shape) -> int:
    if len(s) == 0:
        return 0
    tol = max(shape) * np.finfo(float).eps * float(s[0])
    return int(np.count_nonzero(s > tol))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if den <= 1e-30 else float(np.dot(a, b) / den)


def tangent_audit(grid, g, targets):
    P = tangent_projector(len(g))
    residuals, jacobians, rows = [], [], []

    for slot, (target, omega) in enumerate(zip(targets, OMEGAS)):
        r, J = real_normalized_residual_and_jacobian(grid, g, target, omega)
        Jp = J @ P
        s = np.linalg.svd(Jp, compute_uv=False)
        rank = svd_rank(s, Jp.shape)
        grad = P @ (J.T @ r)
        rows.append({
            "slot": slot,
            "omega": float(omega),
            "residual_norm": float(np.linalg.norm(r)),
            "tangent_rank": rank,
            "largest_singular": float(s[0]) if len(s) else 0.0,
            "smallest_nonzero_singular": float(s[rank - 1]) if rank else 0.0,
            "gradient_norm": float(np.linalg.norm(grad)),
        })
        residuals.append(r)
        jacobians.append(J)

    r_stack = np.concatenate(residuals)
    J_stack = np.vstack(jacobians)
    Jp_stack = J_stack @ P
    s_stack = np.linalg.svd(Jp_stack, compute_uv=False)
    rank_stack = svd_rank(s_stack, Jp_stack.shape)

    z, *_ = np.linalg.lstsq(Jp_stack, -r_stack, rcond=None)
    d = P @ z
    linear_after = r_stack + J_stack @ d
    linear_ratio = float(np.linalg.norm(linear_after) / max(np.linalg.norm(r_stack), 1e-30))

    grad0 = P @ (jacobians[0].T @ residuals[0])
    grad1 = P @ (jacobians[1].T @ residuals[1])

    individual = []
    for slot in (0, 1):
        Jp = jacobians[slot] @ P
        z_i, *_ = np.linalg.lstsq(Jp, -residuals[slot], rcond=None)
        d_i = P @ z_i
        other = 1 - slot
        own_before = float(np.linalg.norm(residuals[slot]))
        other_before = float(np.linalg.norm(residuals[other]))
        own_after = float(np.linalg.norm(residuals[slot] + jacobians[slot] @ d_i))
        other_after = float(np.linalg.norm(residuals[other] + jacobians[other] @ d_i))
        individual.append({
            "slot": slot,
            "own_linear_residual_ratio": own_after / max(own_before, 1e-30),
            "other_linear_residual_ratio": other_after / max(other_before, 1e-30),
        })

    return {
        "per_address": rows,
        "stacked_real_output_dimension": int(len(r_stack)),
        "stacked_tangent_rank": rank_stack,
        "stacked_largest_singular": float(s_stack[0]) if len(s_stack) else 0.0,
        "stacked_smallest_nonzero_singular": float(s_stack[rank_stack - 1]) if rank_stack else 0.0,
        "stacked_condition_number_nonzero": float(s_stack[0] / s_stack[rank_stack - 1]) if rank_stack else float("inf"),
        "gradient_cosine": cosine(grad0, grad1),
        "best_linearized_residual_ratio": linear_ratio,
        "individual_linear_corrections": individual,
        "joint_least_squares_direction": d,
    }


def matrix_objective(grid, g, targets) -> float:
    errs = np.asarray([
        matrix_error(effective_matrix(grid, g, omega=w), t) ** 2
        for t, w in zip(targets, OMEGAS)
    ], dtype=float)
    return scalar_consequence(errs)


def feasible_alpha(g: np.ndarray, d: np.ndarray) -> float:
    alpha = 1.0
    pos = d > 1e-15
    neg = d < -1e-15
    if np.any(pos):
        alpha = min(alpha, float(np.min((4.0 - g[pos]) / d[pos])))
    if np.any(neg):
        alpha = min(alpha, float(np.min((g[neg] - 0.08) / (-d[neg]))))
    return max(0.0, alpha)


def gauss_newton_diagnostic(grid, g_start, targets, x_heldout, steps: int = GN_STEPS):
    """Audit-only global derivative diagnostic, not a claimed learning rule."""
    g = g_start.copy()
    total0 = float(np.sum(g))
    trace = []

    for step in range(steps + 1):
        audit = audit_family(grid, g, targets, x_heldout)
        objective = matrix_objective(grid, g, targets)
        trace.append({
            "step": step,
            "objective": objective,
            "worst_matrix_error": audit["worst_matrix_error"],
            "worst_heldout_error": audit["worst_heldout_error"],
        })
        if step == steps:
            break

        ta = tangent_audit(grid, g, targets)
        d = np.asarray(ta["joint_least_squares_direction"], dtype=float)
        max_alpha = feasible_alpha(g, d)
        if max_alpha <= 1e-12 or np.linalg.norm(d) <= 1e-30:
            break

        base = objective
        best = None
        for k in range(11):
            alpha = max_alpha * (0.5 ** k)
            cand = g + alpha * d
            if np.min(cand) < 0.08 - 1e-12 or np.max(cand) > 4.0 + 1e-12:
                continue
            value = matrix_objective(grid, cand, targets)
            if value < base - 1e-14 and (best is None or value < best[0]):
                best = (value, cand)
        if best is None:
            break
        g = best[1]
        g += (total0 - float(np.sum(g))) / len(g)

    final = audit_family(grid, g, targets, x_heldout)
    return {
        "final": final,
        "trace": trace,
        "steps_taken": len(trace) - 1,
        "material_sum_relative_error": abs(float(np.sum(g)) - total0) / total0,
    }


def continue_local_search(grid, g_start, targets, seed: int, extra_attempts: int = EXTRA_LOCAL_ATTEMPTS):
    """Continue Gate-11's original derivative-free local learner from its exact final g."""
    g = g_start.copy()
    total0 = float(np.sum(g))
    x_train, x_heldout = probe_sets("conflicting_independent_teachers", seed)
    y_train = [target @ x for target, x in zip(targets, x_train)]
    losses = np.asarray([
        normalized_loss_at_address(grid, g, omega, x, y)
        for omega, x, y in zip(OMEGAS, x_train, y_train)
    ], dtype=float)
    objective = scalar_consequence(losses)

    pairs = local_edge_pairs(grid)
    schedule = make_schedule(len(pairs), extra_attempts, 60000 + seed)
    checkpoints = {600, 1200, 2400, 3600, 4800, 7200, extra_attempts}
    trace = []

    def record(k):
        a = audit_family(grid, g, targets, x_heldout)
        trace.append({
            "extra_attempts": int(k),
            "training_objective": float(objective),
            "worst_matrix_error": a["worst_matrix_error"],
            "mean_matrix_error": a["mean_matrix_error"],
            "worst_heldout_error": a["worst_heldout_error"],
        })

    record(0)
    accepted = proposed = 0
    for k in range(extra_attempts):
        ea, eb = map(int, pairs[int(schedule.pair_index[k])])
        cand = transfer_local_material(g, ea, eb, int(schedule.orientation[k]), delta=DELTA)
        if cand is None:
            continue
        proposed += 1
        cand_losses = np.asarray([
            normalized_loss_at_address(grid, cand, omega, x, y)
            for omega, x, y in zip(OMEGAS, x_train, y_train)
        ], dtype=float)
        cand_obj = scalar_consequence(cand_losses)
        if cand_obj < objective - 1e-14:
            g = cand
            losses = cand_losses
            objective = cand_obj
            accepted += 1
        if (k + 1) in checkpoints:
            record(k + 1)

    return {
        "final": audit_family(grid, g, targets, x_heldout),
        "trace": trace,
        "accepted": accepted,
        "proposed": proposed,
        "acceptance_fraction": accepted / max(proposed, 1),
        "material_sum_relative_error": abs(float(np.sum(g)) - total0) / total0,
    }


def run_seed(seed: int, extra_local_attempts: int = EXTRA_LOCAL_ATTEMPTS):
    base = train_joint(
        "conflicting_independent_teachers",
        "consequence",
        seed=seed,
        attempts=GATE11_ATTEMPTS,
    )
    grid, g = base["grid"], base["g"]
    targets, _ = target_family(grid, "conflicting_independent_teachers")
    _, x_heldout = probe_sets("conflicting_independent_teachers", seed)

    tangent = tangent_audit(grid, g, targets)
    tangent.pop("joint_least_squares_direction", None)
    return {
        "seed": seed,
        "gate11_endpoint": base["result"]["final"],
        "tangent": tangent,
        "oracle_gauss_newton": gauss_newton_diagnostic(grid, g, targets, x_heldout),
        "continued_local": continue_local_search(
            grid, g, targets, seed, extra_attempts=extra_local_attempts
        ),
    }


def summarize(runs):
    def arr(fn):
        return np.asarray([float(fn(r)) for r in runs], dtype=float)

    ranks = [int(r["tangent"]["stacked_tangent_rank"]) for r in runs]
    cosines = arr(lambda r: r["tangent"]["gradient_cosine"])
    linear = arr(lambda r: r["tangent"]["best_linearized_residual_ratio"])
    gate11 = arr(lambda r: r["gate11_endpoint"]["worst_matrix_error"])
    local_m = arr(lambda r: r["continued_local"]["final"]["worst_matrix_error"])
    local_h = arr(lambda r: r["continued_local"]["final"]["worst_heldout_error"])
    oracle_m = arr(lambda r: r["oracle_gauss_newton"]["final"]["worst_matrix_error"])
    oracle_h = arr(lambda r: r["oracle_gauss_newton"]["final"]["worst_heldout_error"])

    return {
        "seeds": [int(r["seed"]) for r in runs],
        "stacked_tangent_ranks": ranks,
        "all_stacked_tangent_ranks_full_36": bool(all(v == FULL_STACK_REAL_RANK for v in ranks)),
        "median_gradient_cosine": float(np.median(cosines)),
        "min_gradient_cosine": float(np.min(cosines)),
        "max_gradient_cosine": float(np.max(cosines)),
        "median_best_linearized_residual_ratio": float(np.median(linear)),
        "max_best_linearized_residual_ratio": float(np.max(linear)),
        "median_gate11_endpoint_worst_matrix_error": float(np.median(gate11)),
        "median_continued_local_worst_matrix_error": float(np.median(local_m)),
        "max_continued_local_worst_matrix_error": float(np.max(local_m)),
        "median_continued_local_worst_heldout_error": float(np.median(local_h)),
        "median_oracle_worst_matrix_error": float(np.median(oracle_m)),
        "max_oracle_worst_matrix_error": float(np.max(oracle_m)),
        "median_oracle_worst_heldout_error": float(np.median(oracle_h)),
    }


def run_all(out_dir: Path, seeds: int = 3, extra_local_attempts: int = EXTRA_LOCAL_ATTEMPTS):
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [run_seed(s, extra_local_attempts) for s in range(seeds)]
    summary = summarize(runs)

    full_rank = bool(summary["all_stacked_tangent_ranks_full_36"])
    linear_room = bool(summary["max_best_linearized_residual_ratio"] <= LINEARIZED_RESIDUAL_RATIO)
    local_rescue = bool(
        summary["median_continued_local_worst_matrix_error"] <= RESCUE_MATRIX_ERROR
        and summary["median_continued_local_worst_heldout_error"] <= RESCUE_HELDOUT_ERROR
    )
    oracle_rescue = bool(
        summary["median_oracle_worst_matrix_error"] <= RESCUE_MATRIX_ERROR
        and summary["median_oracle_worst_heldout_error"] <= RESCUE_HELDOUT_ERROR
    )

    if local_rescue:
        verdict = "gate11_near_miss_was_budget_limited_local_search_rescues"
    elif oracle_rescue and full_rank and linear_room:
        verdict = "gate11_penalty_is_optimizer_limited_not_a_local_manifold_wall"
    elif (not full_rank or not linear_room) and not oracle_rescue:
        verdict = "local_operator_manifold_collision_supported"
    else:
        verdict = "mixed_collision_vs_optimization_result"

    receipt = {
        "gate": 12,
        "question": "Was Gate 11's independent-operator penalty a derivative-free optimization limit or a local geometric collision in the shared physical operator manifold?",
        "starting_point": "Recreate each exact Gate-11 conflicting consequence endpoint after 2400 local proposals.",
        "exact_jacobian": "For each address, dM/dg_e = -(C H b_e)(b_e^T H B). Real and imaginary matrix coordinates are stacked and the fixed-total-material tangent is projected explicitly.",
        "tangent_test": "Audit the rank of the stacked two-address real Jacobian (36 output coordinates), cosine of the two per-address loss gradients, and the least-squares first-order residual floor.",
        "oracle_diagnostic": "Use at most five exact-Jacobian Gauss-Newton steps with nonlinear backtracking and the same material box/budget constraints. This is audit-only and is not a claimed local learner.",
        "local_budget_test": "From the same Gate-11 endpoint, continue only the original derivative-free local keep/revert rule for a fresh fixed stream of extra proposals.",
        "fixed_interpretation_thresholds": {
            "matrix_error_rescue_at_most": RESCUE_MATRIX_ERROR,
            "heldout_error_rescue_at_most": RESCUE_HELDOUT_ERROR,
            "full_stacked_real_rank": FULL_STACK_REAL_RANK,
            "best_linearized_residual_ratio_at_most_for_local_room": LINEARIZED_RESIDUAL_RATIO,
        },
        "summary": summary,
        "full_rank": full_rank,
        "linear_room": linear_room,
        "local_rescue": local_rescue,
        "oracle_rescue": oracle_rescue,
        "verdict": verdict,
        "runs": runs,
        "claim_boundary": "A local or oracle rescue disproves the interpretation of Gate 11 as a demonstrated hard capacity wall. Full tangent rank and a small linearized residual floor establish only local degrees of freedom, not global realizability. Failure of both tests would strengthen but still not prove a global impossibility claim.",
    }
    (out_dir / "gate12_collision_or_optimization.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--extra-local-attempts", type=int, default=EXTRA_LOCAL_ATTEMPTS)
    args = p.parse_args()
    receipt = run_all(Path(args.out), args.seeds, args.extra_local_attempts)
    print(json.dumps({"summary": receipt["summary"], "verdict": receipt["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
