from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate2_local_dither import local_edge_pairs, make_schedule, transfer_local_material
from gate10_dense_operator_readback import (
    complex_matrix_json,
    dense_probes,
    effective_matrix,
    hidden_teacher_material,
    matrix_error,
    solve_probe_block,
)
from self_carving import make_grid


OMEGAS = (1.20, 1.55)
SHARED_TEACHER_SEED = 731
CONFLICT_TEACHER_SEEDS = (123, 991)
TRAIN_PROBES = 3
HELDOUT_PROBES = 64
JOINT_ATTEMPTS = 2400
ISOLATED_ATTEMPTS = 1800
DELTA = 0.055

# Fixed before reading the Gate-11 CI receipt.
SUCCESS_MEDIAN_WORST_MATRIX_ERROR = 0.10
SUCCESS_MEDIAN_WORST_HELDOUT_ERROR = 0.10
SUCCESS_EVERY_SEED_WORST_MATRIX_ERROR = 0.20
RANDOM_CONTROL_FACTOR = 0.35
ISOLATED_MEDIAN_ERROR = 0.10
ISOLATED_EVERY_SEED_ERROR = 0.20


def target_family(grid, condition: str):
    """Return two address-conditioned target matrices and their hidden teachers.

    compatible_shared_teacher:
        one hidden material g* generates both targets at two frequencies. A
        single student material can therefore realize the whole target family
        exactly in principle.

    conflicting_independent_teachers:
        each address target comes from a different hidden material. Each target
        is physically reachable by itself, but there need not exist one g that
        realizes both simultaneously. This is the independent-programming
        attack.
    """
    if condition == "compatible_shared_teacher":
        teacher = hidden_teacher_material(grid, seed=SHARED_TEACHER_SEED)
        teachers = [teacher, teacher]
    elif condition == "conflicting_independent_teachers":
        teachers = [
            hidden_teacher_material(grid, seed=CONFLICT_TEACHER_SEEDS[0]),
            hidden_teacher_material(grid, seed=CONFLICT_TEACHER_SEEDS[1]),
        ]
    else:
        raise ValueError(condition)

    targets = [effective_matrix(grid, tg, omega=w) for tg, w in zip(teachers, OMEGAS)]
    return targets, teachers


def probe_sets(condition: str, seed: int):
    # Probe identities are deterministic and condition-independent except for a
    # large seed offset, preventing accidental reuse while keeping reproducible
    # full-rank dense probes. Basis vectors are never used here.
    condition_offset = 0 if condition == "compatible_shared_teacher" else 100000
    x_train = [
        dense_probes(TRAIN_PROBES, condition_offset + 11000 + 100 * seed + slot)
        for slot in range(2)
    ]
    x_heldout = [
        dense_probes(HELDOUT_PROBES, condition_offset + 21000 + 100 * seed + slot)
        for slot in range(2)
    ]
    return x_train, x_heldout


def normalized_loss_at_address(grid, g, omega, x, y_target) -> float:
    y = solve_probe_block(grid, g, x, omega=omega)
    return float(np.linalg.norm(y - y_target) ** 2 / max(np.linalg.norm(y_target) ** 2, 1e-30))


def family_losses(grid, g, targets, x_train):
    return np.asarray(
        [
            normalized_loss_at_address(grid, g, omega, x, target @ x)
            for omega, target, x in zip(OMEGAS, targets, x_train)
        ],
        dtype=float,
    )


def scalar_consequence(losses: np.ndarray) -> float:
    """One bounded scalar judges a proposal while protecting the worse address.

    The max term prevents the learner from sacrificing one address completely;
    the mean term breaks broad plateaus and rewards changes that help both.
    """
    return float(np.max(losses) + 0.25 * np.mean(losses))


def audit_family(grid, g, targets, x_heldout):
    recovered = [effective_matrix(grid, g, omega=w) for w in OMEGAS]
    matrix_errors = [matrix_error(m, t) for m, t in zip(recovered, targets)]
    heldout_errors = []
    for m, target, x in zip(recovered, targets, x_heldout):
        yt = target @ x
        heldout_errors.append(
            float(np.linalg.norm(m @ x - yt) / max(np.linalg.norm(yt), 1e-30))
        )
    return {
        "matrix_errors": [float(v) for v in matrix_errors],
        "heldout_errors": [float(v) for v in heldout_errors],
        "worst_matrix_error": float(max(matrix_errors)),
        "mean_matrix_error": float(np.mean(matrix_errors)),
        "worst_heldout_error": float(max(heldout_errors)),
        "mean_heldout_error": float(np.mean(heldout_errors)),
        "recovered_matrices": [complex_matrix_json(m) for m in recovered],
    }


def train_joint(
    condition: str,
    mode: str,
    seed: int,
    n: int = 11,
    attempts: int = JOINT_ATTEMPTS,
    delta: float = DELTA,
):
    if mode not in {"consequence", "random_accept", "no_write"}:
        raise ValueError(mode)

    grid = make_grid(n)
    g = np.ones(len(grid.edges), dtype=float)
    total0 = float(np.sum(g))
    targets, teachers = target_family(grid, condition)
    x_train, x_heldout = probe_sets(condition, seed)
    y_train = [target @ x for target, x in zip(targets, x_train)]

    train_ranks = [int(np.linalg.matrix_rank(x)) for x in x_train]
    train_min_components = [float(np.min(np.abs(x))) for x in x_train]

    blank_audit = audit_family(grid, g, targets, x_heldout)
    losses = np.asarray(
        [
            normalized_loss_at_address(grid, g, omega, x, y)
            for omega, x, y in zip(OMEGAS, x_train, y_train)
        ],
        dtype=float,
    )
    objective = scalar_consequence(losses)
    initial_objective = objective

    pairs = local_edge_pairs(grid)
    schedule = make_schedule(len(pairs), attempts, seed)
    accepted = 0
    proposed = 0
    checkpoints = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 800, 1200, 1600, 2000, attempts}
    trajectory = [
        {
            "attempt": 0,
            "objective": objective,
            "losses": [float(v) for v in losses],
        }
    ]

    if mode != "no_write":
        for k in range(attempts):
            ea, eb = map(int, pairs[int(schedule.pair_index[k])])
            cand = transfer_local_material(g, ea, eb, int(schedule.orientation[k]), delta=delta)
            if cand is None:
                continue
            proposed += 1
            cand_losses = np.asarray(
                [
                    normalized_loss_at_address(grid, cand, omega, x, y)
                    for omega, x, y in zip(OMEGAS, x_train, y_train)
                ],
                dtype=float,
            )
            cand_objective = scalar_consequence(cand_losses)

            if mode == "consequence":
                take = cand_objective < objective - 1e-14
            else:
                take = bool(schedule.coin[k] < 0.5)

            if take:
                g = cand
                losses = cand_losses
                objective = cand_objective
                accepted += 1

            if (k + 1) in checkpoints:
                trajectory.append(
                    {
                        "attempt": k + 1,
                        "objective": objective,
                        "losses": [float(v) for v in losses],
                    }
                )

    final_audit = audit_family(grid, g, targets, x_heldout)
    teacher_distance = float(
        np.linalg.norm(teachers[0] - teachers[1]) / max(np.linalg.norm(teachers[0]), 1e-30)
    )

    result = {
        "condition": condition,
        "mode": mode,
        "seed": seed,
        "n": n,
        "omegas": [float(w) for w in OMEGAS],
        "attempts": attempts,
        "delta": delta,
        "train_probe_count_per_address": TRAIN_PROBES,
        "train_probe_ranks": train_ranks,
        "train_probe_min_component_abs": train_min_components,
        "basis_vectors_used_during_training": False,
        "teacher_material_hidden_from_learner": True,
        "teacher_material_relative_distance": teacher_distance,
        "initial_objective": initial_objective,
        "final_objective": objective,
        "final_train_losses": [float(v) for v in losses],
        "blank": blank_audit,
        "final": final_audit,
        "accepted": accepted,
        "proposed": proposed,
        "acceptance_fraction": accepted / max(proposed, 1),
        "changed_edges": int(np.count_nonzero(np.abs(g - 1.0) > 1e-12)),
        "material_sum_relative_error": abs(float(np.sum(g)) - total0) / total0,
        "target_matrices": [complex_matrix_json(t) for t in targets],
        "trajectory": trajectory,
    }
    return {"grid": grid, "g": g, "result": result}


def train_isolated_conflict_target(
    slot: int,
    seed: int,
    n: int = 11,
    attempts: int = ISOLATED_ATTEMPTS,
    delta: float = DELTA,
):
    if slot not in {0, 1}:
        raise ValueError(slot)
    condition = "conflicting_independent_teachers"
    grid = make_grid(n)
    g = np.ones(len(grid.edges), dtype=float)
    total0 = float(np.sum(g))
    targets, _ = target_family(grid, condition)
    target = targets[slot]
    omega = OMEGAS[slot]
    x_train, x_heldout = probe_sets(condition, seed)
    x = x_train[slot]
    y = target @ x

    current = normalized_loss_at_address(grid, g, omega, x, y)
    pairs = local_edge_pairs(grid)
    schedule = make_schedule(len(pairs), attempts, 5000 + 100 * slot + seed)
    accepted = 0
    proposed = 0

    for k in range(attempts):
        ea, eb = map(int, pairs[int(schedule.pair_index[k])])
        cand = transfer_local_material(g, ea, eb, int(schedule.orientation[k]), delta=delta)
        if cand is None:
            continue
        proposed += 1
        candidate = normalized_loss_at_address(grid, cand, omega, x, y)
        if candidate < current - 1e-14:
            g = cand
            current = candidate
            accepted += 1

    recovered = effective_matrix(grid, g, omega=omega)
    merr = matrix_error(recovered, target)
    yt = target @ x_heldout[slot]
    hout = float(
        np.linalg.norm(recovered @ x_heldout[slot] - yt) / max(np.linalg.norm(yt), 1e-30)
    )
    return {
        "slot": slot,
        "omega": float(omega),
        "seed": seed,
        "attempts": attempts,
        "matrix_error": merr,
        "heldout_error": hout,
        "accepted": accepted,
        "proposed": proposed,
        "material_sum_relative_error": abs(float(np.sum(g)) - total0) / total0,
    }


def summarize_joint(runs):
    rows = [x["result"] for x in runs]
    arr = lambda fn: np.asarray([float(fn(r)) for r in rows], dtype=float)
    return {
        "seeds": [int(r["seed"]) for r in rows],
        "median_blank_worst_matrix_error": float(np.median(arr(lambda r: r["blank"]["worst_matrix_error"]))),
        "median_final_worst_matrix_error": float(np.median(arr(lambda r: r["final"]["worst_matrix_error"]))),
        "max_final_worst_matrix_error": float(np.max(arr(lambda r: r["final"]["worst_matrix_error"]))),
        "median_final_mean_matrix_error": float(np.median(arr(lambda r: r["final"]["mean_matrix_error"]))),
        "median_final_worst_heldout_error": float(np.median(arr(lambda r: r["final"]["worst_heldout_error"]))),
        "median_final_objective": float(np.median(arr(lambda r: r["final_objective"]))),
        "median_acceptance_fraction": float(np.median(arr(lambda r: r["acceptance_fraction"]))),
        "max_material_sum_relative_error": float(np.max(arr(lambda r: r["material_sum_relative_error"]))),
        "median_teacher_material_relative_distance": float(np.median(arr(lambda r: r["teacher_material_relative_distance"]))),
    }


def summarize_isolated(runs):
    by_slot = {}
    for slot in (0, 1):
        rows = [r for r in runs if int(r["slot"]) == slot]
        errors = np.asarray([float(r["matrix_error"]) for r in rows])
        held = np.asarray([float(r["heldout_error"]) for r in rows])
        by_slot[str(slot)] = {
            "omega": float(OMEGAS[slot]),
            "median_matrix_error": float(np.median(errors)),
            "max_matrix_error": float(np.max(errors)),
            "median_heldout_error": float(np.median(held)),
        }
    return by_slot


def passes_joint(summary, random_summary):
    return bool(
        summary["median_final_worst_matrix_error"] <= SUCCESS_MEDIAN_WORST_MATRIX_ERROR
        and summary["median_final_worst_heldout_error"] <= SUCCESS_MEDIAN_WORST_HELDOUT_ERROR
        and summary["max_final_worst_matrix_error"] <= SUCCESS_EVERY_SEED_WORST_MATRIX_ERROR
        and summary["median_final_worst_matrix_error"]
        <= RANDOM_CONTROL_FACTOR * max(random_summary["median_final_worst_matrix_error"], 1e-30)
    )


def run_all(
    out_dir: Path,
    seeds: int = 3,
    joint_attempts: int = JOINT_ATTEMPTS,
    isolated_attempts: int = ISOLATED_ATTEMPTS,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = ("compatible_shared_teacher", "conflicting_independent_teachers")
    modes = ("consequence", "random_accept", "no_write")
    joint_runs = {
        condition: {
            mode: [
                train_joint(condition, mode, seed=s, attempts=joint_attempts)
                for s in range(seeds)
            ]
            for mode in modes
        }
        for condition in conditions
    }
    joint_summary = {
        condition: {
            mode: summarize_joint(joint_runs[condition][mode])
            for mode in modes
        }
        for condition in conditions
    }

    isolated_runs = [
        train_isolated_conflict_target(slot, seed=s, attempts=isolated_attempts)
        for slot in (0, 1)
        for s in range(seeds)
    ]
    isolated_summary = summarize_isolated(isolated_runs)

    compatible_success = passes_joint(
        joint_summary["compatible_shared_teacher"]["consequence"],
        joint_summary["compatible_shared_teacher"]["random_accept"],
    )
    isolated_success = bool(
        all(
            isolated_summary[str(slot)]["median_matrix_error"] <= ISOLATED_MEDIAN_ERROR
            and isolated_summary[str(slot)]["max_matrix_error"] <= ISOLATED_EVERY_SEED_ERROR
            for slot in (0, 1)
        )
    )
    conflict_success = passes_joint(
        joint_summary["conflicting_independent_teachers"]["consequence"],
        joint_summary["conflicting_independent_teachers"]["random_accept"],
    )

    if not compatible_success:
        verdict = "two_address_family_readback_not_established"
    elif not isolated_success:
        verdict = "independent_teacher_attack_inconclusive_because_isolated_fit_failed"
    elif conflict_success:
        verdict = "two_independently_programmed_addressed_operators_survive_shared_material"
    else:
        verdict = "shared_family_recovered_but_independent_operator_programming_collides"

    conflict_joint = joint_summary["conflicting_independent_teachers"]["consequence"]
    isolated_reference = max(
        isolated_summary["0"]["median_matrix_error"],
        isolated_summary["1"]["median_matrix_error"],
    )
    interference_factor = float(
        conflict_joint["median_final_worst_matrix_error"] / max(isolated_reference, 1e-30)
    )

    receipt = {
        "gate": 11,
        "question": "Can one conserved material g hold two dense 3x3 operators that are selected by physical address (frequency), and can those operators be programmed independently?",
        "addresses": [{"slot": i, "omega": float(w)} for i, w in enumerate(OMEGAS)],
        "compatible_control": "One hidden material generates both target matrices at the two frequencies. Therefore one exact shared g exists by construction.",
        "independent_programming_attack": "The two target matrices come from two different hidden material states. Each target is physically reachable alone, but one shared g may be unable to realize both at once.",
        "isolated_controls": "Each conflicting target is separately trained from blank material with the same dense-probe principle to verify that joint failure is not merely single-target unlearnability.",
        "learning_signal": "For a proposed local conserved material transfer, compute the two normalized dense-probe losses and expose only max(loss)+0.25*mean(loss) as the scalar keep/revert consequence.",
        "basis_vectors_used_during_training": False,
        "readback": "After learning stops, basis queries reconstruct the full 3x3 matrix independently at each frequency address.",
        "success_thresholds_fixed_before_ci_receipt": {
            "joint_median_worst_matrix_error_at_most": SUCCESS_MEDIAN_WORST_MATRIX_ERROR,
            "joint_median_worst_heldout_error_at_most": SUCCESS_MEDIAN_WORST_HELDOUT_ERROR,
            "joint_every_seed_worst_matrix_error_at_most": SUCCESS_EVERY_SEED_WORST_MATRIX_ERROR,
            "joint_median_worst_error_at_most_random_control_times": RANDOM_CONTROL_FACTOR,
            "isolated_each_slot_median_matrix_error_at_most": ISOLATED_MEDIAN_ERROR,
            "isolated_each_slot_every_seed_matrix_error_at_most": ISOLATED_EVERY_SEED_ERROR,
        },
        "joint_summary": joint_summary,
        "isolated_summary": isolated_summary,
        "compatible_success": compatible_success,
        "isolated_success": isolated_success,
        "conflict_success": conflict_success,
        "conflict_joint_vs_isolated_error_factor": interference_factor,
        "verdict": verdict,
        "joint_runs": {
            condition: {
                mode: [x["result"] for x in rs]
                for mode, rs in by_mode.items()
            }
            for condition, by_mode in joint_runs.items()
        },
        "isolated_runs": isolated_runs,
        "claim_boundary": "Success of the compatible control establishes recoverability of two address-conditioned slices from one realizable operator family. Success of the independent-teacher attack would additionally show that one shared material can approximate two separately specified reachable operators at different addresses. Failure of that attack, when isolated controls pass, is evidence of cross-address physical interference/capacity limits rather than evidence against single-operator storage.",
    }
    (out_dir / "gate11_two_address_operator_family.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--joint-attempts", type=int, default=JOINT_ATTEMPTS)
    p.add_argument("--isolated-attempts", type=int, default=ISOLATED_ATTEMPTS)
    args = p.parse_args()
    receipt = run_all(
        Path(args.out),
        seeds=args.seeds,
        joint_attempts=args.joint_attempts,
        isolated_attempts=args.isolated_attempts,
    )
    print(
        json.dumps(
            {
                "joint_summary": receipt["joint_summary"],
                "isolated_summary": receipt["isolated_summary"],
                "verdict": receipt["verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
