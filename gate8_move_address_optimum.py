from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate2_local_dither import local_edge_pairs, transfer_local_material
from gate6_learned_addressing import (
    Address,
    addresses,
    baseline_floor,
    choice_records,
    exact_matrix,
    family_measure,
    oracle_choices,
    task_port_indices,
)
from self_carving import make_grid, write_svg


# Gate 7 showed that source position alone is sufficient for the current tasks.
# Gate 8 therefore removes frequency as a degree of freedom and asks whether
# learning can rewrite the *spatial* coordinate chart itself.
POSITION_ADDRESSES = [Address(source_dr=dr, omega=1.35) for dr in (-1, 0, 1)]

# Blank geometry naturally maps upper target -> upper source, center -> center,
# lower -> lower. Force a cyclic permutation so every task starts at the wrong
# source row. Indices refer to POSITION_ADDRESSES above.
FORCED_CHOICES = np.asarray([1, 2, 0], dtype=int)
ALIGNED_CHOICES = np.asarray([0, 1, 2], dtype=int)

RANDOM_ACCEPT_PROBABILITY = 0.18


def common_sigma2(grid, tasks: list[int], floor_fraction: float) -> float:
    """Keep the Gate-6/7 observer definition unchanged."""
    return baseline_floor(grid, tasks, addresses(), floor_fraction)


def exact_oracle(grid, g, tasks, sigma2):
    matrix = exact_matrix(grid, g, tasks, POSITION_ADDRESSES, sigma2)
    return matrix, oracle_choices(matrix)


def run_mode(
    mode: str,
    seed: int,
    n: int = 11,
    epochs: int = 60,
    slow_proposals: int = 40,
    delta: float = 0.055,
    floor_fraction: float = 0.01,
):
    allowed = {"forced_consequence", "aligned_consequence", "forced_random", "forced_no_write"}
    if mode not in allowed:
        raise ValueError(mode)

    grid = make_grid(n)
    tasks = task_port_indices(grid)
    g = np.ones(len(grid.edges), dtype=float)
    total0 = float(np.sum(g))
    sigma2 = common_sigma2(grid, tasks, floor_fraction)

    blank_matrix, blank_oracle = exact_oracle(grid, g, tasks, sigma2)
    blank_forced = family_measure(
        grid, g, tasks, POSITION_ADDRESSES, FORCED_CHOICES, sigma2
    )
    blank_aligned = family_measure(
        grid, g, tasks, POSITION_ADDRESSES, ALIGNED_CHOICES, sigma2
    )

    if mode == "aligned_consequence":
        train_choices = ALIGNED_CHOICES.copy()
    else:
        train_choices = FORCED_CHOICES.copy()

    proposal_rng = np.random.default_rng(70000 + seed)
    accept_rng = np.random.default_rng(80000 + seed)
    pairs = local_edge_pairs(grid)
    total_steps = epochs * slow_proposals
    pair_schedule = proposal_rng.integers(len(pairs), size=total_steps)
    orientation_schedule = np.where(
        proposal_rng.random(total_steps) < 0.5, 1, -1
    ).astype(int)

    accepted = 0
    proposed = 0
    cursor = 0
    oracle_switches = 0
    last_oracle = blank_oracle.copy()
    trajectory = []

    current = family_measure(
        grid, g, tasks, POSITION_ADDRESSES, train_choices, sigma2
    )

    for epoch in range(epochs):
        for _ in range(slow_proposals):
            ea, eb = map(int, pairs[int(pair_schedule[cursor])])
            orientation = int(orientation_schedule[cursor])
            cursor += 1
            cand = transfer_local_material(g, ea, eb, orientation, delta=delta)
            if cand is None:
                continue
            proposed += 1

            if mode == "forced_no_write":
                continue

            if mode == "forced_random":
                if accept_rng.random() < RANDOM_ACCEPT_PROBABILITY:
                    g = cand
                    accepted += 1
                    current = family_measure(
                        grid, g, tasks, POSITION_ADDRESSES, train_choices, sigma2
                    )
                continue

            cm = family_measure(
                grid, cand, tasks, POSITION_ADDRESSES, train_choices, sigma2
            )
            if float(cm["worst_utility"]) > float(current["worst_utility"]) + 1e-14:
                g = cand
                current = cm
                accepted += 1

        audit_matrix, oracle = exact_oracle(grid, g, tasks, sigma2)
        oracle_switches += int(np.count_nonzero(oracle != last_oracle))
        last_oracle = oracle.copy()
        forced_eval = family_measure(
            grid, g, tasks, POSITION_ADDRESSES, FORCED_CHOICES, sigma2
        )
        oracle_eval = family_measure(
            grid, g, tasks, POSITION_ADDRESSES, oracle, sigma2
        )
        trajectory.append(
            {
                "epoch": epoch + 1,
                "forced_min_ratio": float(forced_eval["min_ratio"]),
                "forced_rank1_fraction": float(forced_eval["rank1_fraction"]),
                "forced_is_oracle_fraction": float(np.mean(oracle == FORCED_CHOICES)),
                "oracle_choices": choice_records(POSITION_ADDRESSES, oracle),
                "oracle_min_ratio": float(oracle_eval["min_ratio"]),
            }
        )

    final_matrix, final_oracle = exact_oracle(grid, g, tasks, sigma2)
    final_forced = family_measure(
        grid, g, tasks, POSITION_ADDRESSES, FORCED_CHOICES, sigma2
    )
    final_train = family_measure(
        grid, g, tasks, POSITION_ADDRESSES, train_choices, sigma2
    )
    final_oracle_eval = family_measure(
        grid, g, tasks, POSITION_ADDRESSES, final_oracle, sigma2
    )

    return {
        "grid": grid,
        "g": g,
        "result": {
            "mode": mode,
            "seed": seed,
            "epochs": epochs,
            "slow_proposals_per_epoch": slow_proposals,
            "total_attempted_slots": total_steps,
            "position_addresses": [a.__dict__ for a in POSITION_ADDRESSES],
            "forced_choices": choice_records(POSITION_ADDRESSES, FORCED_CHOICES),
            "aligned_choices": choice_records(POSITION_ADDRESSES, ALIGNED_CHOICES),
            "blank_oracle_choices": choice_records(POSITION_ADDRESSES, blank_oracle),
            "blank_forced": blank_forced,
            "blank_aligned": blank_aligned,
            "final_train": final_train,
            "final_forced": final_forced,
            "final_oracle": final_oracle_eval,
            "final_oracle_choices": choice_records(POSITION_ADDRESSES, final_oracle),
            "final_forced_is_oracle_fraction": float(np.mean(final_oracle == FORCED_CHOICES)),
            "oracle_address_switches": int(oracle_switches),
            "accepted": accepted,
            "proposed": proposed,
            "acceptance_fraction": accepted / max(proposed, 1),
            "material_sum_relative_error": abs(float(np.sum(g)) - total0) / total0,
            "changed_edges": int(np.count_nonzero(np.abs(g - 1.0) > 1e-12)),
            "trajectory": trajectory,
        },
    }


def summarize(runs):
    rows = [x["result"] for x in runs]

    def arr(section, key):
        return np.asarray([float(r[section][key]) for r in rows])

    return {
        "seeds": [int(r["seed"]) for r in rows],
        "median_blank_forced_min_ratio": float(np.median(arr("blank_forced", "min_ratio"))),
        "median_blank_forced_rank1_fraction": float(np.median(arr("blank_forced", "rank1_fraction"))),
        "median_blank_aligned_min_ratio": float(np.median(arr("blank_aligned", "min_ratio"))),
        "median_final_forced_min_ratio": float(np.median(arr("final_forced", "min_ratio"))),
        "min_final_forced_min_ratio": float(np.min(arr("final_forced", "min_ratio"))),
        "median_final_forced_rank1_fraction": float(np.median(arr("final_forced", "rank1_fraction"))),
        "median_final_forced_is_oracle_fraction": float(np.median(np.asarray([float(r["final_forced_is_oracle_fraction"]) for r in rows]))),
        "min_final_forced_is_oracle_fraction": float(np.min(np.asarray([float(r["final_forced_is_oracle_fraction"]) for r in rows]))),
        "median_final_oracle_min_ratio": float(np.median(arr("final_oracle", "min_ratio"))),
        "median_oracle_address_switches": float(np.median(np.asarray([float(r["oracle_address_switches"]) for r in rows]))),
        "median_acceptance_fraction": float(np.median(np.asarray([float(r["acceptance_fraction"]) for r in rows]))),
        "max_material_sum_relative_error": float(np.max(np.asarray([float(r["material_sum_relative_error"]) for r in rows]))),
    }


def classify(summary):
    s = summary["forced_consequence"]
    moved_all = bool(s["median_final_forced_is_oracle_fraction"] >= 1.0 - 1e-12)
    moved_any = bool(s["median_final_forced_is_oracle_fraction"] > 1e-12)
    rank1_all = bool(s["median_final_forced_rank1_fraction"] >= 1.0 - 1e-12)
    useful = bool(s["median_final_forced_min_ratio"] > 1.0)

    if moved_all and rank1_all and useful:
        verdict = "material_moves_all_forced_address_optima"
    elif moved_any and useful:
        verdict = "partial_address_optimum_movement"
    elif useful:
        verdict = "forced_routes_improve_but_coordinate_chart_does_not_move"
    else:
        verdict = "forced_relations_not_learned"

    return {
        "success_requires": "forced address is exact audit oracle for all 3 tasks in the median run, all 3 forced tasks rank-1, and worst forced ratio > 1",
        "moved_all_forced_optima": moved_all,
        "moved_any_forced_optimum": moved_any,
        "all_forced_tasks_rank1": rank1_all,
        "forced_worst_ratio_above_one": useful,
        "verdict": verdict,
    }


def run_all(out_dir: Path, seeds: int = 3, epochs: int = 60, slow_proposals: int = 40):
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["forced_consequence", "aligned_consequence", "forced_random", "forced_no_write"]
    runs = {
        mode: [
            run_mode(mode, seed=s, epochs=epochs, slow_proposals=slow_proposals)
            for s in range(seeds)
        ]
        for mode in modes
    }
    summary = {mode: summarize(rs) for mode, rs in runs.items()}
    interpretation = classify(summary)

    best = max(
        runs["forced_consequence"],
        key=lambda x: float(x["result"]["final_forced"]["min_ratio"]),
    )
    write_svg(out_dir / "gate8_forced_topology.svg", best["grid"], best["g"])

    receipt = {
        "gate": 8,
        "question": "Can conservative material learning make deliberately wrong spatial addresses become the best addresses, or does blank geometry fix the coordinate chart?",
        "preregistered_forced_map": "cyclic permutation: upper target <- center source; center target <- lower source; lower target <- upper source, all at omega=1.35",
        "learning_rule": "fixed-address local conservative edge dither; keep only if worst-task scalar consequence improves",
        "audit": "after every epoch, exact scan of all 3 spatial addresses; audit never enters the learning rule",
        "controls": {
            "aligned_consequence": "same learning rule at the blank-geometry aligned addresses",
            "forced_random": f"forced addresses with random acceptance probability {RANDOM_ACCEPT_PROBABILITY}",
            "forced_no_write": "forced addresses, no persistent material change",
        },
        "preregistered_success_rule": "strong success requires all 3 forced addresses to become the exact audit oracles in the median run, all forced tasks rank-1, and forced worst-task ratio > 1",
        "summary": summary,
        "interpretation": interpretation,
        "runs": {mode: [x["result"] for x in rs] for mode, rs in runs.items()},
        "claim_boundary": "This is a spatial-address rewrite test in one engineered reciprocal lattice. It does not establish biological remapping or general operator-manifold learning.",
    }
    (out_dir / "gate8_move_address_optimum.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--slow-proposals", type=int, default=40)
    args = p.parse_args()
    receipt = run_all(
        Path(args.out), seeds=args.seeds, epochs=args.epochs, slow_proposals=args.slow_proposals
    )
    print(json.dumps({"summary": receipt["summary"], "interpretation": receipt["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
