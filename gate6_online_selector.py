from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate2_local_dither import local_edge_pairs, transfer_local_material
from gate6_learned_addressing import (
    addresses,
    baseline_floor,
    choice_records,
    exact_matrix,
    family_measure,
    oracle_choices,
    task_measure,
    task_port_indices,
)
from self_carving import make_grid, write_svg


def default_address_index(addr) -> int:
    for i, a in enumerate(addr):
        if a.source_dr == 0 and abs(a.omega - 1.55) < 1e-12:
            return i
    raise ValueError("default address missing")


def learned_choices(values: np.ndarray, counts: np.ndarray, default: int) -> np.ndarray:
    out = []
    for q in range(values.shape[0]):
        seen = np.flatnonzero(counts[q] > 0)
        if len(seen) == 0:
            out.append(default)
        else:
            out.append(int(seen[np.argmax(values[q, seen])]))
    return np.asarray(out, dtype=int)


def run_mode(
    mode: str,
    seed: int,
    n: int = 11,
    epochs: int = 30,
    probes_per_task: int = 3,
    slow_proposals: int = 24,
    delta: float = 0.055,
    floor_fraction: float = 0.01,
    epsilon: float = 0.30,
    ema: float = 0.45,
):
    """Learn task -> physical address online while one shared material also changes.

    Unlike Gate 6A, the fast selector is not initialized from an exact address scan.
    Unseen addresses are deliberately sampled first; after coverage, scalar
    epsilon-greedy probes keep refreshing the task/address values. Exact scans are
    audit-only and never enter the selector or the structural acceptance rule.
    """
    allowed = {"coadaptive_online", "stale_default_material", "selector_only"}
    if mode not in allowed:
        raise ValueError(mode)

    grid = make_grid(n)
    addr = addresses()
    tasks = task_port_indices(grid)
    default = default_address_index(addr)
    g = np.ones(len(grid.edges), dtype=float)
    total0 = float(np.sum(g))
    sigma2 = baseline_floor(grid, tasks, addr, floor_fraction)

    # Audit only. The learner starts with no address utilities.
    baseline_matrix = exact_matrix(grid, g, tasks, addr, sigma2)
    baseline_oracle = oracle_choices(baseline_matrix)
    baseline_oracle_eval = family_measure(
        grid, g, tasks, addr, baseline_oracle, sigma2
    )

    values = np.zeros((len(tasks), len(addr)), dtype=float)
    counts = np.zeros_like(values, dtype=int)
    rng = np.random.default_rng(30000 + seed)
    proposal_rng = np.random.default_rng(40000 + seed)
    pairs = local_edge_pairs(grid)
    total_steps = epochs * slow_proposals
    pair_schedule = proposal_rng.integers(len(pairs), size=total_steps)
    orientation_schedule = np.where(
        proposal_rng.random(total_steps) < 0.5, 1, -1
    ).astype(int)

    accepted = 0
    proposed = 0
    cursor = 0
    selector_switches = 0
    oracle_switches = 0
    last_selected = np.full(len(tasks), default, dtype=int)
    last_oracle = baseline_oracle.copy()
    trajectory = []

    def probe_one(q: int) -> None:
        unseen = np.flatnonzero(counts[q] == 0)
        if len(unseen) > 0:
            ai = int(unseen[int(rng.integers(len(unseen)))])
        elif rng.random() < epsilon:
            ai = int(rng.integers(len(addr)))
        else:
            ai = int(np.argmax(values[q]))
        m = task_measure(grid, g, tasks[q], addr[ai], sigma2)
        if counts[q, ai] == 0:
            values[q, ai] = float(m["utility"])
        else:
            values[q, ai] = (
                (1.0 - ema) * values[q, ai] + ema * float(m["utility"])
            )
        counts[q, ai] += 1

    for epoch in range(epochs):
        if mode != "stale_default_material":
            for q in range(len(tasks)):
                for _ in range(probes_per_task):
                    probe_one(q)
            choices = learned_choices(values, counts, default)
        else:
            choices = np.full(len(tasks), default, dtype=int)

        selector_switches += int(np.count_nonzero(choices != last_selected))
        last_selected = choices.copy()

        if mode != "selector_only":
            current = family_measure(grid, g, tasks, addr, choices, sigma2)
            for _ in range(slow_proposals):
                ea, eb = map(int, pairs[int(pair_schedule[cursor])])
                orientation = int(orientation_schedule[cursor])
                cursor += 1
                cand = transfer_local_material(g, ea, eb, orientation, delta=delta)
                if cand is None:
                    continue
                proposed += 1
                cm = family_measure(grid, cand, tasks, addr, choices, sigma2)
                if float(cm["worst_utility"]) > float(current["worst_utility"]) + 1e-14:
                    g = cand
                    current = cm
                    accepted += 1
        else:
            cursor += slow_proposals

        audit = exact_matrix(grid, g, tasks, addr, sigma2)
        oracle = oracle_choices(audit)
        oracle_switches += int(np.count_nonzero(oracle != last_oracle))
        last_oracle = oracle.copy()
        selected_eval = family_measure(grid, g, tasks, addr, choices, sigma2)
        oracle_eval = family_measure(grid, g, tasks, addr, oracle, sigma2)
        trajectory.append(
            {
                "epoch": epoch + 1,
                "observed_address_fraction": float(np.mean(counts > 0)),
                "selected_choices": choice_records(addr, choices),
                "oracle_choices": choice_records(addr, oracle),
                "selected_min_ratio": float(selected_eval["min_ratio"]),
                "selected_rank1_fraction": float(selected_eval["rank1_fraction"]),
                "oracle_min_ratio": float(oracle_eval["min_ratio"]),
            }
        )

    if mode == "stale_default_material":
        final_selected = np.full(len(tasks), default, dtype=int)
    else:
        final_selected = learned_choices(values, counts, default)
    final_matrix = exact_matrix(grid, g, tasks, addr, sigma2)
    final_oracle = oracle_choices(final_matrix)
    final_selected_eval = family_measure(
        grid, g, tasks, addr, final_selected, sigma2
    )
    final_oracle_eval = family_measure(
        grid, g, tasks, addr, final_oracle, sigma2
    )

    result = {
        "mode": mode,
        "seed": seed,
        "epochs": epochs,
        "probes_per_task": probes_per_task,
        "slow_proposals_per_epoch": slow_proposals,
        "default_address": choice_records(addr, np.asarray([default]))[0],
        "baseline_oracle_choices": choice_records(addr, baseline_oracle),
        "final_selected_choices": choice_records(addr, final_selected),
        "final_oracle_choices": choice_records(addr, final_oracle),
        "observed_address_fraction": float(np.mean(counts > 0)),
        "selector_switches": int(selector_switches),
        "oracle_address_switches": int(oracle_switches),
        "selector_oracle_exact_fraction": float(np.mean(final_selected == final_oracle)),
        "distinct_final_selected_addresses": int(len(set(map(int, final_selected)))),
        "accepted": accepted,
        "proposed": proposed,
        "acceptance_fraction": accepted / max(proposed, 1),
        "material_sum_relative_error": abs(float(np.sum(g)) - total0) / total0,
        "changed_edges": int(np.count_nonzero(np.abs(g - 1.0) > 1e-12)),
        "baseline_oracle": baseline_oracle_eval,
        "final_selected": final_selected_eval,
        "final_oracle": final_oracle_eval,
        "trajectory": trajectory,
    }
    return {"grid": grid, "g": g, "result": result}


def summarize(runs):
    rows = [x["result"] for x in runs]

    def vals(section, key):
        return np.asarray([float(r[section][key]) for r in rows])

    return {
        "seeds": [int(r["seed"]) for r in rows],
        "median_baseline_oracle_min_ratio": float(np.median(vals("baseline_oracle", "min_ratio"))),
        "median_final_selected_min_ratio": float(np.median(vals("final_selected", "min_ratio"))),
        "min_final_selected_min_ratio": float(np.min(vals("final_selected", "min_ratio"))),
        "median_final_selected_rank1_fraction": float(np.median(vals("final_selected", "rank1_fraction"))),
        "median_final_oracle_min_ratio": float(np.median(vals("final_oracle", "min_ratio"))),
        "median_selector_oracle_exact_fraction": float(np.median(np.asarray([float(r["selector_oracle_exact_fraction"]) for r in rows]))),
        "median_distinct_final_selected_addresses": float(np.median(np.asarray([float(r["distinct_final_selected_addresses"]) for r in rows]))),
        "median_observed_address_fraction": float(np.median(np.asarray([float(r["observed_address_fraction"]) for r in rows]))),
        "median_selector_switches": float(np.median(np.asarray([float(r["selector_switches"]) for r in rows]))),
        "median_oracle_address_switches": float(np.median(np.asarray([float(r["oracle_address_switches"]) for r in rows]))),
        "median_acceptance_fraction": float(np.median(np.asarray([float(r["acceptance_fraction"]) for r in rows]))),
        "max_material_sum_relative_error": float(np.max(np.asarray([float(r["material_sum_relative_error"]) for r in rows]))),
    }


def run_all(out_dir: Path, seeds: int = 3, epochs: int = 30, slow_proposals: int = 24):
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["coadaptive_online", "stale_default_material", "selector_only"]
    runs = {
        mode: [
            run_mode(mode, seed=s, epochs=epochs, slow_proposals=slow_proposals)
            for s in range(seeds)
        ]
        for mode in modes
    }
    summary = {mode: summarize(rs) for mode, rs in runs.items()}
    best = max(
        runs["coadaptive_online"],
        key=lambda x: float(x["result"]["final_selected"]["min_ratio"]),
    )
    write_svg(out_dir / "gate6_online_selector_topology.svg", best["grid"], best["g"])
    receipt = {
        "gate": "6C",
        "question": "Can task-to-(source,frequency) addressing be learned from scratch by bounded scalar probes while the same substrate is slowly rewritten?",
        "initial_selector_information": "none; exact address scans are audit-only",
        "fast_rule": "unseen-first then epsilon-greedy scalar utility tracking",
        "slow_rule": "local conservative edge dithers retained by worst-task consequence at currently selected addresses",
        "controls": {
            "stale_default_material": "fixed center/omega=1.55 address for every task",
            "selector_only": "online address learning with material frozen",
        },
        "summary": summary,
        "runs": {mode: [x["result"] for x in rs] for mode, rs in runs.items()},
        "claim_boundary": "This uses a tiny engineered digital selector and externally named receiver tasks. It establishes only whether bounded scalar address search can coexist with slow material learning in this deterministic wave toy.",
    }
    (out_dir / "gate6_online_selector.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--slow-proposals", type=int, default=24)
    args = p.parse_args()
    receipt = run_all(Path(args.out), seeds=args.seeds, epochs=args.epochs, slow_proposals=args.slow_proposals)
    print(json.dumps(receipt["summary"], indent=2))


if __name__ == "__main__":
    main()
