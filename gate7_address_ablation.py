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
    task_measure,
    task_port_indices,
)
from self_carving import make_grid, write_svg


MATCH_FRACTION = 0.90


def address_spaces() -> dict[str, list[Address]]:
    """Pre-registered Gate-7 ablations."""
    return {
        "full": addresses(),
        "position_only": [Address(source_dr=dr, omega=1.35) for dr in (-1, 0, 1)],
        "frequency_only": [Address(source_dr=0, omega=omega) for omega in (1.35, 1.55, 1.75)],
    }


def common_sigma2(grid, tasks: list[int], floor_fraction: float) -> float:
    """Use the full blank-space observer floor for every ablation."""
    return baseline_floor(grid, tasks, address_spaces()["full"], floor_fraction)


def learned_choices(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out: list[int] = []
    for q in range(values.shape[0]):
        seen = np.flatnonzero(counts[q] > 0)
        if len(seen) == 0:
            out.append(0)
        else:
            out.append(int(seen[np.argmax(values[q, seen])]))
    return np.asarray(out, dtype=int)


def run_space(
    space_name: str,
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
    spaces = address_spaces()
    if space_name not in spaces:
        raise ValueError(space_name)

    grid = make_grid(n)
    addr = spaces[space_name]
    tasks = task_port_indices(grid)
    g = np.ones(len(grid.edges), dtype=float)
    total0 = float(np.sum(g))
    sigma2 = common_sigma2(grid, tasks, floor_fraction)

    baseline_matrix = exact_matrix(grid, g, tasks, addr, sigma2)
    baseline_oracle = oracle_choices(baseline_matrix)
    baseline_eval = family_measure(grid, g, tasks, addr, baseline_oracle, sigma2)

    values = np.zeros((len(tasks), len(addr)), dtype=float)
    counts = np.zeros_like(values, dtype=int)
    rng = np.random.default_rng(50000 + 1000 * seed + len(addr))
    proposal_rng = np.random.default_rng(60000 + seed)
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
    last_selected = np.zeros(len(tasks), dtype=int)
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
            values[q, ai] = (1.0 - ema) * values[q, ai] + ema * float(m["utility"])
        counts[q, ai] += 1

    for epoch in range(epochs):
        for q in range(len(tasks)):
            for _ in range(probes_per_task):
                probe_one(q)

        choices = learned_choices(values, counts)
        selector_switches += int(np.count_nonzero(choices != last_selected))
        last_selected = choices.copy()

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

    final_selected = learned_choices(values, counts)
    final_matrix = exact_matrix(grid, g, tasks, addr, sigma2)
    final_oracle = oracle_choices(final_matrix)
    final_selected_eval = family_measure(grid, g, tasks, addr, final_selected, sigma2)
    final_oracle_eval = family_measure(grid, g, tasks, addr, final_oracle, sigma2)

    result = {
        "space": space_name,
        "seed": seed,
        "epochs": epochs,
        "probes_per_task": probes_per_task,
        "slow_proposals_per_epoch": slow_proposals,
        "address_count": len(addr),
        "addresses": [a.__dict__ for a in addr],
        "sigma2_common_full_space": sigma2,
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
        "baseline_oracle": baseline_eval,
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
        "address_count": int(rows[0]["address_count"]),
        "sigma2_common_full_space": float(rows[0]["sigma2_common_full_space"]),
        "median_baseline_oracle_min_ratio": float(np.median(vals("baseline_oracle", "min_ratio"))),
        "median_baseline_oracle_rank1_fraction": float(np.median(vals("baseline_oracle", "rank1_fraction"))),
        "median_final_selected_min_ratio": float(np.median(vals("final_selected", "min_ratio"))),
        "min_final_selected_min_ratio": float(np.min(vals("final_selected", "min_ratio"))),
        "median_final_selected_rank1_fraction": float(np.median(vals("final_selected", "rank1_fraction"))),
        "median_final_oracle_min_ratio": float(np.median(vals("final_oracle", "min_ratio"))),
        "median_selector_oracle_exact_fraction": float(np.median(np.asarray([float(r["selector_oracle_exact_fraction"]) for r in rows]))),
        "median_distinct_final_selected_addresses": float(np.median(np.asarray([float(r["distinct_final_selected_addresses"]) for r in rows]))),
        "median_oracle_address_switches": float(np.median(np.asarray([float(r["oracle_address_switches"]) for r in rows]))),
        "median_acceptance_fraction": float(np.median(np.asarray([float(r["acceptance_fraction"]) for r in rows]))),
        "max_material_sum_relative_error": float(np.max(np.asarray([float(r["material_sum_relative_error"]) for r in rows]))),
    }


def classify(summary: dict[str, dict]) -> dict[str, object]:
    full = summary["full"]
    position = summary["position_only"]
    frequency = summary["frequency_only"]
    p_ratio = float(position["median_final_selected_min_ratio"] / max(full["median_final_selected_min_ratio"], 1e-30))
    f_ratio = float(frequency["median_final_selected_min_ratio"] / max(full["median_final_selected_min_ratio"], 1e-30))
    p_rank = bool(position["median_final_selected_rank1_fraction"] >= full["median_final_selected_rank1_fraction"] - 1e-12)
    f_rank = bool(frequency["median_final_selected_rank1_fraction"] >= full["median_final_selected_rank1_fraction"] - 1e-12)
    p_match = bool(p_rank and p_ratio >= MATCH_FRACTION)
    f_match = bool(f_rank and f_ratio >= MATCH_FRACTION)
    if p_match and f_match:
        verdict = "redundant_coordinates_joint_not_needed"
    elif p_match:
        verdict = "position_dominated_joint_not_needed"
    elif f_match:
        verdict = "frequency_dominated_joint_not_needed"
    else:
        verdict = "joint_space_earns_ablation_advantage"
    return {
        "match_fraction_preregistered": MATCH_FRACTION,
        "position_over_full_final_min_ratio": p_ratio,
        "frequency_over_full_final_min_ratio": f_ratio,
        "position_matches_full_rank1": p_rank,
        "frequency_matches_full_rank1": f_rank,
        "position_matches_full": p_match,
        "frequency_matches_full": f_match,
        "verdict": verdict,
    }


def run_all(out_dir: Path, seeds: int = 3, epochs: int = 30, slow_proposals: int = 24):
    out_dir.mkdir(parents=True, exist_ok=True)
    spaces = list(address_spaces())
    runs = {
        name: [
            run_space(name, seed=s, epochs=epochs, slow_proposals=slow_proposals)
            for s in range(seeds)
        ]
        for name in spaces
    }
    summary = {name: summarize(rs) for name, rs in runs.items()}
    interpretation = classify(summary)

    best = max(
        runs["position_only"],
        key=lambda x: float(x["result"]["final_selected"]["min_ratio"]),
    )
    write_svg(out_dir / "gate7_position_only_topology.svg", best["grid"], best["g"])

    receipt = {
        "gate": 7,
        "question": "Did Gate 6 actually need joint source-position and frequency addressing, or can one address dimension explain the result?",
        "preregistered_spaces": {
            "full": "3 source rows x 3 frequencies",
            "position_only": "3 source rows at fixed omega=1.35, the frequency selected by every Gate-6 blank/final oracle task",
            "frequency_only": "center source row x 3 frequencies",
        },
        "fairness": "All three ablations use the same bounded observer floor computed once from the full blank-material Gate-6 address space and the same slow material proposal schedule per seed.",
        "preregistered_match_rule": "An ablation matches FULL only if it reaches the same median rank-1 fraction and at least 90% of FULL's median final worst-task ratio.",
        "preregistered_outcomes": {
            "both_match": "address coordinates are redundant here; joint position-frequency interpretation is not earned",
            "position_only_matches": "position-dominated; kill joint position-frequency interpretation",
            "frequency_only_matches": "frequency-dominated; kill joint position-frequency interpretation",
            "neither_matches": "the joint space earns an ablation advantage in this protocol",
        },
        "summary": summary,
        "interpretation_metrics": interpretation,
        "runs": {name: [x["result"] for x in rs] for name, rs in runs.items()},
        "claim_boundary": "This is an ablation of the existing engineered address space. It does not test biological dendritic frequency coding or prove that one coordinate is generally more useful than another.",
    }
    (out_dir / "gate7_address_ablation.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--slow-proposals", type=int, default=24)
    args = p.parse_args()
    receipt = run_all(Path(args.out), seeds=args.seeds, epochs=args.epochs, slow_proposals=args.slow_proposals)
    print(json.dumps({"summary": receipt["summary"], "interpretation_metrics": receipt["interpretation_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
