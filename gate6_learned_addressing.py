from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gate2_local_dither import local_edge_pairs, transfer_local_material
from gate3_multiport_dither import right_ports
from self_carving import make_grid, node, solve_field, write_svg


@dataclass(frozen=True)
class Address:
    source_dr: int
    omega: float


def addresses() -> list[Address]:
    return [
        Address(source_dr=dr, omega=omega)
        for dr in (-1, 0, 1)
        for omega in (1.35, 1.55, 1.75)
    ]


def task_port_indices(grid) -> list[int]:
    ports = right_ports(grid)
    if len(ports) < 5:
        raise ValueError("Gate 6 requires at least five right-side ports")
    mid = len(ports) // 2
    return [mid - 2, mid, mid + 2]


def address_source(grid, address: Address) -> int:
    row = grid.source // grid.n + address.source_dr
    col = grid.source % grid.n
    return node(grid.n, row, col)


def task_measure(
    grid,
    g: np.ndarray,
    target_port_index: int,
    address: Address,
    sigma2: float,
) -> dict[str, float | int]:
    ports = right_ports(grid)
    source = address_source(grid, address)
    u = solve_field(grid, g, source, omega=address.omega)
    p = np.abs(u[ports]) ** 2
    target_power = float(p[target_port_index])
    other_indices = [i for i in range(len(ports)) if i != target_port_index]
    other_values = p[other_indices]
    other_local = int(np.argmax(other_values))
    strongest_index = int(other_indices[other_local])
    strongest_power = float(other_values[other_local])
    rank = int(np.where(np.argsort(-p) == target_port_index)[0][0]) + 1
    utility = math.log(target_power + sigma2) - math.log(strongest_power + sigma2)
    return {
        "utility": float(utility),
        "target_power": target_power,
        "strongest_other_power": strongest_power,
        "target_over_strongest_other": target_power / max(strongest_power, 1e-30),
        "target_rank": rank,
        "strongest_other_port_index": strongest_index,
    }


def baseline_floor(grid, task_indices: list[int], addr: list[Address], floor_fraction: float) -> float:
    g = np.ones(len(grid.edges), dtype=float)
    powers = []
    for ti in task_indices:
        for a in addr:
            m = task_measure(grid, g, ti, a, sigma2=0.0)
            powers.append(float(m["target_power"]))
    return floor_fraction * float(np.median(np.asarray(powers)))


def exact_matrix(grid, g: np.ndarray, task_indices: list[int], addr: list[Address], sigma2: float):
    matrix = []
    for ti in task_indices:
        matrix.append([task_measure(grid, g, ti, a, sigma2) for a in addr])
    return matrix


def oracle_choices(matrix) -> np.ndarray:
    return np.asarray(
        [int(np.argmax([float(m["utility"]) for m in row])) for row in matrix],
        dtype=int,
    )


def family_measure(
    grid,
    g: np.ndarray,
    task_indices: list[int],
    addr: list[Address],
    choices: np.ndarray,
    sigma2: float,
):
    rows = [
        task_measure(grid, g, ti, addr[int(ai)], sigma2)
        for ti, ai in zip(task_indices, choices)
    ]
    utilities = np.asarray([float(r["utility"]) for r in rows])
    ratios = np.asarray([float(r["target_over_strongest_other"]) for r in rows])
    ranks = np.asarray([int(r["target_rank"]) for r in rows])
    return {
        "rows": rows,
        "worst_utility": float(np.min(utilities)),
        "median_utility": float(np.median(utilities)),
        "min_ratio": float(np.min(ratios)),
        "median_ratio": float(np.median(ratios)),
        "rank1_fraction": float(np.mean(ranks == 1)),
    }


def choice_records(addr: list[Address], choices: np.ndarray) -> list[dict[str, float | int]]:
    return [
        {
            "address_index": int(i),
            "source_dr": int(addr[int(i)].source_dr),
            "omega": float(addr[int(i)].omega),
        }
        for i in choices
    ]


def init_selector(matrix) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        [[float(m["utility"]) for m in row] for row in matrix], dtype=float
    )
    counts = np.ones_like(values, dtype=int)
    return values, counts


def selector_choices(values: np.ndarray) -> np.ndarray:
    return np.argmax(values, axis=1).astype(int)


def run_mode(
    mode: str,
    seed: int,
    n: int = 11,
    epochs: int = 20,
    probes_per_task: int = 5,
    slow_proposals: int = 24,
    delta: float = 0.055,
    floor_fraction: float = 0.01,
    epsilon: float = 0.25,
    ema: float = 0.35,
):
    allowed = {"coadaptive", "stale_selector_material", "selector_only", "random_address_material"}
    if mode not in allowed:
        raise ValueError(mode)

    grid = make_grid(n)
    addr = addresses()
    tasks = task_port_indices(grid)
    g = np.ones(len(grid.edges), dtype=float)
    total0 = float(np.sum(g))
    sigma2 = baseline_floor(grid, tasks, addr, floor_fraction)
    baseline_matrix = exact_matrix(grid, g, tasks, addr, sigma2)
    values, counts = init_selector(baseline_matrix)
    initial_choices = selector_choices(values)
    frozen_choices = initial_choices.copy()

    rng = np.random.default_rng(seed)
    pairs = local_edge_pairs(grid)
    accepted = 0
    proposed = 0
    selector_switches = 0
    oracle_switches = 0
    last_selector = initial_choices.copy()
    last_oracle = oracle_choices(baseline_matrix)
    trajectory = []

    def probe_selector():
        nonlocal selector_switches, last_selector
        if mode in {"stale_selector_material", "random_address_material"}:
            return
        for q, ti in enumerate(tasks):
            for _ in range(probes_per_task):
                if rng.random() < epsilon:
                    ai = int(rng.integers(len(addr)))
                else:
                    ai = int(np.argmax(values[q]))
                m = task_measure(grid, g, ti, addr[ai], sigma2)
                values[q, ai] = (1.0 - ema) * values[q, ai] + ema * float(m["utility"])
                counts[q, ai] += 1
        now = selector_choices(values)
        selector_switches += int(np.count_nonzero(now != last_selector))
        last_selector = now.copy()

    for epoch in range(epochs):
        probe_selector()

        if mode == "random_address_material":
            choices = rng.integers(len(addr), size=len(tasks), dtype=int)
        elif mode == "stale_selector_material":
            choices = frozen_choices.copy()
        else:
            choices = selector_choices(values)

        if mode != "selector_only":
            current = family_measure(grid, g, tasks, addr, choices, sigma2)
            for _ in range(slow_proposals):
                pi = int(rng.integers(len(pairs)))
                ea, eb = map(int, pairs[pi])
                orientation = 1 if rng.random() < 0.5 else -1
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
        selected = (
            rng.integers(len(addr), size=len(tasks), dtype=int)
            if mode == "random_address_material"
            else frozen_choices.copy()
            if mode == "stale_selector_material"
            else selector_choices(values)
        )
        selected_eval = family_measure(grid, g, tasks, addr, selected, sigma2)
        oracle_eval = family_measure(grid, g, tasks, addr, oracle, sigma2)
        trajectory.append(
            {
                "epoch": epoch + 1,
                "selected_choices": choice_records(addr, selected),
                "oracle_choices": choice_records(addr, oracle),
                "selected_min_ratio": float(selected_eval["min_ratio"]),
                "selected_rank1_fraction": float(selected_eval["rank1_fraction"]),
                "oracle_min_ratio": float(oracle_eval["min_ratio"]),
                "oracle_rank1_fraction": float(oracle_eval["rank1_fraction"]),
            }
        )

    final_matrix = exact_matrix(grid, g, tasks, addr, sigma2)
    final_oracle = oracle_choices(final_matrix)
    if mode == "stale_selector_material":
        final_selected = frozen_choices.copy()
    elif mode == "random_address_material":
        final_selected = rng.integers(len(addr), size=len(tasks), dtype=int)
    else:
        final_selected = selector_choices(values)

    baseline_oracle = oracle_choices(baseline_matrix)
    baseline_oracle_eval = family_measure(grid, np.ones_like(g), tasks, addr, baseline_oracle, sigma2)
    final_selected_eval = family_measure(grid, g, tasks, addr, final_selected, sigma2)
    final_oracle_eval = family_measure(grid, g, tasks, addr, final_oracle, sigma2)

    result = {
        "mode": mode,
        "seed": seed,
        "epochs": epochs,
        "probes_per_task": probes_per_task,
        "slow_proposals_per_epoch": slow_proposals,
        "task_port_indices": [int(x) for x in tasks],
        "addresses": [a.__dict__ for a in addr],
        "sigma2": sigma2,
        "accepted": accepted,
        "proposed": proposed,
        "acceptance_fraction": accepted / max(proposed, 1),
        "material_sum_relative_error": abs(float(np.sum(g)) - total0) / total0,
        "changed_edges": int(np.count_nonzero(np.abs(g - 1.0) > 1e-12)),
        "selector_switches": int(selector_switches),
        "oracle_address_switches": int(oracle_switches),
        "initial_choices": choice_records(addr, initial_choices),
        "final_selected_choices": choice_records(addr, final_selected),
        "final_oracle_choices": choice_records(addr, final_oracle),
        "distinct_final_selected_addresses": int(len(set(map(int, final_selected)))),
        "selector_oracle_exact_fraction": float(np.mean(final_selected == final_oracle)),
        "baseline_oracle": baseline_oracle_eval,
        "final_selected": final_selected_eval,
        "final_oracle": final_oracle_eval,
        "trajectory": trajectory,
    }
    return {"grid": grid, "g": g, "result": result}


def summarize(runs):
    rows = [x["result"] for x in runs]
    def arr(section, key):
        return np.asarray([float(r[section][key]) for r in rows])
    return {
        "seeds": [int(r["seed"]) for r in rows],
        "median_baseline_oracle_min_ratio": float(np.median(arr("baseline_oracle", "min_ratio"))),
        "median_final_selected_min_ratio": float(np.median(arr("final_selected", "min_ratio"))),
        "min_final_selected_min_ratio": float(np.min(arr("final_selected", "min_ratio"))),
        "median_final_selected_ratio": float(np.median(arr("final_selected", "median_ratio"))),
        "median_final_selected_rank1_fraction": float(np.median(arr("final_selected", "rank1_fraction"))),
        "median_final_oracle_min_ratio": float(np.median(arr("final_oracle", "min_ratio"))),
        "median_final_oracle_rank1_fraction": float(np.median(arr("final_oracle", "rank1_fraction"))),
        "median_selector_oracle_exact_fraction": float(np.median(np.asarray([float(r["selector_oracle_exact_fraction"]) for r in rows]))),
        "median_distinct_final_selected_addresses": float(np.median(np.asarray([float(r["distinct_final_selected_addresses"]) for r in rows]))),
        "median_selector_switches": float(np.median(np.asarray([float(r["selector_switches"]) for r in rows]))),
        "median_oracle_address_switches": float(np.median(np.asarray([float(r["oracle_address_switches"]) for r in rows]))),
        "median_acceptance_fraction": float(np.median(np.asarray([float(r["acceptance_fraction"]) for r in rows]))),
        "max_material_sum_relative_error": float(np.max(np.asarray([float(r["material_sum_relative_error"]) for r in rows]))),
    }


def run_all(out_dir: Path, seeds: int = 3, epochs: int = 20, slow_proposals: int = 24):
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["coadaptive", "stale_selector_material", "selector_only", "random_address_material"]
    runs = {
        mode: [
            run_mode(mode, seed=s, epochs=epochs, slow_proposals=slow_proposals)
            for s in range(seeds)
        ]
        for mode in modes
    }
    summary = {mode: summarize(runs[mode]) for mode in modes}
    best = max(
        runs["coadaptive"],
        key=lambda x: float(x["result"]["final_selected"]["min_ratio"]),
    )
    write_svg(out_dir / "gate6_learned_addressing_topology.svg", best["grid"], best["g"])
    receipt = {
        "gate": 6,
        "question": "Can a fast learned (source-position, frequency) address policy co-adapt with one slowly self-carved resonant substrate while the best addresses move?",
        "task": "three task labels request three distinct right-side receiver ports",
        "address_space": "3 nearby source landing rows x 3 carrier frequencies = 9 addresses",
        "fast_rule": "per-task exponentially weighted epsilon-greedy scalar utility estimates",
        "slow_rule": "local material-conserving edge transfer retained only if one worst-task scalar improves at the currently selected addresses",
        "structural_information_hidden": "edge gradient, strongest competitor identity, oracle address matrix",
        "controls": {
            "stale_selector_material": "freeze the blank-medium best address for each task while material changes",
            "selector_only": "learn addresses while material is frozen",
            "random_address_material": "change material under randomly chosen task addresses",
        },
        "summary": summary,
        "runs": {mode: [x["result"] for x in rs] for mode, rs in runs.items()},
        "claim_boundary": "Gate 6 uses a small digital task-to-address value table and an externally defined receiver utility. It tests co-adaptation and moving operator addresses; it is not autonomous semantics, a proof of hardware advantage, or a claim that brains use this exact selector.",
    }
    (out_dir / "gate6_learned_addressing.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--slow-proposals", type=int, default=24)
    args = p.parse_args()
    receipt = run_all(Path(args.out), seeds=args.seeds, epochs=args.epochs, slow_proposals=args.slow_proposals)
    print(json.dumps(receipt["summary"], indent=2))


if __name__ == "__main__":
    main()
