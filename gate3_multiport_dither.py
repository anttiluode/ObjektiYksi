from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from gate2_local_dither import (
    local_edge_pairs,
    make_schedule,
    transfer_local_material,
)
from self_carving import make_grid, query_metrics, solve_field, write_svg


def right_ports(grid) -> list[int]:
    col = grid.n - 2
    return [r * grid.n + col for r in range(1, grid.n - 1)]


def multiport_measure(grid, g: np.ndarray, sigma2: float) -> dict[str, float | int]:
    """One source-only observation summarized into one scalar consequence.

    The learning rule receives only `utility`. It is not told which competitor
    is currently strongest and receives no edgewise or spatial error signal.
    """
    u = solve_field(grid, g, grid.source)
    ports = right_ports(grid)
    power = np.abs(u[ports]) ** 2
    target_i = ports.index(grid.target)
    target_power = float(power[target_i])
    other = np.delete(power, target_i)
    strongest_other = float(np.max(other))
    strongest_other_local_index = int(np.argmax(other))
    other_port_indices = [i for i in range(len(ports)) if i != target_i]
    strongest_port_index = other_port_indices[strongest_other_local_index]
    order = np.argsort(-power)
    rank = int(np.where(order == target_i)[0][0]) + 1
    utility = math.log(target_power + sigma2) - math.log(strongest_other + sigma2)
    return {
        "utility": float(utility),
        "target_power": target_power,
        "strongest_other_power": strongest_other,
        "target_over_strongest_other": target_power / max(strongest_other, 1e-30),
        "target_rank": rank,
        "strongest_other_port_index": int(strongest_port_index),
    }


def two_port_utility(grid, g: np.ndarray, sigma2: float) -> float:
    u = solve_field(grid, g, grid.source)
    pt = float(abs(u[grid.target]) ** 2)
    pd = float(abs(u[grid.decoy]) ** 2)
    return math.log(pt + sigma2) - math.log(pd + sigma2)


def train_multiport(
    mode: str,
    seed: int,
    n: int = 11,
    attempts: int = 650,
    delta: float = 0.055,
    floor_fraction: float = 0.01,
) -> dict:
    """Local derivative-free structural search with a multiport scalar objective.

    Modes share the same proposal stream.
      true_multiport: keep iff target beats its strongest current competitor better.
      two_port_control: Gate-2 target-vs-mirror objective, evaluated on all ports.
      random_accept: ignore consequence and accept with p=0.5.
      inverted_multiport: keep iff the multiport objective gets worse.
      no_write: frozen material.
    """
    allowed = {
        "true_multiport",
        "two_port_control",
        "random_accept",
        "inverted_multiport",
        "no_write",
    }
    if mode not in allowed:
        raise ValueError(mode)

    grid = make_grid(n)
    g = np.ones(len(grid.edges), dtype=float)
    total0 = float(np.sum(g))
    pairs = local_edge_pairs(grid)
    schedule = make_schedule(len(pairs), attempts, seed)

    baseline0 = multiport_measure(grid, g, sigma2=0.0)
    sigma2 = float(floor_fraction * float(baseline0["target_power"]))
    current = multiport_measure(grid, g, sigma2=sigma2)
    current_two = two_port_utility(grid, g, sigma2=sigma2)
    accepted = 0
    proposed = 0
    switches = 0
    last_competitor = int(current["strongest_other_port_index"])
    checkpoints = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, attempts}
    trajectory: list[dict[str, float | int]] = [
        {"attempt": 0, **current}
    ]

    if mode != "no_write":
        for k in range(attempts):
            ea, eb = map(int, pairs[int(schedule.pair_index[k])])
            cand = transfer_local_material(
                g, ea, eb, int(schedule.orientation[k]), delta=delta
            )
            if cand is None:
                continue
            proposed += 1
            cm = multiport_measure(grid, cand, sigma2=sigma2)
            cand_two = two_port_utility(grid, cand, sigma2=sigma2)

            if mode == "true_multiport":
                take = float(cm["utility"]) > float(current["utility"]) + 1e-14
            elif mode == "two_port_control":
                take = cand_two > current_two + 1e-14
            elif mode == "inverted_multiport":
                take = float(cm["utility"]) < float(current["utility"]) - 1e-14
            else:
                take = bool(schedule.coin[k] < 0.5)

            if take:
                g = cand
                current = cm
                current_two = cand_two
                accepted += 1
                competitor = int(current["strongest_other_port_index"])
                if competitor != last_competitor:
                    switches += 1
                    last_competitor = competitor

            if (k + 1) in checkpoints:
                trajectory.append({"attempt": k + 1, **current})

    # Final metric is always the multiport one, irrespective of training mode.
    final = multiport_measure(grid, g, sigma2=sigma2)
    path = query_metrics(grid, g)
    result = {
        "mode": mode,
        "seed": seed,
        "n": n,
        "attempts": attempts,
        "delta": delta,
        "floor_fraction": floor_fraction,
        "sigma2": sigma2,
        "baseline": baseline0,
        "final": final,
        "utility_gain": float(final["utility"]) - float(
            multiport_measure(grid, np.ones_like(g), sigma2=sigma2)["utility"]
        ),
        "accepted": accepted,
        "proposed": proposed,
        "acceptance_fraction": accepted / max(proposed, 1),
        "strongest_competitor_switches_on_accepted_steps": switches,
        "material_sum_relative_error": abs(float(np.sum(g)) - total0) / total0,
        "changed_edges": int(np.count_nonzero(np.abs(g - 1.0) > 1e-12)),
        "material_min": float(np.min(g)),
        "material_max": float(np.max(g)),
        "target_power_vs_baseline": float(
            float(final["target_power"]) / max(float(baseline0["target_power"]), 1e-30)
        ),
        "target_vs_mirror_decoy": float(path["desired_over_decoy"]),
        "static_path": path,
        "trajectory": trajectory,
    }
    return {"grid": grid, "g": g, "result": result}


def summarize(runs: list[dict]) -> dict:
    rows = [r["result"] for r in runs]
    ratio = np.asarray(
        [float(r["final"]["target_over_strongest_other"]) for r in rows]
    )
    rank = np.asarray([float(r["final"]["target_rank"]) for r in rows])
    target_gain = np.asarray([float(r["target_power_vs_baseline"]) for r in rows])
    util_gain = np.asarray([float(r["utility_gain"]) for r in rows])
    accept = np.asarray([float(r["acceptance_fraction"]) for r in rows])
    switches = np.asarray(
        [float(r["strongest_competitor_switches_on_accepted_steps"]) for r in rows]
    )
    material_error = np.asarray([float(r["material_sum_relative_error"]) for r in rows])
    return {
        "seeds": [int(r["seed"]) for r in rows],
        "median_target_over_strongest_other": float(np.median(ratio)),
        "min_target_over_strongest_other": float(np.min(ratio)),
        "max_target_over_strongest_other": float(np.max(ratio)),
        "all_target_rank_one": bool(np.all(rank == 1)),
        "median_target_rank": float(np.median(rank)),
        "median_target_power_vs_baseline": float(np.median(target_gain)),
        "median_utility_gain": float(np.median(util_gain)),
        "median_acceptance_fraction": float(np.median(accept)),
        "median_competitor_switches": float(np.median(switches)),
        "max_material_sum_relative_error": float(np.max(material_error)),
    }


def run_all(out_dir: Path, seeds: int = 6) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = [
        "true_multiport",
        "two_port_control",
        "random_accept",
        "inverted_multiport",
        "no_write",
    ]
    runs = {mode: [train_multiport(mode, s) for s in range(seeds)] for mode in modes}
    summary = {mode: summarize(runs[mode]) for mode in modes}
    best = max(
        runs["true_multiport"],
        key=lambda r: float(r["result"]["final"]["target_over_strongest_other"]),
    )
    write_svg(out_dir / "gate3_multiport_topology.svg", best["grid"], best["g"])
    receipt = {
        "gate": 3,
        "question": (
            "Can local conservative dithers, selected only by one scalar comparing the target "
            "with the strongest competing output, carve a genuinely multiport-selective resonant topology?"
        ),
        "utility": "log(P_target + sigma^2) - log(max_{other ports} P + sigma^2)",
        "information_available_to_structural_rule": (
            "candidate local material move, then one scalar utility; no gradient, no reference field, "
            "no identity/location of the strongest competitor"
        ),
        "summary": summary,
        "runs": {mode: [x["result"] for x in runs[mode]] for mode in modes},
        "claim_boundary": (
            "This remains an externally specified routing objective in a toy resonant network. "
            "It tests multiport consequence-selected structural search, not autonomous goal formation "
            "or biological structural plasticity."
        ),
    }
    (out_dir / "gate3_multiport_dither.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--seeds", type=int, default=6)
    args = parser.parse_args()
    receipt = run_all(Path(args.out), seeds=args.seeds)
    print(json.dumps(receipt["summary"], indent=2))


if __name__ == "__main__":
    main()
