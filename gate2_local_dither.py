from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from gate1_route_or_resonance import spectral_metrics
from self_carving import make_grid, query_metrics, solve_field, write_svg


@dataclass(frozen=True)
class DitherSchedule:
    pair_index: np.ndarray
    orientation: np.ndarray
    coin: np.ndarray


def local_edge_pairs(grid) -> np.ndarray:
    """Pairs of edges that meet at one lattice node.

    A proposal moves coupling from one member of a pair to the other, so the
    structural perturbation is spatially local and conserves total material
    exactly without a global renormalization step.
    """
    incident: list[list[int]] = [[] for _ in range(grid.n * grid.n)]
    for k, (i, j) in enumerate(grid.edges):
        incident[int(i)].append(k)
        incident[int(j)].append(k)
    pairs: list[tuple[int, int]] = []
    for edges in incident:
        pairs.extend(combinations(edges, 2))
    return np.asarray(pairs, dtype=int)


def make_schedule(n_pairs: int, attempts: int, seed: int) -> DitherSchedule:
    rng = np.random.default_rng(seed)
    return DitherSchedule(
        pair_index=rng.integers(0, n_pairs, size=attempts),
        orientation=rng.integers(0, 2, size=attempts),
        coin=rng.random(attempts),
    )


def transfer_local_material(
    g: np.ndarray,
    edge_a: int,
    edge_b: int,
    orientation: int,
    delta: float,
    min_g: float = 0.08,
    max_g: float = 4.0,
) -> np.ndarray | None:
    """Move a fixed amount of material between two adjacent edges.

    Only two couplings change and sum(g) is preserved exactly (up to floating
    arithmetic). No derivative or reference field enters this operation.
    """
    up, down = (edge_a, edge_b) if orientation == 0 else (edge_b, edge_a)
    room_up = max_g - float(g[up])
    room_down = float(g[down]) - min_g
    amount = min(float(delta), room_up, room_down)
    if amount <= 1e-14:
        return None
    cand = g.copy()
    cand[up] += amount
    cand[down] -= amount
    return cand


def powers_and_utility(grid, g: np.ndarray, sigma2: float) -> tuple[float, float, float]:
    """Source-only delayed consequence.

    The observer returns one scalar after the perturbation. The noise floor
    prevents the old silence trick: if both outputs disappear, utility tends
    back toward zero instead of diverging as a raw ratio.
    """
    field = solve_field(grid, g, grid.source)
    pt = float(abs(field[grid.target]) ** 2)
    pd = float(abs(field[grid.decoy]) ** 2)
    utility = math.log(pt + sigma2) - math.log(pd + sigma2)
    return pt, pd, utility


def train_dither(
    mode: str,
    seed: int,
    n: int = 11,
    attempts: int = 420,
    delta: float = 0.055,
    floor_fraction: float = 0.01,
) -> dict:
    """Derivative-free local structural search.

    `true_consequence` receives only the delayed scalar utility after each
    physical dither and keeps a perturbation iff that scalar improves.
    `random_accept` sees the same proposal stream but ignores consequence.
    `inverted_consequence` deliberately keeps worse perturbations.
    `no_write` leaves material untouched.
    """
    if mode not in {"true_consequence", "random_accept", "inverted_consequence", "no_write"}:
        raise ValueError(mode)

    grid = make_grid(n)
    g = np.ones(len(grid.edges), dtype=float)
    total0 = float(np.sum(g))
    pairs = local_edge_pairs(grid)
    schedule = make_schedule(len(pairs), attempts, seed)

    base_pt, base_pd, _ = powers_and_utility(grid, g, sigma2=0.0)
    sigma2 = float(floor_fraction * base_pt)
    pt, pd, utility = powers_and_utility(grid, g, sigma2=sigma2)
    baseline_utility = utility
    accepted = 0
    proposed = 0
    trajectory: list[dict[str, float | int]] = [
        {"attempt": 0, "utility": utility, "desired_power": pt, "decoy_power": pd}
    ]
    checkpoints = {1, 2, 4, 8, 16, 32, 64, 128, 256, attempts}

    if mode != "no_write":
        for k in range(attempts):
            ea, eb = map(int, pairs[int(schedule.pair_index[k])])
            cand = transfer_local_material(
                g, ea, eb, int(schedule.orientation[k]), delta=delta
            )
            if cand is None:
                continue
            proposed += 1
            cpt, cpd, cu = powers_and_utility(grid, cand, sigma2=sigma2)

            if mode == "true_consequence":
                take = cu > utility + 1e-14
            elif mode == "inverted_consequence":
                take = cu < utility - 1e-14
            else:
                take = bool(schedule.coin[k] < 0.5)

            if take:
                g = cand
                pt, pd, utility = cpt, cpd, cu
                accepted += 1

            if (k + 1) in checkpoints:
                trajectory.append(
                    {
                        "attempt": k + 1,
                        "utility": utility,
                        "desired_power": pt,
                        "decoy_power": pd,
                    }
                )

    final_wave = spectral_metrics(grid, g, 1.55)
    final_paths = query_metrics(grid, g)
    changed = int(np.count_nonzero(np.abs(g - 1.0) > 1e-12))
    result = {
        "mode": mode,
        "seed": seed,
        "n": n,
        "attempts": attempts,
        "delta": delta,
        "floor_fraction": floor_fraction,
        "sigma2": sigma2,
        "baseline_utility": baseline_utility,
        "final_utility": utility,
        "utility_gain": utility - baseline_utility,
        "accepted": accepted,
        "proposed": proposed,
        "acceptance_fraction": accepted / max(proposed, 1),
        "material_sum_relative_error": abs(float(np.sum(g)) - total0) / total0,
        "changed_edges": changed,
        "material_min": float(np.min(g)),
        "material_max": float(np.max(g)),
        "final_wave": final_wave,
        "final_paths": final_paths,
        "target_power_vs_baseline": float(final_wave["desired_power"] / max(base_pt, 1e-30)),
        "trajectory": trajectory,
    }
    return {"grid": grid, "g": g, "result": result}


def summarize_runs(runs: list[dict]) -> dict:
    rows = [x["result"] for x in runs]
    def arr(key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in rows], dtype=float)

    ratios = np.asarray(
        [float(r["final_wave"]["desired_over_decoy"]) for r in rows], dtype=float
    )
    ranks = np.asarray(
        [float(r["final_wave"]["target_rank_on_right_port_strip"]) for r in rows], dtype=float
    )
    target_gain = arr("target_power_vs_baseline")
    utility_gain = arr("utility_gain")
    return {
        "seeds": [int(r["seed"]) for r in rows],
        "median_desired_over_decoy": float(np.median(ratios)),
        "min_desired_over_decoy": float(np.min(ratios)),
        "max_desired_over_decoy": float(np.max(ratios)),
        "median_target_power_vs_baseline": float(np.median(target_gain)),
        "median_utility_gain": float(np.median(utility_gain)),
        "median_target_rank": float(np.median(ranks)),
        "median_acceptance_fraction": float(np.median(arr("acceptance_fraction"))),
        "max_material_sum_relative_error": float(np.max(arr("material_sum_relative_error"))),
    }


def run_all(out_dir: Path, seeds: int = 6) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["true_consequence", "random_accept", "inverted_consequence", "no_write"]
    runs = {mode: [train_dither(mode, seed=s) for s in range(seeds)] for mode in modes}
    summary = {mode: summarize_runs(runs[mode]) for mode in modes}

    true_runs = runs["true_consequence"]
    best = max(
        true_runs,
        key=lambda x: float(x["result"]["final_wave"]["desired_over_decoy"]),
    )
    write_svg(out_dir / "gate2_dither_topology.svg", best["grid"], best["g"])

    receipt = {
        "gate": 2,
        "question": (
            "Can strictly local conservative material dithers discover a persistent "
            "receiver-selective resonant topology using only delayed scalar consequence, "
            "with no analytic edge gradient and no receiver-launched reference field?"
        ),
        "mechanism": (
            "Choose two edges sharing one node; transfer coupling locally from one to the other; "
            "erase/re-solve fast field; observe one bounded scalar utility; keep or revert."
        ),
        "utility": "log(P_target + sigma^2) - log(P_decoy + sigma^2)",
        "summary": summary,
        "runs": {mode: [x["result"] for x in runs[mode]] for mode in modes},
        "claim_boundary": (
            "A positive result would show derivative-free consequence-selected self-carving in this toy. "
            "The external scalar still identifies what counts as success; this is not autonomous goal formation, "
            "not a biological credit mechanism, and not a claim about dendritic growth."
        ),
    }
    (out_dir / "gate2_local_dither.json").write_text(
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
