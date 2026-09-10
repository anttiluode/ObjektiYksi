from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from gate2_local_dither import local_edge_pairs, make_schedule, transfer_local_material
from gate3_multiport_dither import right_ports
from self_carving import make_grid, node, solve_field, write_svg


OMEGA = 1.35
SOURCE_DRS = (-1, 0, 1)
SUCCESS_RATIO = 1.10
RANDOM_MARGIN = 1.10


def relation_defs(grid, mapping: str):
    """Three fixed source positions mapped to three fixed output ports.

    `identity` follows the native vertical ordering.
    `crossed` is a 3-cycle: upper -> lower, center -> upper, lower -> center.
    No selector may relabel sources during learning.
    """
    ports = right_ports(grid)
    mid = len(ports) // 2
    target_local = [mid - 2, mid, mid + 2]
    if mapping == "identity":
        mapped_local = target_local
    elif mapping == "crossed":
        mapped_local = [target_local[2], target_local[0], target_local[1]]
    else:
        raise ValueError(mapping)

    base_row = grid.source // grid.n
    source_col = grid.source % grid.n
    defs = []
    for dr, ti in zip(SOURCE_DRS, mapped_local):
        defs.append(
            {
                "source_dr": int(dr),
                "source": int(node(grid.n, base_row + dr, source_col)),
                "target_port_index": int(ti),
                "target_node": int(ports[ti]),
            }
        )
    return defs


def relation_measure(grid, g: np.ndarray, rel, sigma2: float):
    ports = right_ports(grid)
    u = solve_field(grid, g, int(rel["source"]), omega=OMEGA)
    power = np.abs(u[ports]) ** 2
    ti = int(rel["target_port_index"])
    pt = float(power[ti])
    others = [i for i in range(len(ports)) if i != ti]
    other_power = power[others]
    oi_local = int(np.argmax(other_power))
    oi = int(others[oi_local])
    po = float(other_power[oi_local])
    rank = int(np.where(np.argsort(-power) == ti)[0][0]) + 1
    return {
        **rel,
        "utility": float(math.log(pt + sigma2) - math.log(po + sigma2)),
        "target_power": pt,
        "strongest_other_power": po,
        "target_over_strongest_other": pt / max(po, 1e-30),
        "target_rank": rank,
        "strongest_other_port_index": oi,
    }


def mapping_measure(grid, g: np.ndarray, mapping: str, sigma2: float):
    rows = [relation_measure(grid, g, r, sigma2) for r in relation_defs(grid, mapping)]
    utilities = np.asarray([float(r["utility"]) for r in rows])
    ratios = np.asarray([float(r["target_over_strongest_other"]) for r in rows])
    ranks = np.asarray([int(r["target_rank"]) for r in rows])
    wi = int(np.argmin(utilities))
    return {
        "rows": rows,
        "worst_utility": float(utilities[wi]),
        "worst_relation_index": wi,
        "worst_relation_source_dr": int(rows[wi]["source_dr"]),
        "worst_relation_competitor": int(rows[wi]["strongest_other_port_index"]),
        "min_ratio": float(np.min(ratios)),
        "median_ratio": float(np.median(ratios)),
        "rank1_fraction": float(np.mean(ranks == 1)),
    }


def common_sigma2(grid, floor_fraction: float = 0.01) -> float:
    """Mapping-independent observer floor from all 3 sources x all 3 task ports."""
    g = np.ones(len(grid.edges), dtype=float)
    ports = right_ports(grid)
    mid = len(ports) // 2
    selected = [mid - 2, mid, mid + 2]
    base_row = grid.source // grid.n
    source_col = grid.source % grid.n
    powers = []
    for dr in SOURCE_DRS:
        source = node(grid.n, base_row + dr, source_col)
        u = solve_field(grid, g, source, omega=OMEGA)
        p = np.abs(u[ports]) ** 2
        powers.extend(float(p[i]) for i in selected)
    return float(floor_fraction * np.median(np.asarray(powers)))


def train_mode(
    mode: str,
    seed: int,
    n: int = 11,
    attempts: int = 1200,
    delta: float = 0.055,
    floor_fraction: float = 0.01,
):
    allowed = {"crossed", "identity", "random_crossed", "no_write_crossed"}
    if mode not in allowed:
        raise ValueError(mode)

    mapping = "identity" if mode == "identity" else "crossed"
    grid = make_grid(n)
    g = np.ones(len(grid.edges), dtype=float)
    total0 = float(np.sum(g))
    sigma2 = common_sigma2(grid, floor_fraction)
    pairs = local_edge_pairs(grid)
    schedule = make_schedule(len(pairs), attempts, seed)

    baseline = mapping_measure(grid, g, mapping, sigma2)
    current = baseline
    accepted = 0
    proposed = 0
    relation_switches = 0
    competitor_switches = 0
    last_relation = int(current["worst_relation_index"])
    last_competitor = int(current["worst_relation_competitor"])
    checkpoints = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 800, attempts}
    trajectory = [{"attempt": 0, "min_ratio": baseline["min_ratio"], "rank1_fraction": baseline["rank1_fraction"]}]

    if mode != "no_write_crossed":
        for k in range(attempts):
            ea, eb = map(int, pairs[int(schedule.pair_index[k])])
            cand = transfer_local_material(g, ea, eb, int(schedule.orientation[k]), delta=delta)
            if cand is None:
                continue
            proposed += 1
            cm = mapping_measure(grid, cand, mapping, sigma2)
            if mode == "random_crossed":
                take = bool(schedule.coin[k] < 0.5)
            else:
                take = float(cm["worst_utility"]) > float(current["worst_utility"]) + 1e-14

            if take:
                g = cand
                current = cm
                accepted += 1
                rel = int(current["worst_relation_index"])
                comp = int(current["worst_relation_competitor"])
                if rel != last_relation:
                    relation_switches += 1
                    last_relation = rel
                if comp != last_competitor:
                    competitor_switches += 1
                    last_competitor = comp

            if (k + 1) in checkpoints:
                trajectory.append(
                    {
                        "attempt": k + 1,
                        "min_ratio": float(current["min_ratio"]),
                        "rank1_fraction": float(current["rank1_fraction"]),
                    }
                )

    final = mapping_measure(grid, g, mapping, sigma2)
    result = {
        "mode": mode,
        "mapping": mapping,
        "seed": seed,
        "omega": OMEGA,
        "attempts": attempts,
        "delta": delta,
        "sigma2_common": sigma2,
        "relations": relation_defs(grid, mapping),
        "baseline": baseline,
        "final": final,
        "accepted": accepted,
        "proposed": proposed,
        "acceptance_fraction": accepted / max(proposed, 1),
        "worst_relation_switches": relation_switches,
        "worst_competitor_switches": competitor_switches,
        "material_sum_relative_error": abs(float(np.sum(g)) - total0) / total0,
        "changed_edges": int(np.count_nonzero(np.abs(g - 1.0) > 1e-12)),
        "trajectory": trajectory,
    }
    return {"grid": grid, "g": g, "result": result}


def summarize(runs):
    rows = [x["result"] for x in runs]

    def vals(section, key):
        return np.asarray([float(r[section][key]) for r in rows])

    return {
        "seeds": [int(r["seed"]) for r in rows],
        "median_baseline_min_ratio": float(np.median(vals("baseline", "min_ratio"))),
        "median_baseline_rank1_fraction": float(np.median(vals("baseline", "rank1_fraction"))),
        "median_final_min_ratio": float(np.median(vals("final", "min_ratio"))),
        "min_final_min_ratio": float(np.min(vals("final", "min_ratio"))),
        "median_final_rank1_fraction": float(np.median(vals("final", "rank1_fraction"))),
        "median_acceptance_fraction": float(np.median(np.asarray([float(r["acceptance_fraction"]) for r in rows]))),
        "median_worst_relation_switches": float(np.median(np.asarray([float(r["worst_relation_switches"]) for r in rows]))),
        "median_worst_competitor_switches": float(np.median(np.asarray([float(r["worst_competitor_switches"]) for r in rows]))),
        "max_material_sum_relative_error": float(np.max(np.asarray([float(r["material_sum_relative_error"]) for r in rows]))),
    }


def run_all(out_dir: Path, seeds: int = 3, attempts: int = 1200):
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["crossed", "identity", "random_crossed", "no_write_crossed"]
    runs = {m: [train_mode(m, seed=s, attempts=attempts) for s in range(seeds)] for m in modes}
    summary = {m: summarize(rs) for m, rs in runs.items()}

    crossed = summary["crossed"]
    identity = summary["identity"]
    random = summary["random_crossed"]
    no_write = summary["no_write_crossed"]

    crossed_success = bool(
        crossed["median_final_rank1_fraction"] >= 1.0 - 1e-12
        and crossed["median_final_min_ratio"] >= SUCCESS_RATIO
        and crossed["min_final_min_ratio"] > 1.0
        and crossed["median_final_min_ratio"] >= RANDOM_MARGIN * max(random["median_final_min_ratio"], 1e-30)
    )
    identity_success = bool(
        identity["median_final_rank1_fraction"] >= 1.0 - 1e-12
        and identity["median_final_min_ratio"] >= SUCCESS_RATIO
        and identity["min_final_min_ratio"] > 1.0
    )

    if crossed_success:
        verdict = "crossed_permutation_survives_native_geometry_attack"
    elif identity_success:
        verdict = "crossed_fails_while_native_map_works"
    else:
        verdict = "inconclusive_under_preregistered_budget"

    best = max(runs["crossed"], key=lambda x: float(x["result"]["final"]["min_ratio"]))
    write_svg(out_dir / "gate8_crossed_topology.svg", best["grid"], best["g"])

    receipt = {
        "gate": 8,
        "question": "Is the spatial address result programmable, or is it merely the blank lattice's native vertical geometry?",
        "fixed_frequency": OMEGA,
        "identity_map": "upper->upper, center->center, lower->lower task ports",
        "crossed_map": "upper->lower, center->upper, lower->center (3-cycle); no source may be relabeled by a selector",
        "learning_scalar": "minimum over three fixed source-target relations of log(P_target+sigma^2)-log(max_other P+sigma^2)",
        "information_hidden": "identity of current worst relation, identity of strongest competitor, edge gradient",
        "fairness": "all modes share a mapping-independent observer floor and, for a given seed, the same local material proposal schedule",
        "preregistered_success": {
            "median_crossed_rank1_fraction": 1.0,
            "median_crossed_min_ratio_at_least": SUCCESS_RATIO,
            "every_seed_crossed_min_ratio_above": 1.0,
            "crossed_median_ratio_at_least_random_times": RANDOM_MARGIN,
        },
        "summary": summary,
        "crossed_success": crossed_success,
        "identity_success": identity_success,
        "verdict": verdict,
        "runs": {m: [x["result"] for x in rs] for m, rs in runs.items()},
        "claim_boundary": "Success would establish only that this toy substrate can be locally reconfigured into a non-native three-relation spatial permutation under bounded scalar feedback. It would not establish arbitrary mappings, biological learning, or a hardware advantage.",
    }
    (out_dir / "gate8_crossed_spatial_map.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--attempts", type=int, default=1200)
    args = p.parse_args()
    receipt = run_all(Path(args.out), seeds=args.seeds, attempts=args.attempts)
    print(json.dumps({"summary": receipt["summary"], "verdict": receipt["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
