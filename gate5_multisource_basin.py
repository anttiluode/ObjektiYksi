from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from gate2_local_dither import local_edge_pairs, make_schedule, transfer_local_material
from gate3_multiport_dither import right_ports
from self_carving import make_grid, node, solve_field, write_svg


TRAIN_OMEGA = 1.55


def source_defs(grid):
    r0 = grid.source // grid.n
    c0 = grid.source % grid.n
    all_defs = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            all_defs.append({"dr": dr, "dc": dc, "node": node(grid.n, r0 + dr, c0 + dc)})
    train = [d for d in all_defs if abs(d["dr"]) + abs(d["dc"]) <= 1]
    heldout = [d for d in all_defs if abs(d["dr"]) == 1 and abs(d["dc"]) == 1]
    return train, heldout, all_defs


def source_margin(grid, g: np.ndarray, source: int, sigma2: float, omega: float = TRAIN_OMEGA):
    u = solve_field(grid, g, source, omega=omega)
    ports = right_ports(grid)
    p = np.abs(u[ports]) ** 2
    ti = ports.index(grid.target)
    pt = float(p[ti])
    other_indices = [i for i in range(len(ports)) if i != ti]
    other_values = p[other_indices]
    oi_local = int(np.argmax(other_values))
    oi = other_indices[oi_local]
    po = float(other_values[oi_local])
    return {
        "utility": math.log(pt + sigma2) - math.log(po + sigma2),
        "target_power": pt,
        "strongest_other_power": po,
        "target_over_strongest_other": pt / max(po, 1e-30),
        "target_rank": int(np.where(np.argsort(-p) == ti)[0][0]) + 1,
        "strongest_other_port_index": int(oi),
    }


def set_measure(grid, g: np.ndarray, defs, sigma2: float, omega: float = TRAIN_OMEGA):
    rows = []
    for d in defs:
        m = source_margin(grid, g, int(d["node"]), sigma2=sigma2, omega=omega)
        rows.append({**d, **m})
    utilities = np.asarray([float(r["utility"]) for r in rows])
    ratios = np.asarray([float(r["target_over_strongest_other"]) for r in rows])
    ranks = np.asarray([int(r["target_rank"]) for r in rows])
    worst = int(np.argmin(utilities))
    return {
        "rows": rows,
        "worst_utility": float(utilities[worst]),
        "worst_source_index": worst,
        "worst_source_dr": int(rows[worst]["dr"]),
        "worst_source_dc": int(rows[worst]["dc"]),
        "worst_source_competitor": int(rows[worst]["strongest_other_port_index"]),
        "min_target_over_strongest_other": float(np.min(ratios)),
        "median_target_over_strongest_other": float(np.median(ratios)),
        "rank1_fraction": float(np.mean(ranks == 1)),
    }


def baseline_floor(grid, train_defs, floor_fraction: float):
    g = np.ones(len(grid.edges), dtype=float)
    raw = []
    for d in train_defs:
        u = solve_field(grid, g, int(d["node"]), omega=TRAIN_OMEGA)
        raw.append(float(abs(u[grid.target]) ** 2))
    return floor_fraction * float(np.median(np.asarray(raw)))


def train(
    mode: str,
    seed: int,
    n: int = 11,
    attempts: int = 520,
    delta: float = 0.055,
    floor_fraction: float = 0.01,
):
    allowed = {"worst_case_multi_source", "center_only", "random_accept", "no_write"}
    if mode not in allowed:
        raise ValueError(mode)

    grid = make_grid(n)
    train_defs, heldout_defs, all_defs = source_defs(grid)
    g = np.ones(len(grid.edges), dtype=float)
    total0 = float(np.sum(g))
    sigma2 = baseline_floor(grid, train_defs, floor_fraction)
    pairs = local_edge_pairs(grid)
    schedule = make_schedule(len(pairs), attempts, seed)

    train_now = set_measure(grid, g, train_defs, sigma2)
    center_def = next(d for d in train_defs if d["dr"] == 0 and d["dc"] == 0)
    center_now = source_margin(grid, g, int(center_def["node"]), sigma2)
    accepted = 0
    proposed = 0
    worst_source_switches = 0
    worst_competitor_switches = 0
    last_worst = (int(train_now["worst_source_dr"]), int(train_now["worst_source_dc"]))
    last_comp = int(train_now["worst_source_competitor"])

    if mode != "no_write":
        for k in range(attempts):
            ea, eb = map(int, pairs[int(schedule.pair_index[k])])
            cand = transfer_local_material(g, ea, eb, int(schedule.orientation[k]), delta=delta)
            if cand is None:
                continue
            proposed += 1

            if mode == "worst_case_multi_source":
                cand_train = set_measure(grid, cand, train_defs, sigma2)
                take = float(cand_train["worst_utility"]) > float(train_now["worst_utility"]) + 1e-14
            elif mode == "center_only":
                cand_center = source_margin(grid, cand, int(center_def["node"]), sigma2)
                take = float(cand_center["utility"]) > float(center_now["utility"]) + 1e-14
            else:
                take = bool(schedule.coin[k] < 0.5)

            if take:
                g = cand
                accepted += 1
                if mode == "worst_case_multi_source":
                    train_now = cand_train
                    center_now = source_margin(grid, g, int(center_def["node"]), sigma2)
                    new_worst = (int(train_now["worst_source_dr"]), int(train_now["worst_source_dc"]))
                    new_comp = int(train_now["worst_source_competitor"])
                    if new_worst != last_worst:
                        worst_source_switches += 1
                        last_worst = new_worst
                    if new_comp != last_comp:
                        worst_competitor_switches += 1
                        last_comp = new_comp
                elif mode == "center_only":
                    center_now = cand_center
                    train_now = set_measure(grid, g, train_defs, sigma2)

    train_final = set_measure(grid, g, train_defs, sigma2)
    heldout_final = set_measure(grid, g, heldout_defs, sigma2)
    all_final = set_measure(grid, g, all_defs, sigma2)
    baseline_g = np.ones_like(g)
    baseline_train = set_measure(grid, baseline_g, train_defs, sigma2)
    baseline_heldout = set_measure(grid, baseline_g, heldout_defs, sigma2)

    result = {
        "mode": mode,
        "seed": seed,
        "n": n,
        "attempts": attempts,
        "delta": delta,
        "sigma2": sigma2,
        "accepted": accepted,
        "proposed": proposed,
        "acceptance_fraction": accepted / max(proposed, 1),
        "worst_source_switches": worst_source_switches,
        "worst_competitor_switches": worst_competitor_switches,
        "material_sum_relative_error": abs(float(np.sum(g)) - total0) / total0,
        "changed_edges": int(np.count_nonzero(np.abs(g - 1.0) > 1e-12)),
        "baseline_train": {k: v for k, v in baseline_train.items() if k != "rows"},
        "baseline_heldout": {k: v for k, v in baseline_heldout.items() if k != "rows"},
        "train": train_final,
        "heldout_corners": heldout_final,
        "all_3x3": all_final,
    }
    return {"grid": grid, "g": g, "result": result}


def summarize(runs):
    rows = [x["result"] for x in runs]
    def vals(section, key):
        return np.asarray([float(r[section][key]) for r in rows])
    return {
        "seeds": [int(r["seed"]) for r in rows],
        "median_train_min_ratio": float(np.median(vals("train", "min_target_over_strongest_other"))),
        "min_train_min_ratio": float(np.min(vals("train", "min_target_over_strongest_other"))),
        "median_train_rank1_fraction": float(np.median(vals("train", "rank1_fraction"))),
        "median_heldout_min_ratio": float(np.median(vals("heldout_corners", "min_target_over_strongest_other"))),
        "min_heldout_min_ratio": float(np.min(vals("heldout_corners", "min_target_over_strongest_other"))),
        "median_heldout_ratio": float(np.median(vals("heldout_corners", "median_target_over_strongest_other"))),
        "median_heldout_rank1_fraction": float(np.median(vals("heldout_corners", "rank1_fraction"))),
        "median_all_3x3_rank1_fraction": float(np.median(vals("all_3x3", "rank1_fraction"))),
        "median_acceptance_fraction": float(np.median(np.asarray([float(r["acceptance_fraction"]) for r in rows]))),
        "median_worst_source_switches": float(np.median(np.asarray([float(r["worst_source_switches"]) for r in rows]))),
        "median_worst_competitor_switches": float(np.median(np.asarray([float(r["worst_competitor_switches"]) for r in rows]))),
        "max_material_sum_relative_error": float(np.max(np.asarray([float(r["material_sum_relative_error"]) for r in rows]))),
    }


def run_all(out_dir: Path, seeds: int = 4):
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["worst_case_multi_source", "center_only", "random_accept", "no_write"]
    runs = {m: [train(m, seed=s) for s in range(seeds)] for m in modes}
    summary = {m: summarize(runs[m]) for m in modes}
    best = max(
        runs["worst_case_multi_source"],
        key=lambda x: float(x["result"]["heldout_corners"]["min_target_over_strongest_other"]),
    )
    write_svg(out_dir / "gate5_multisource_topology.svg", best["grid"], best["g"])
    receipt = {
        "gate": 5,
        "question": "Can one material learn a receiver basin across several source landing points and generalize to unseen nearby sources?",
        "training_sources": "center plus four cardinal one-step neighbors",
        "heldout_sources": "four diagonal one-step neighbors",
        "learning_scalar": "minimum across training sources of [log(P_target+sigma^2)-log(max_other P+sigma^2)]",
        "information_hidden_from_structural_rule": "identity of worst source, identity of strongest competitor, edge gradient",
        "summary": summary,
        "runs": {m: [x["result"] for x in runs[m]] for m in modes},
        "claim_boundary": "This is still externally specified robust routing in a toy. Success would show a spatial basin can be compiled from local dither; failure would support source-coordinate specificity rather than prove it intrinsic.",
    }
    (out_dir / "gate5_multisource_basin.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, default=4)
    args = p.parse_args()
    receipt = run_all(Path(args.out), seeds=args.seeds)
    print(json.dumps(receipt["summary"], indent=2))


if __name__ == "__main__":
    main()
