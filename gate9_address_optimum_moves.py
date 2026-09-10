from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate8_crossed_spatial_map import (
    OMEGA,
    SOURCE_DRS,
    common_sigma2,
    relation_defs,
    relation_measure,
    train_mode,
)
from gate3_multiport_dither import right_ports
from self_carving import make_grid, node


MIN_ADVANTAGE = 0.05


def source_node(grid, dr: int) -> int:
    row = grid.source // grid.n + dr
    col = grid.source % grid.n
    return int(node(grid.n, row, col))


def target_source_maps(grid):
    identity = relation_defs(grid, "identity")
    crossed = relation_defs(grid, "crossed")
    native = {int(r["target_port_index"]): int(r["source_dr"]) for r in identity}
    assigned = {int(r["target_port_index"]): int(r["source_dr"]) for r in crossed}
    return native, assigned


def audit_target_sources(grid, g: np.ndarray, sigma2: float):
    """Audit which of the three physical source rows best addresses each task target.

    This function is audit-only. It is never used by Gate-8 learning.
    """
    ports = right_ports(grid)
    native, assigned = target_source_maps(grid)
    target_indices = sorted(native)
    rows = []

    for ti in target_indices:
        by_source = []
        for dr in SOURCE_DRS:
            rel = {
                "source_dr": int(dr),
                "source": source_node(grid, int(dr)),
                "target_port_index": int(ti),
                "target_node": int(ports[ti]),
            }
            by_source.append(relation_measure(grid, g, rel, sigma2))

        utilities = np.asarray([float(x["utility"]) for x in by_source])
        best_i = int(np.argmax(utilities))
        best_dr = int(SOURCE_DRS[best_i])
        assigned_dr = int(assigned[ti])
        native_dr = int(native[ti])
        assigned_i = SOURCE_DRS.index(assigned_dr)
        native_i = SOURCE_DRS.index(native_dr)
        other = np.delete(utilities, assigned_i)
        rows.append(
            {
                "target_port_index": int(ti),
                "native_source_dr": native_dr,
                "crossed_assigned_source_dr": assigned_dr,
                "audit_best_source_dr": best_dr,
                "assigned_is_best": bool(best_dr == assigned_dr),
                "native_is_best": bool(best_dr == native_dr),
                "assigned_utility": float(utilities[assigned_i]),
                "native_utility": float(utilities[native_i]),
                "assigned_minus_native_utility": float(utilities[assigned_i] - utilities[native_i]),
                "assigned_minus_runner_up_utility": float(utilities[assigned_i] - np.max(other)),
                "by_source": [
                    {
                        "source_dr": int(x["source_dr"]),
                        "utility": float(x["utility"]),
                        "ratio": float(x["target_over_strongest_other"]),
                        "rank": int(x["target_rank"]),
                    }
                    for x in by_source
                ],
            }
        )

    return {
        "rows": rows,
        "assigned_exact_fraction": float(np.mean([r["assigned_is_best"] for r in rows])),
        "native_exact_fraction": float(np.mean([r["native_is_best"] for r in rows])),
        "min_assigned_minus_native_utility": float(min(r["assigned_minus_native_utility"] for r in rows)),
        "median_assigned_minus_native_utility": float(np.median([r["assigned_minus_native_utility"] for r in rows])),
        "min_assigned_minus_runner_up_utility": float(min(r["assigned_minus_runner_up_utility"] for r in rows)),
    }


def audit_mode(mode: str, seed: int, n: int = 11, attempts: int = 1200):
    trained = train_mode(mode, seed=seed, n=n, attempts=attempts)
    grid = trained["grid"]
    g = trained["g"]
    sigma2 = common_sigma2(grid)
    blank = audit_target_sources(grid, np.ones(len(grid.edges), dtype=float), sigma2)
    final = audit_target_sources(grid, g, sigma2)
    return {
        "mode": mode,
        "seed": seed,
        "blank": blank,
        "final": final,
        "gate8_final_min_ratio": float(trained["result"]["final"]["min_ratio"]),
        "gate8_final_rank1_fraction": float(trained["result"]["final"]["rank1_fraction"]),
        "material_sum_relative_error": float(trained["result"]["material_sum_relative_error"]),
    }


def summarize(rows):
    def a(section, key):
        return np.asarray([float(r[section][key]) for r in rows])

    return {
        "seeds": [int(r["seed"]) for r in rows],
        "median_blank_assigned_exact_fraction": float(np.median(a("blank", "assigned_exact_fraction"))),
        "median_blank_native_exact_fraction": float(np.median(a("blank", "native_exact_fraction"))),
        "median_final_assigned_exact_fraction": float(np.median(a("final", "assigned_exact_fraction"))),
        "min_final_assigned_exact_fraction": float(np.min(a("final", "assigned_exact_fraction"))),
        "median_final_native_exact_fraction": float(np.median(a("final", "native_exact_fraction"))),
        "median_min_assigned_minus_native_utility": float(np.median(a("final", "min_assigned_minus_native_utility"))),
        "min_over_seeds_assigned_minus_native_utility": float(np.min(a("final", "min_assigned_minus_native_utility"))),
        "median_min_assigned_minus_runner_up_utility": float(np.median(a("final", "min_assigned_minus_runner_up_utility"))),
        "median_gate8_final_min_ratio": float(np.median(np.asarray([float(r["gate8_final_min_ratio"]) for r in rows]))),
        "max_material_sum_relative_error": float(np.max(np.asarray([float(r["material_sum_relative_error"]) for r in rows]))),
    }


def run_all(out_dir: Path, seeds: int = 3, attempts: int = 1200):
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["crossed", "identity", "random_crossed", "no_write_crossed"]
    runs = {m: [audit_mode(m, seed=s, attempts=attempts) for s in range(seeds)] for m in modes}
    summary = {m: summarize(rs) for m, rs in runs.items()}

    crossed = summary["crossed"]
    random = summary["random_crossed"]
    no_write = summary["no_write_crossed"]

    blank_is_native = bool(
        crossed["median_blank_assigned_exact_fraction"] == 0.0
        and crossed["median_blank_native_exact_fraction"] == 1.0
    )
    crossed_moves = bool(
        blank_is_native
        and crossed["median_final_assigned_exact_fraction"] == 1.0
        and crossed["min_final_assigned_exact_fraction"] >= (2.0 / 3.0) - 1e-12
        and crossed["median_min_assigned_minus_native_utility"] >= MIN_ADVANTAGE
        and crossed["median_final_assigned_exact_fraction"] >= random["median_final_assigned_exact_fraction"] + (1.0 / 3.0) - 1e-12
        and crossed["median_final_assigned_exact_fraction"] >= no_write["median_final_assigned_exact_fraction"] + (1.0 / 3.0) - 1e-12
    )

    if crossed_moves:
        verdict = "address_optima_move_with_learned_material"
    elif blank_is_native and crossed["median_final_assigned_exact_fraction"] <= 1.0 / 3.0 + 1e-12:
        verdict = "crossed_outputs_change_but_address_optima_stay_native"
    else:
        verdict = "partial_or_inconclusive_address_movement"

    receipt = {
        "gate": 9,
        "question": "After Gate-8 crossed remapping, does the audit-best physical source address for each target move from the blank native source to the source assigned by the crossed task?",
        "training": "No new learning rule: rerun the preregistered Gate-8 crossed, identity, random, and no-write conditions, then audit all three source rows for each target.",
        "audit_only": "Source scans never enter Gate-8 acceptance decisions.",
        "preregistered_success": {
            "blank_assigned_exact_fraction": 0.0,
            "blank_native_exact_fraction": 1.0,
            "median_crossed_final_assigned_exact_fraction": 1.0,
            "every_crossed_seed_assigned_exact_fraction_at_least": 2.0 / 3.0,
            "median_min_assigned_minus_old_native_utility_at_least": MIN_ADVANTAGE,
            "crossed_assigned_fraction_exceeds_random_and_no_write_by_at_least": 1.0 / 3.0,
        },
        "summary": summary,
        "blank_is_native": blank_is_native,
        "crossed_moves": crossed_moves,
        "verdict": verdict,
        "runs": runs,
        "claim_boundary": "A positive result would show only that Gate-8 material deformation can move the best source coordinate for these three target relations in this engineered lattice. It would not establish autonomous address learning, arbitrary coordinate warping, or biological dendritic remapping.",
    }
    (out_dir / "gate9_address_optimum_moves.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
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
