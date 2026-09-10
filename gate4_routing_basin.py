from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate3_multiport_dither import right_ports, train_multiport
from self_carving import make_grid, node, solve_field


TRAIN_OMEGA = 1.55


def measure_at(grid, g: np.ndarray, omega: float, source: int) -> dict[str, float | int]:
    u = solve_field(grid, g, source, omega=omega)
    ports = right_ports(grid)
    power = np.abs(u[ports]) ** 2
    target_i = ports.index(grid.target)
    target_power = float(power[target_i])
    other = np.delete(power, target_i)
    strongest = float(np.max(other))
    order = np.argsort(-power)
    rank = int(np.where(order == target_i)[0][0]) + 1
    return {
        "omega": float(omega),
        "source": int(source),
        "target_power": target_power,
        "target_over_strongest_other": target_power / max(strongest, 1e-30),
        "target_rank": rank,
    }


def nearby_sources(grid) -> list[dict[str, int]]:
    r0 = grid.source // grid.n
    c0 = grid.source % grid.n
    out: list[dict[str, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r = r0 + dr
            c = c0 + dc
            if 0 <= r < grid.n and 0 <= c < grid.n:
                out.append({"dr": dr, "dc": dc, "node": node(grid.n, r, c)})
    return out


def contiguous_rank1_band(rows: list[dict], center: float = TRAIN_OMEGA) -> dict[str, float | int]:
    rows = sorted(rows, key=lambda x: float(x["omega"]))
    omegas = np.asarray([float(r["omega"]) for r in rows])
    good = np.asarray([int(r["target_rank"]) == 1 for r in rows], dtype=bool)
    k = int(np.argmin(np.abs(omegas - center)))
    if not good[k]:
        return {"low": float(center), "high": float(center), "width": 0.0, "samples": 0}
    lo = k
    hi = k
    while lo > 0 and good[lo - 1]:
        lo -= 1
    while hi + 1 < len(good) and good[hi + 1]:
        hi += 1
    return {
        "low": float(omegas[lo]),
        "high": float(omegas[hi]),
        "width": float(omegas[hi] - omegas[lo]),
        "samples": int(hi - lo + 1),
    }


def summarize_boolean_and_ratio(rows: list[dict], exclude=None) -> dict[str, float | int]:
    selected = [r for r in rows if exclude is None or not exclude(r)]
    ratio = np.asarray([float(r["target_over_strongest_other"]) for r in selected])
    rank1 = np.asarray([int(r["target_rank"]) == 1 for r in selected], dtype=float)
    return {
        "samples": len(selected),
        "rank1_fraction": float(np.mean(rank1)) if len(rank1) else float("nan"),
        "median_target_over_strongest_other": float(np.median(ratio)) if len(ratio) else float("nan"),
        "min_target_over_strongest_other": float(np.min(ratio)) if len(ratio) else float("nan"),
        "max_target_over_strongest_other": float(np.max(ratio)) if len(ratio) else float("nan"),
    }


def evaluate_material(grid, g: np.ndarray) -> dict:
    frequency_grid = np.round(np.arange(1.35, 1.7501, 0.025), 6)
    freq_rows = [measure_at(grid, g, float(w), grid.source) for w in frequency_grid]

    source_defs = nearby_sources(grid)
    source_rows = []
    for s in source_defs:
        m = measure_at(grid, g, TRAIN_OMEGA, int(s["node"]))
        source_rows.append({**s, **m})

    coarse_frequency_grid = [1.45, 1.50, 1.55, 1.60, 1.65]
    joint_rows = []
    for w in coarse_frequency_grid:
        for s in source_defs:
            m = measure_at(grid, g, w, int(s["node"]))
            joint_rows.append({**s, **m})

    return {
        "frequency_rows": freq_rows,
        "frequency_all": summarize_boolean_and_ratio(freq_rows),
        "frequency_heldout": summarize_boolean_and_ratio(
            freq_rows, exclude=lambda r: abs(float(r["omega"]) - TRAIN_OMEGA) < 1e-12
        ),
        "contiguous_rank1_band": contiguous_rank1_band(freq_rows),
        "source_rows": source_rows,
        "source_all": summarize_boolean_and_ratio(source_rows),
        "source_heldout": summarize_boolean_and_ratio(
            source_rows, exclude=lambda r: int(r["dr"]) == 0 and int(r["dc"]) == 0
        ),
        "joint_rows": joint_rows,
        "joint_all": summarize_boolean_and_ratio(joint_rows),
        "joint_heldout": summarize_boolean_and_ratio(
            joint_rows,
            exclude=lambda r: (
                abs(float(r["omega"]) - TRAIN_OMEGA) < 1e-12
                and int(r["dr"]) == 0
                and int(r["dc"]) == 0
            ),
        ),
    }


def aggregate(evals: list[dict]) -> dict:
    def vals(path: tuple[str, str]) -> np.ndarray:
        return np.asarray([float(e[path[0]][path[1]]) for e in evals], dtype=float)

    bands = np.asarray([float(e["contiguous_rank1_band"]["width"]) for e in evals])
    return {
        "median_contiguous_rank1_frequency_width": float(np.median(bands)),
        "min_contiguous_rank1_frequency_width": float(np.min(bands)),
        "median_heldout_frequency_rank1_fraction": float(np.median(vals(("frequency_heldout", "rank1_fraction")))),
        "median_heldout_frequency_ratio": float(np.median(vals(("frequency_heldout", "median_target_over_strongest_other")))),
        "median_heldout_source_rank1_fraction": float(np.median(vals(("source_heldout", "rank1_fraction")))),
        "median_heldout_source_ratio": float(np.median(vals(("source_heldout", "median_target_over_strongest_other")))),
        "median_joint_rank1_fraction": float(np.median(vals(("joint_heldout", "rank1_fraction")))),
        "median_joint_ratio": float(np.median(vals(("joint_heldout", "median_target_over_strongest_other")))),
        "worst_seed_joint_min_ratio": float(np.min(vals(("joint_heldout", "min_target_over_strongest_other")))),
    }


def run_all(out_dir: Path, seeds: int = 6) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = make_grid(11)
    baseline_g = np.ones(len(grid.edges), dtype=float)
    baseline_eval = evaluate_material(grid, baseline_g)

    trained = [train_multiport("true_multiport", seed=s) for s in range(seeds)]
    trained_evals = [evaluate_material(x["grid"], x["g"]) for x in trained]

    receipt = {
        "gate": 4,
        "question": (
            "After Gate-3 learning is frozen, does receiver selectivity survive held-out "
            "carrier frequencies and nearby source positions without any further write?"
        ),
        "training": {
            "omega": TRAIN_OMEGA,
            "source_node": int(grid.source),
            "seeds": seeds,
            "attempts_per_seed": int(trained[0]["result"]["attempts"]),
        },
        "baseline_uncarved": baseline_eval,
        "trained": [
            {"seed": s, "gate3_final": trained[s]["result"]["final"], "basin": trained_evals[s]}
            for s in range(seeds)
        ],
        "aggregate_trained": aggregate(trained_evals),
        "claim_boundary": (
            "This measures an off-training basin of a frozen toy wave operator. Nearby source "
            "locations are perturbations of the input port, not new tasks; frequency robustness "
            "does not imply broadband biological computation."
        ),
    }
    (out_dir / "gate4_routing_basin.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, default=6)
    args = p.parse_args()
    receipt = run_all(Path(args.out), seeds=args.seeds)
    print(json.dumps(receipt["aggregate_trained"], indent=2))


if __name__ == "__main__":
    main()
