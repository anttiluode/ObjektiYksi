from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from self_carving import make_grid, query_metrics, solve_field, train


def mirror_couplings(grid, g: np.ndarray) -> np.ndarray:
    """Mirror persistent material across the horizontal midline."""
    edge_index = {tuple(sorted(map(int, e))): k for k, e in enumerate(grid.edges)}
    out = np.empty_like(g)
    n = grid.n

    def mirror_node(k: int) -> int:
        r, c = divmod(k, n)
        return (n - 1 - r) * n + c

    for k, (i, j) in enumerate(grid.edges):
        mi, mj = mirror_node(int(i)), mirror_node(int(j))
        out[edge_index[tuple(sorted((mi, mj)))]] = g[k]
    return out


def spectral_metrics(grid, g: np.ndarray, omega: float) -> dict[str, float | int]:
    u = solve_field(grid, g, grid.source, omega=float(omega))
    power = np.abs(u) ** 2
    pt = float(power[grid.target])
    pd = float(power[grid.decoy])
    total = float(np.sum(power))

    col = grid.n - 2
    port_nodes = [r * grid.n + col for r in range(1, grid.n - 1)]
    port_power = power[port_nodes]
    target_pos = port_nodes.index(grid.target)
    order = np.argsort(-port_power)
    rank = int(np.where(order == target_pos)[0][0]) + 1
    other_max = float(np.max(np.delete(port_power, target_pos)))

    return {
        "omega": float(omega),
        "desired_power": pt,
        "decoy_power": pd,
        "desired_over_decoy": pt / max(pd, 1e-30),
        "total_field_power": total,
        "desired_fraction_of_total": pt / max(total, 1e-30),
        "target_rank_on_right_port_strip": rank,
        "target_over_strongest_other_right_port": pt / max(other_max, 1e-30),
    }


def audit() -> dict:
    run = train("phase_reference")
    grid = run["grid"]
    carved = run["g"]
    baseline = np.ones_like(carved)
    mirrored = mirror_couplings(grid, carved)

    freq = np.linspace(1.25, 1.85, 25)
    base_sweep = [spectral_metrics(grid, baseline, w) for w in freq]
    carved_sweep = [spectral_metrics(grid, carved, w) for w in freq]

    base_train = spectral_metrics(grid, baseline, 1.55)
    carved_train = spectral_metrics(grid, carved, 1.55)
    mirror_train = spectral_metrics(grid, mirrored, 1.55)
    carved_paths = query_metrics(grid, carved)

    carved_peak = max(carved_sweep, key=lambda x: x["desired_power"])
    base_peak = max(base_sweep, key=lambda x: x["desired_power"])
    half_peak = 0.5 * float(carved_peak["desired_power"])
    above_half = [x["omega"] for x in carved_sweep if x["desired_power"] >= half_peak]
    bandwidth = float(max(above_half) - min(above_half)) if above_half else 0.0

    return {
        "gate": 1,
        "question": "did Gate 0 carve a static shortest path, or a frequency-dependent transfer topology?",
        "trained_omega": 1.55,
        "baseline_at_trained_omega": base_train,
        "carved_at_trained_omega": carved_train,
        "mirrored_material_at_trained_omega": mirror_train,
        "mirror_product_ratio": float(
            carved_train["desired_over_decoy"] * mirror_train["desired_over_decoy"]
        ),
        "static_path_metrics_after_carving": carved_paths,
        "shortest_path_favors_desired": bool(
            carved_paths["desired_path_cost"] < carved_paths["decoy_path_cost"]
        ),
        "baseline_peak": base_peak,
        "carved_peak": carved_peak,
        "carved_half_power_sampled_bandwidth": bandwidth,
        "baseline_frequency_sweep": base_sweep,
        "carved_frequency_sweep": carved_sweep,
        "interpretation_guardrail": (
            "If target transfer improves while 1/g shortest-path cost does not favor the target, "
            "the persistent object is not adequately described as a static high-conductance road. "
            "It is a frequency-dependent wave operator."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    receipt = audit()
    (out / "gate1_route_or_resonance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
