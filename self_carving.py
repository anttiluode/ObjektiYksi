from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Grid:
    n: int
    edges: np.ndarray
    xy: np.ndarray
    source: int
    target: int
    decoy: int


def node(n: int, row: int, col: int) -> int:
    return row * n + col


def make_grid(n: int = 11) -> Grid:
    if n < 7 or n % 2 == 0:
        raise ValueError("n must be odd and >= 7")
    edges: list[tuple[int, int]] = []
    xy = np.zeros((n * n, 2), dtype=float)
    for r in range(n):
        for c in range(n):
            k = node(n, r, c)
            xy[k] = (float(c), float(r))
            if c + 1 < n:
                edges.append((k, node(n, r, c + 1)))
            if r + 1 < n:
                edges.append((k, node(n, r + 1, c)))
    mid = n // 2
    off = max(2, n // 4)
    return Grid(
        n=n,
        edges=np.asarray(edges, dtype=int),
        xy=xy,
        source=node(n, mid, 1),
        target=node(n, mid - off, n - 2),
        decoy=node(n, mid + off, n - 2),
    )


def stiffness(grid: Grid, g: np.ndarray, pin: float = 0.55) -> np.ndarray:
    n_nodes = grid.n * grid.n
    K = np.eye(n_nodes, dtype=float) * pin
    for ge, (i, j) in zip(g, grid.edges):
        K[i, i] += ge
        K[j, j] += ge
        K[i, j] -= ge
        K[j, i] -= ge
    return K


def operator_matrix(
    grid: Grid,
    g: np.ndarray,
    omega: float = 1.55,
    damping: float = 0.14,
    pin: float = 0.55,
) -> np.ndarray:
    n_nodes = grid.n * grid.n
    K = stiffness(grid, g, pin=pin)
    return K - (omega**2) * np.eye(n_nodes) + 1j * omega * damping * np.eye(n_nodes)


def solve_field(
    grid: Grid,
    g: np.ndarray,
    drive_node: int,
    drive: complex = 1.0 + 0.0j,
    omega: float = 1.55,
    damping: float = 0.14,
    pin: float = 0.55,
) -> np.ndarray:
    f = np.zeros(grid.n * grid.n, dtype=complex)
    f[drive_node] = drive
    A = operator_matrix(grid, g, omega=omega, damping=damping, pin=pin)
    return np.linalg.solve(A, f)


def edge_drop(grid: Grid, u: np.ndarray) -> np.ndarray:
    i = grid.edges[:, 0]
    j = grid.edges[:, 1]
    return u[i] - u[j]


def receiver_power_gradient_logg(
    grid: Grid,
    g: np.ndarray,
    source: int | None = None,
    target: int | None = None,
    omega: float = 1.55,
    damping: float = 0.14,
    pin: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, complex]:
    """Exact first derivative of |H[target, source]|^2 w.r.t. log edge coupling.

    The dynamic operator is complex symmetric, not Hermitian. Reciprocity therefore
    uses transpose symmetry. The real scalar derivative is

        d|y|^2/dg_e = -2 Re[y* (b_e^T r) (b_e^T u)]

    where u = H s and r = H t.
    """
    source = grid.source if source is None else source
    target = grid.target if target is None else target
    u = solve_field(grid, g, source, omega=omega, damping=damping, pin=pin)
    r = solve_field(grid, g, target, omega=omega, damping=damping, pin=pin)
    y = u[target]
    du = edge_drop(grid, u)
    dr = edge_drop(grid, r)
    grad_g = -2.0 * np.real(np.conj(y) * dr * du)
    return g * grad_g, u, r, y


def project_fixed_mean_tangent(g: np.ndarray, score_log: np.ndarray) -> np.ndarray:
    """Project a log-coupling step onto the first-order fixed-mean material budget."""
    c = float(np.sum(g * score_log) / np.sum(g))
    q = score_log - c
    rms = float(np.sqrt(np.mean(q * q)))
    if rms < 1e-15:
        return np.zeros_like(q)
    return q / rms


def material_step(
    grid: Grid,
    g: np.ndarray,
    rule: str,
    permutation: np.ndarray,
    eta: float = 0.045,
    omega: float = 1.55,
    damping: float = 0.14,
    pin: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    grad_log, u, r, _ = receiver_power_gradient_logg(
        grid, g, omega=omega, damping=damping, pin=pin
    )
    du = edge_drop(grid, u)
    dr = edge_drop(grid, r)

    if rule == "phase_reference":
        raw = grad_log
    elif rule == "magnitude_reference":
        raw = g * np.abs(du) * np.abs(dr)
    elif rule == "forward_only":
        raw = g * np.abs(du) ** 2
    elif rule == "scrambled_reference":
        raw = grad_log[permutation]
    elif rule == "no_write":
        return g.copy(), np.zeros_like(g)
    else:
        raise ValueError(f"unknown rule: {rule}")

    q = project_fixed_mean_tangent(g, np.asarray(raw, dtype=float))
    logg = np.log(g) + eta * q
    logg = np.clip(logg, -2.2, 2.2)
    g_new = np.exp(logg)
    # Exact resource control after the finite step.
    g_new /= float(np.mean(g_new))
    return g_new, q


def dijkstra_cost(grid: Grid, g: np.ndarray, start: int, goal: int) -> float:
    adj: list[list[tuple[int, float]]] = [[] for _ in range(grid.n * grid.n)]
    for ge, (i, j) in zip(g, grid.edges):
        cost = 1.0 / max(float(ge), 1e-12)
        adj[i].append((j, cost))
        adj[j].append((i, cost))
    dist = [math.inf] * len(adj)
    dist[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start)]
    while heap:
        d, u = heappop(heap)
        if d != dist[u]:
            continue
        if u == goal:
            return d
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heappush(heap, (nd, v))
    return math.inf


def query_metrics(grid: Grid, g: np.ndarray) -> dict[str, float]:
    u = solve_field(grid, g, grid.source)
    pt = float(abs(u[grid.target]) ** 2)
    pd = float(abs(u[grid.decoy]) ** 2)
    desired_cost = dijkstra_cost(grid, g, grid.source, grid.target)
    decoy_cost = dijkstra_cost(grid, g, grid.source, grid.decoy)
    return {
        "desired_power": pt,
        "decoy_power": pd,
        "desired_over_decoy": pt / max(pd, 1e-30),
        "desired_path_cost": desired_cost,
        "decoy_path_cost": decoy_cost,
        "decoy_over_desired_path_cost": decoy_cost / max(desired_cost, 1e-30),
    }


def train(
    rule: str,
    n: int = 11,
    epochs: int = 70,
    eta: float = 0.045,
    seed: int = 7,
) -> dict:
    grid = make_grid(n)
    g = np.ones(len(grid.edges), dtype=float)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(g))

    initial_grad, _, _, _ = receiver_power_gradient_logg(grid, g)
    initial_q = project_fixed_mean_tangent(g, initial_grad)
    trajectory = [query_metrics(grid, g)]

    for epoch in range(epochs):
        g, _ = material_step(grid, g, rule, permutation, eta=eta)
        if epoch in {0, 1, 3, 7, 15, 31, epochs - 1}:
            m = query_metrics(grid, g)
            m["epoch"] = epoch + 1
            trajectory.append(m)

    final = query_metrics(grid, g)
    log_change = np.log(g)
    corr = float(np.corrcoef(initial_q, log_change)[0, 1]) if np.std(log_change) > 0 else 0.0

    k = max(1, len(g) // 5)
    hi = np.argpartition(initial_q, -k)[-k:]
    lo = np.argpartition(initial_q, k)[:k]
    summary = {
        "rule": rule,
        "n": n,
        "epochs": epochs,
        "eta": eta,
        "baseline": trajectory[0],
        "final": final,
        "material_mean": float(np.mean(g)),
        "material_std": float(np.std(g)),
        "material_min": float(np.min(g)),
        "material_max": float(np.max(g)),
        "initial_functional_to_final_structure_corr": corr,
        "mean_g_initial_positive_fifth": float(np.mean(g[hi])),
        "mean_g_initial_negative_fifth": float(np.mean(g[lo])),
        "trajectory": trajectory,
    }
    return {"grid": grid, "g": g, "summary": summary, "initial_q": initial_q}


def write_svg(path: Path, grid: Grid, g: np.ndarray) -> None:
    scale = 42
    margin = 38
    width = margin * 2 + scale * (grid.n - 1)
    height = width
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#faf8f1"/>',
        '<text x="20" y="24" font-family="monospace" font-size="14">ObjektiYksi Gate 0: persistent coupling after fast state erase</text>',
    ]
    for ge, (i, j) in zip(g, grid.edges):
        x1, y1 = grid.xy[i]
        x2, y2 = grid.xy[j]
        x1 = margin + scale * x1
        x2 = margin + scale * x2
        y1 = margin + scale * y1
        y2 = margin + scale * y2
        if ge >= 1.0:
            color = "#1f6f8b"
        else:
            color = "#b36b3d"
        sw = 0.7 + 2.5 * min(2.5, max(0.15, float(ge)))
        opacity = 0.35 + 0.55 * min(1.0, abs(float(ge) - 1.0) + 0.15)
        parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{sw:.2f}" opacity="{opacity:.3f}" stroke-linecap="round"/>'
        )
    labels = [(grid.source, "S", "#111111"), (grid.target, "T", "#138a36"), (grid.decoy, "D", "#8f1d1d")]
    for idx, label, color in labels:
        x, y = grid.xy[idx]
        x = margin + scale * x
        y = margin + scale * y
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="9" fill="{color}"/>')
        parts.append(f'<text x="{x:.2f}" y="{y+4:.2f}" text-anchor="middle" fill="white" font-family="sans-serif" font-weight="bold" font-size="11">{label}</text>')
    parts.append('<text x="20" y="100%" dy="-12" font-family="sans-serif" font-size="12" fill="#1f6f8b">blue: channel g&gt;=1</text>')
    parts.append('<text x="190" y="100%" dy="-12" font-family="sans-serif" font-size="12" fill="#b36b3d">brown: dam g&lt;1</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def finite_difference_gradient_check(edge_index: int = 37, eps: float = 1e-6) -> dict[str, float]:
    grid = make_grid()
    g = np.ones(len(grid.edges), dtype=float)
    grad_log, _, _, _ = receiver_power_gradient_logg(grid, g)
    # At g=1, d/dlog(g) equals d/dg.
    gp = g.copy()
    gm = g.copy()
    gp[edge_index] += eps
    gm[edge_index] -= eps
    pp = query_metrics(grid, gp)["desired_power"]
    pm = query_metrics(grid, gm)["desired_power"]
    fd = (pp - pm) / (2.0 * eps)
    analytic = float(grad_log[edge_index])
    rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-14)
    return {"edge": edge_index, "analytic": analytic, "finite_difference": fd, "relative_error": rel}


def run_all(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rules = ["phase_reference", "magnitude_reference", "forward_only", "scrambled_reference", "no_write"]
    runs = {rule: train(rule) for rule in rules}
    receipt = {
        "gate": 0,
        "hypothesis": "temporary functional topology can be hardened into persistent routing topology",
        "gradient_check": finite_difference_gradient_check(),
        "runs": {rule: runs[rule]["summary"] for rule in rules},
    }
    (out_dir / "gate0.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    best = runs["phase_reference"]
    write_svg(out_dir / "gate0_topology.svg", best["grid"], best["g"])
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    receipt = run_all(Path(args.out))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
