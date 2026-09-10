# ObjektiYksi

**Object one. Trying to grasp for it.**

This repo tests one concrete version of the idea that emerged from `ResonantNeuron`, `SighImageSuper`, `FunctionalArbors`, `Dig`, `TheMycelialCortex`, and `DendriteAsIteratedFeedbackOperator`:

> **A fast signal can define a temporary functional topology; a slower material rule can harden part of that topology; the changed topology then changes the next signal.**

The experiments are deliberately small and attackable. This is not a claim that cortex is an acoustic cavity, that dendrites literally contain water-wave dams, or that a returned signal automatically solves biological credit assignment.

## What happened

GitHub Actions reproduced Gates 0–2 on Python 3.11 and 3.12.

Gate 0 starts from exactly mirror-symmetric material and uses a signed forward × receiver-reference sensitivity as a positive-control write rule. With mean coupling fixed, the source-only replay reaches `1366.48x` desired/mirror-decoy and raises absolute desired power `980.94x`.

Gate 1 then shows that this is not adequately described as a static conductive road: after carving, ordinary `1/g` shortest-path cost actually favors the decoy while the wave operator strongly favors the desired receiver. The learned object is frequency-dependent resonant topology.

Gate 2 removes the handed analytic sign entirely. It picks two edges sharing one lattice node, transfers a small amount of coupling locally between them, re-solves the fast field from the source alone, receives one bounded scalar consequence, and either keeps or reverts the perturbation. Across six fixed proposal streams:

```text
rule                    median target/decoy   median target gain   median rank
true consequence              41.1746x              2.2091x          1 / 9
random accept                   1.1882x              0.9635x          4 / 9
inverted consequence            0.02136x             0.04243x         9 / 9
no write                        1.0000x              1.0000x          6 / 9
```

Every true-consequence seed put the target first. Total structural material is conserved to `2.58e-16` relative error. The strongest caveat is equally important: target / strongest-other is only about `1.04x–1.40x`, because Gate 2's scalar consequence explicitly names one mirror decoy. Gate 3 therefore attacks **multiport** selectivity instead of celebrating the `41x` two-port number.

See [`GATE0_1_RESULTS.md`](GATE0_1_RESULTS.md), [`GATE2_RESULTS.md`](GATE2_RESULTS.md), and the compact receipts in [`results/`](results/).

## Gate 0 — a wave field carves channels and dams

We use a damped frequency-domain wave network on a 2-D lattice. Each lattice edge has a positive coupling / permeability `g_e`. For one angular frequency,

```math
H_g(\omega)
=\left[K(g)-\omega^2 I+i\omega\Gamma\right]^{-1}.
```

A source at the left launches a field

```math
|u\rangle = H_g|s\rangle.
```

A receiver/reference at the desired port launches the reciprocal field

```math
|r\rangle = H_g|t\rangle.
```

For edge incidence vector `b_e`, changing one edge coupling gives the exact first-order sensitivity of receiver power

```math
\frac{\partial |y|^2}{\partial g_e}
=-2\,\mathrm{Re}\left[
 y^* (b_e^T r)(b_e^T u)
\right],
\qquad y=\langle t|u\rangle.
```

Gate 0 uses that expression as a positive-control write rule. It is not presented as an emergent biological learning rule.

The slow material update acts in log-coupling space and keeps the mean coupling fixed, so the system cannot win by simply making every edge more conductive. Positive parts of the sensitivity become operational **channels**; negative parts become **dams**. Then the field is solved again through the changed material.

```text
fixed material
    ↓
source + receiver-reference fields
    ↓
temporary functional edge pattern
    ↓
slow redistribution of edge permeability
    ↓
new K(g), therefore new H_g
    ↓
fast state erased
    ↓
source-only query traverses changed topology
```

This is the missing reverse arrow from the older resonator work:

```text
ResonantNeuron:       TOPOLOGY -> SIGNAL
ObjektiYksi:          TOPOLOGY <-> SIGNAL
```

## Gate 1 — road or resonant topology?

`gate1_route_or_resonance.py` mirrors the learned material, sweeps frequency, ranks all right-side ports, and compares wave transfer with ordinary `1/g` shortest paths.

Mirroring only the persistent material swaps the preference essentially exactly. Yet the cheapest static path goes to the decoy. So what was carved is not just a wire-like road; the function is in the material arrangement as seen through the resonant operator.

## Gate 2 — the medium pokes itself

`gate2_local_dither.py` removes both the receiver-launched reference field and the analytic per-edge derivative from the learning rule.

A proposal is strictly local: two couplings that meet at one node exchange a fixed amount of material,

```text
g_a <- g_a + delta
g_b <- g_b - delta
```

so `sum(g)` is exactly conserved. After the perturbation, the source-only field is recomputed and the observer emits one scalar:

```math
U = \log(P_{target}+\sigma^2)-\log(P_{decoy}+\sigma^2).
```

The observer floor prevents the silence trick: if both outputs vanish, utility tends back toward zero. `true_consequence` keeps the local structural experiment only when this scalar improves.

This is the decomposition we wanted to test:

```text
WHERE / WHICH WAY:  random local physical dither
WHETHER TO KEEP:    delayed bounded consequence
MEMORY:             persistent changed material
NEXT COMPUTATION:   same source sees a new H_g
```

It works in the two-port task without ever calculating `dU/dg`.

## Why a symmetric task

The source sits on the left. Two readout ports are mirror-symmetric on the right. Before carving, symmetry forces equal transfer to the desired port and the mirror decoy. After writing, all fast state is discarded and the source is queried alone.

The Gate-0/2 two-port observable is

```math
R = \frac{|u_{\rm desired}|^2}{|u_{\rm decoy}|^2}.
```

Gate 3 will deliberately stop relying on a single named decoy.

## What counts as a dam

A nodal line is not automatically a wall. This repo therefore uses `dam` operationally: an edge whose persistent coupling has been driven substantially below the material mean. Likewise a `channel` is an edge driven above the mean. The code measures persistent coupling changes rather than calling every low-amplitude wave line a topological barrier.

## Run

```bash
python -m pip install -r requirements.txt
pytest -q
python self_carving.py --out results
python gate1_route_or_resonance.py --out results
python gate2_local_dither.py --out results --seeds 6
```

## Claim boundary

Gates 0–2 establish only this computational-physics statement in the toy:

> **A reciprocal resonant medium can be persistently restructured by signal-related interaction; moreover, the useful write direction need not be supplied analytically—local conservative physical dithers can be retained or rejected using only a bounded scalar consequence, so that later source-only transfer changes after all fast state is erased.**

They do **not** establish autonomous goals, biological credit assignment, dendritic growth, literal fluid dams, or a long-delay eligibility mechanism.

The next gate replaces the named mirror decoy with the strongest competing output port. If the same local-dither mechanism can make one chosen receiver robustly dominate an entire output strip, the self-carving result becomes multiport routing rather than two-port discrimination.
