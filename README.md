# ObjektiYksi

**Object one. Trying to grasp for it.**

This repo tests one concrete version of the idea that emerged from `ResonantNeuron`, `SighImageSuper`, `FunctionalArbors`, `Dig`, `TheMycelialCortex`, and `DendriteAsIteratedFeedbackOperator`:

> **A fast signal can define a temporary functional topology; a slower material rule can harden part of that topology; the changed topology then changes the next signal.**

The experiments are deliberately small and attackable. This is not a claim that cortex is an acoustic cavity, that dendrites literally contain water-wave dams, or that a returned signal automatically solves biological credit assignment.

## What happened

GitHub Actions reproduced Gates 0–4 on Python 3.11 and 3.12.

Gate 0 starts from exactly mirror-symmetric material and uses a signed forward × receiver-reference sensitivity as a positive-control write rule. With mean coupling fixed, source-only replay reaches `1366.48x` desired/mirror-decoy and raises absolute desired power `980.94x`.

Gate 1 shows this is not adequately described as a static conductive road: after carving, ordinary `1/g` shortest-path cost actually favors the decoy while the wave operator strongly favors the desired receiver. The learned object is frequency-dependent resonant topology.

Gate 2 removes the handed analytic sign. Two adjacent edges exchange a small amount of coupling; the fast field is re-solved from the source alone; one bounded scalar consequence decides keep/revert. Against one named mirror decoy, six proposal streams reach median `41.17x` target/decoy and target rank `1/9` in every seed. But the target's margin over the strongest *other* output is only about `1.04x–1.40x`.

Gate 3 attacks exactly that weakness. It compares the chosen target with whichever output is currently strongest, while never telling the structural rule which competitor that is. Across six fixed proposal streams, target/strongest-other reaches median `2.05223x` (range `1.67773–2.19309x`), target power rises `2.80903x`, and the target ends rank `1/9` in all six runs. The limiting competitor changes about sixty times on accepted physical steps.

Gate 4 freezes those learned materials and moves the query. The result is sharply asymmetric:

```text
held-out frequency, source fixed
  median target rank-1 fraction       0.71875
  median target / strongest other     1.58396x
  median contiguous rank-1 width      0.2875 omega units

held-out nearby source, omega fixed
  median target rank-1 fraction       0.31250
  median target / strongest other     0.61953x

held-out source + frequency jointly
  median target rank-1 fraction       0.43182
  median target / strongest other     0.86578x
```

So Gate 3 did **not** carve a destination-independent attractor. It carved something closer to a **source-conditioned transfer relation**. The material tolerates substantial carrier-frequency change at the trained landing point, but a one-site source displacement often changes the effective topology enough to lose the target preference.

See [`GATE0_1_RESULTS.md`](GATE0_1_RESULTS.md), [`GATE2_RESULTS.md`](GATE2_RESULTS.md), [`GATE3_RESULTS.md`](GATE3_RESULTS.md), and [`GATE4_RESULTS.md`](GATE4_RESULTS.md).

## Gate 0 — a wave field carves channels and dams

We use a damped frequency-domain wave network on a 2-D lattice. Each lattice edge has a positive coupling / permeability `g_e`. For one angular frequency,

```math
H_g(\omega)=\left[K(g)-\omega^2 I+i\omega\Gamma\right]^{-1}.
```

A source at the left launches `|u> = H_g|s>`. A desired receiver launches a reciprocal reference `|r> = H_g|t>` only in the Gate-0 positive control. For edge incidence vector `b_e`, the exact first-order sensitivity of receiver power is

```math
\frac{\partial |y|^2}{\partial g_e}
=-2\,\mathrm{Re}\left[y^*(b_e^T r)(b_e^T u)\right].
```

The slow material update preserves the global material budget. Positive sensitivity becomes operational **channels**; negative sensitivity becomes **dams**. After the write, all fast field state is erased and the source is queried again.

```text
fixed material
    ↓
wave interaction
    ↓
temporary functional edge pattern
    ↓
slow material deformation
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

Mirroring only the persistent material swaps the receiver preference essentially exactly. Yet the cheapest static path goes to the decoy. What was carved is therefore not just a wire-like road; the function is in the material arrangement as seen through the resonant operator.

## Gate 2 — the medium pokes itself

`gate2_local_dither.py` removes both the receiver-launched reference and the analytic edge derivative from the learning rule. A proposal is strictly local:

```text
g_a <- g_a + delta
g_b <- g_b - delta
```

for two edges sharing one node, so total material is conserved exactly. The source-only field is recomputed and a bounded scalar decides whether the perturbation survives.

```text
WHERE / WHICH WAY:  random local physical dither
WHETHER TO KEEP:    delayed bounded consequence
MEMORY:             persistent changed material
NEXT COMPUTATION:   same source sees a new H_g
```

## Gate 3 — the boundary moves

`gate3_multiport_dither.py` replaces the single mirror decoy with the strongest of all other output ports. The structural rule still receives only one number after each local material experiment. It does not receive the identity of the current competitor.

```math
U_{multi}=\log(P_{target}+\sigma^2)-\log(\max_{j\ne target}P_j+\sigma^2).
```

The competitor changes repeatedly during learning, yet all six runs end with the chosen target ranked first. The matched two-port control does not reliably do that.

## Gate 4 — the object is relational

`gate4_routing_basin.py` makes **no further writes**. It evaluates each frozen Gate-3 material across a frequency sweep, a 3×3 source-position neighborhood, and their joint perturbations.

The learned routing generalizes much better over frequency than over source coordinate. This matters conceptually: the persistent material alone is not the whole functional topology.

```text
material + source coordinate + carrier + receiver
                    ↓
              effective transfer
```

In Green-function language, the learned object is closer to a favorable relation

```math
H_g(t \leftarrow s;\omega)
```

than a universal arrow pointing toward `t` from anywhere. Gate 4 therefore reconnects this repo to the older `Landing Zones` idea: **where a signal enters can be part of what the signal means**, not merely nuisance variation.

## What counts as a dam

A nodal line is not automatically a wall. `Dam` is operational here: an edge whose persistent coupling is driven below the material mean. A `channel` is an edge driven above it. The actual routing property remains frequency-dependent and is not equivalent to ordinary graph connectivity.

## Run

```bash
python -m pip install -r requirements.txt
pytest -q
python self_carving.py --out results
python gate1_route_or_resonance.py --out results
python gate2_local_dither.py --out results --seeds 6
python gate3_multiport_dither.py --out results --seeds 6
python gate4_routing_basin.py --out results --seeds 6
```

## Claim boundary

Gates 0–4 establish only this computational-physics statement in the toy:

> **A reciprocal resonant medium can be persistently restructured by signal-related interaction. Useful material changes can be discovered through strictly local, material-conserving physical perturbations plus one bounded scalar consequence, without an analytic edge gradient. The resulting topology can enforce a receiver against moving output competitors and has a measurable off-training frequency basin, but it remains strongly conditioned on the source landing coordinate.**

They do **not** establish autonomous goals, biological credit assignment, dendritic growth, literal fluid dams, or a long-delay eligibility mechanism.

The next attack is whether source specificity is merely under-training or is intrinsic to the addressing scheme. Gate 5 should train one material against a set of source landing points using a single worst-case scalar consequence, then test unseen source positions. If this creates a held-out spatial basin, the medium can compile an invariant route. If it does not, the stronger interpretation is that the medium naturally stores **relations between coordinates**, not destination labels.
