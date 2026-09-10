# ObjektiYksi

**Object one. Trying to grasp for it.**

This repo tests one concrete version of the idea that emerged from `ResonantNeuron`, `SighImageSuper`, `FunctionalArbors`, `Dig`, `TheMycelialCortex`, and `DendriteAsIteratedFeedbackOperator`:

> **A fast signal can define a temporary functional topology; a slower material rule can harden part of that topology; the changed topology then changes the next signal.**

The experiments are deliberately small and attackable. This is not a claim that cortex is an acoustic cavity, that dendrites literally contain water-wave dams, or that a returned signal automatically solves biological credit assignment.

## What happened

GitHub Actions reproduced Gates 0–3 on Python 3.11 and 3.12.

Gate 0 starts from exactly mirror-symmetric material and uses a signed forward × receiver-reference sensitivity as a positive-control write rule. With mean coupling fixed, source-only replay reaches `1366.48x` desired/mirror-decoy and raises absolute desired power `980.94x`.

Gate 1 shows this is not adequately described as a static conductive road: after carving, ordinary `1/g` shortest-path cost actually favors the decoy while the wave operator strongly favors the desired receiver. The learned object is frequency-dependent resonant topology.

Gate 2 removes the handed analytic sign. Two adjacent edges exchange a small amount of coupling; the fast field is re-solved from the source alone; one bounded scalar consequence decides keep/revert. Against one named mirror decoy, six proposal streams reach median `41.17x` target/decoy and target rank `1/9` in every seed. But the target's margin over the strongest *other* output is only about `1.04x–1.40x`.

Gate 3 attacks exactly that weakness. Its scalar consequence is

```math
U_{multi}=\log(P_{target}+\sigma^2)-\log(\max_{j\ne target}P_j+\sigma^2).
```

The structural rule is never told which competitor is strongest. Across six fixed proposal streams:

```text
rule                       target / strongest other   target gain   target rank
true multiport consequence      median 2.05223x          2.80903x     1/9 all seeds
Gate-2 two-port control          median 1.22740x          3.32306x     not all rank 1
random accept                    median 0.58733x          0.80799x     median 6/9
inverted multiport               median 1.04e-5           2.80e-5x    median 9/9
no write                                0.737698x          1.00000x     6/9
```

The true multiport result ranges from `1.67773x` to `2.19309x` target/strongest-other, while absolute target transfer rises rather than disappearing into the silence trap. The active competitor changes identity a median **59 times** on accepted structural steps. So the medium is not merely learning to suppress one fixed rival: the scalar boundary moves as different ports become limiting.

See [`GATE0_1_RESULTS.md`](GATE0_1_RESULTS.md), [`GATE2_RESULTS.md`](GATE2_RESULTS.md), and [`GATE3_RESULTS.md`](GATE3_RESULTS.md).

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

`gate2_local_dither.py` removes both the receiver-launched reference and the analytic edge derivative from the learning rule.

A proposal is strictly local:

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

`gate3_multiport_dither.py` replaces the single mirror decoy with the strongest of all other output ports. The rule still receives only one number after each local material experiment. It does not receive the identity of the current competitor.

That competitor changes repeatedly during learning, yet all six runs end with the chosen target ranked first. The two-port control does not reliably do that. This turns the Gate-2 result into a genuine multiport routing result in the toy.

The useful conceptual change is small but important:

```text
not: carve a road around one known obstacle

but:
    perturb local material
          ↓
    ask the whole output boundary one scalar question
          ↓
    whichever competitor is currently limiting defines the consequence
          ↓
    retain only useful physical changes
          ↓
    the limiting boundary moves
          ↓
    persistent topology settles into a receiver-selective configuration
```

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
```

## Claim boundary

Gates 0–3 establish only this computational-physics statement in the toy:

> **A reciprocal resonant medium can be persistently restructured by signal-related interaction. A useful write direction need not be supplied analytically: strictly local, material-conserving physical perturbations can be retained or rejected using only a bounded scalar consequence, and that scalar can enforce a chosen receiver against a moving set of competitors. The resulting material changes later source-only transfer after all fast state is erased.**

They do **not** establish autonomous goals, biological credit assignment, dendritic growth, literal fluid dams, or a long-delay eligibility mechanism.

The next attack is generalization. Train at one carrier and one source location, freeze the material, then query nearby frequencies and nearby source positions without further writing. If the preference collapses immediately, Gate 3 carved a narrow resonant trick. If a useful target rank survives a finite frequency/source neighborhood, the topology has a measurable routing basin.
