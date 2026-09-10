# ObjektiYksi

**Object one. Trying to grasp for it.**

This repo tests one concrete version of the idea that emerged from `ResonantNeuron`, `SighImageSuper`, `FunctionalArbors`, `Dig`, `TheMycelialCortex`, and `DendriteAsIteratedFeedbackOperator`:

> **A fast signal can expose a temporary functional topology; a slower material rule can reshape the substrate that generates it; later signals can then expose a different family of effective operators.**

The experiments are deliberately small and attackable. This is not a claim that cortex is an acoustic cavity, that dendrites literally contain water-wave dams, or that a returned signal automatically solves biological credit assignment.

## The object so far

The persistent state is not a bank of learned matrices. It is one shared material vector `g`.

For a source landing coordinate `s`, carrier `omega`, and selected input/output ports, the material is interrogated through

```math
H_g(\omega)=\left[K(g)-\omega^2 I+i\omega\Gamma\right]^{-1},
```

and an address-conditioned effective operator can be written schematically as

```math
M_{\mathrm{eff}}(s,\omega;g)
= P_{\mathrm{out}} H_g(\omega) P_{\mathrm{in}}(s).
```

If `P_in(s)` selects one source this is an output transfer vector; with multiple input and output ports it is a transfer matrix. The important point is the same:

> **`M_eff` is not stored. The shared material `g` generates different effective operators depending on how it is interrogated.**

Gate 6 adds a tiny learned task-to-address selector on top of that shared substrate:

```text
 task q
   ↓
 fast bounded search
 address a = (source position, frequency)
   ↓
 address-conditioned H_g
   ↓
 output / consequence
   ↓
 slow conservative material dither
   ↓
 new g, therefore a changed family of address-conditioned operators
```

So the current object is closer to

```text
fast selector chooses which operator slice is exposed
                    +
slow material changes the family of slices that can be exposed
```

than to a conventional network storing one weight matrix per task.

## What happened

GitHub Actions reproduced Gates 0–6 on Python 3.11 and 3.12.

### Gate 0 — a wave field carves channels and dams

Gate 0 starts from exactly mirror-symmetric material and uses a signed forward × receiver-reference sensitivity as a positive-control write rule. With mean coupling fixed, source-only replay reaches `1366.48x` desired/mirror-decoy and raises absolute desired power `980.94x`.

A source at the left launches `|u> = H_g|s>`. A desired receiver launches a reciprocal reference `|r> = H_g|t>` only in this positive control. For edge incidence vector `b_e`, the exact first-order sensitivity of receiver power is

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

### Gate 1 — road or resonant topology?

`gate1_route_or_resonance.py` mirrors the learned material, sweeps frequency, ranks all right-side ports, and compares wave transfer with ordinary `1/g` shortest paths.

The learned target becomes rank `1/9` and beats the strongest other output by `14.84x`. Mirroring only the persistent material swaps the receiver preference essentially exactly. Yet the ordinary shortest path is cheaper to the decoy (`7.4477`) than to the desired receiver (`8.7864`).

So the learned object is not adequately described as a static conductive road:

> **learned topology = material arrangement + phase + resonance + frequency.**

### Gate 2 — the medium pokes itself

`gate2_local_dither.py` removes both the receiver-launched reference and the analytic edge derivative from the learning rule. A proposal is strictly local:

```text
g_a <- g_a + delta
g_b <- g_b - delta
```

for two edges sharing one node, so total material is conserved exactly. The source-only field is recomputed and one bounded scalar consequence decides whether the perturbation survives.

Across six fixed proposal streams the true consequence reaches median `41.17x` target/mirror-decoy and target rank `1/9` in every seed. The important attack is that the margin over the strongest *other* output is only about `1.04x–1.40x`.

```text
WHERE / WHICH WAY:  random local physical dither
WHETHER TO KEEP:    delayed bounded consequence
MEMORY:             persistent changed material
NEXT COMPUTATION:   same source sees a new H_g
```

### Gate 3 — the output boundary moves

`gate3_multiport_dither.py` replaces the single mirror decoy with whichever output port is currently strongest. The structural rule still receives only one scalar number and is never told which competitor generated it:

```math
U_{multi}=\log(P_{target}+\sigma^2)-\log(\max_{j\ne target}P_j+\sigma^2).
```

Across six proposal streams, target/strongest-other reaches median `2.05223x` (range `1.67773–2.19309x`), absolute target power rises `2.80903x`, and the chosen target ends rank `1/9` in every run. The limiting competitor changes identity roughly sixty times on accepted physical steps.

So local conservative edits plus one scalar consequence can continue improving the material even while the identity of the current output error changes.

### Gate 4 — the object is relational

`gate4_routing_basin.py` makes **no further writes**. It freezes Gate-3 material and varies frequency, source landing position, and both together.

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

The material generalizes much better across carrier frequency than across source coordinate. Gate 3 therefore did **not** carve a destination-independent attractor. At this training budget the learned object is closer to a favorable transfer relation

```math
H_g(t \leftarrow s;\omega)
```

than to a universal arrow pointing toward `t` from anywhere.

This reconnects the repo to the older `Landing Zones` idea: **where a signal enters can be part of the question asked of the operator**, rather than merely nuisance variation.

### Gate 5 — one material, several landing coordinates

Gate 5 attacks the source fragility directly. One material is trained against five source landing positions: center plus its four cardinal neighbors. Every accepted write is still only a local material transfer, and the learner gets only the worst scalar target-vs-strongest-competitor consequence across the five sources. It is not told which source or which competitor caused the worst case.

At the initial `520`-proposal budget the experiment looked close to a wall: median worst trained-source ratio reached only `0.8796x`, so not every trained source was rank-1.

The longer `1600`-proposal confirmation changed that interpretation:

```text
median worst trained-source target / competitor   2.5398x
minimum across the three runs                     2.4830x
median trained-source rank-1 fraction              1.0000
median held-out diagonal rank-1 fraction           0.7500
median full 3x3-neighborhood rank-1 fraction       0.8889
median worst-source switches                       122
median worst-competitor switches                   125
```

So much of Gate 4's source-coordinate fragility was an optimization-budget effect rather than a demonstrated hard incompatibility. A single conserved material can be sculpted so that several source-conditioned relations are useful at once, even while the currently limiting input and currently limiting output repeatedly change.

This is not destination invariance: held-out positions still fail sometimes. But it is stronger than one learned route. The substrate can support a **family of jointly useful addressed transfer relations**.

### Gate 6 — learn the address while the material learns

Gate 6 asks the next question directly: if `(source position, frequency)` behaves like an address into the operator family, can that address itself be learned while the shared material is changing?

There are three receiver tasks and nine possible physical addresses: three source landing rows × three carrier frequencies.

The strongest version, `gate6_online_selector.py`, starts with **no task-to-address information**. The fast selector must sample addresses using only scalar task utility. Exact scans of all nine addresses are audit-only and never enter either the selector or the slow material acceptance rule.

At the same time, the shared material continues changing through local conservative keep/revert dithers judged by one worst-task consequence at the currently selected addresses.

Across three seeds:

```text
blank-material exact-oracle worst-task ratio       1.1488x
coadaptive learned-address + learned-material       1.7492x median
worst coadaptive run                                1.6407x
final task rank-1 fraction                          1.0000
selector / audit-oracle exact agreement             1.0000
median distinct final addresses                     3
fraction of address table actually explored         1.0000
material-sum relative error                         <= 1.3e-16
```

The controls separate the jobs:

```text
selector only, material frozen
  worst-task ratio                                  1.1488x
  final rank-1 fraction                             1.0000

one stale center / omega=1.55 address, material learns
  worst-task ratio                                  0.9989x
  final rank-1 fraction                             0.3333
```

So the combined machine does something neither component does alone:

> **a fast learner discovers which physical operator slice to expose, while slow local material learning improves the shared substrate from which all slices are generated.**

Gate 6 also attacked the tempting "moving address target" interpretation. In `gate6_moving_target.py`, inactive tasks are damaged during a median `42/45` epochs, showing that their transfer values really do move as other tasks reshape the substrate. Nevertheless the identity of the audit-best address does not move in these runs: median oracle address switches are `0`.

That negative result matters. In this toy, `(position, frequency)` behaves less like a drifting weight and more like a **relatively stable coordinate chart over a deforming operator family**.

Schematically, define

```math
\mathcal{M}_g = \left\{M_{\mathrm{eff}}(s,\omega;g)\right\}_{s,\omega}.
```

Slow learning deforms

```math
\mathcal{M}_g \longrightarrow \mathcal{M}_{g'},
```

while the fast selector can continue navigating the family by address.

That is the current object.

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
python gate5_multisource_basin.py --out results --seeds 4
python gate5_long_confirm.py --out results --seeds 3 --attempts 1600
python gate6_learned_addressing.py --out results --seeds 3 --epochs 20 --slow-proposals 24
python gate6_moving_target.py --out results --seeds 3 --epochs 45 --slow-proposals 36
python gate6_online_selector.py --out results --seeds 3 --epochs 30 --slow-proposals 24
```

The experiment scripts write their numerical receipts into `results/`. GitHub Actions also uploads the reproduced Gates 0–6 receipts as workflow artifacts for Python 3.11 and 3.12.

Committed result notes currently cover Gates 0–4:

- [`GATE0_1_RESULTS.md`](GATE0_1_RESULTS.md)
- [`GATE2_RESULTS.md`](GATE2_RESULTS.md)
- [`GATE3_RESULTS.md`](GATE3_RESULTS.md)
- [`GATE4_RESULTS.md`](GATE4_RESULTS.md)

## Claim boundary

Gates 0–6 establish only a computational-physics result in this deterministic toy.

A reciprocal resonant medium can be persistently restructured by strictly local, material-conserving physical perturbations retained or reverted using bounded scalar consequences. One shared material can support several useful source-conditioned relations. A tiny fast selector, starting without task-to-address knowledge, can learn which `(source position, frequency)` address exposes a useful transfer while that same shared material is slowly rewritten.

The current evidence therefore supports this description:

> **One persistent substrate can generate a family of address-conditioned effective operators. Fast learning can select among those operators while slower local learning deforms the shared substrate that generates them.**

It does **not** establish autonomous goals, semantic representations, a new universal learning algorithm, biological credit assignment, dendritic growth, literal fluid dams, or a brain mechanism. The selector is deliberately engineered, the receiver tasks and scalar utilities are externally defined, and exact address scans are used only for auditing.

## Next attack

The next useful failure test is no longer "can an address be learned?" Gate 6 says yes in this toy.

The harder question is whether the coordinate system remains useful under **larger and less friendly deformations of the shared substrate**.

One direct Gate 7 would deliberately force best addresses to move: enlarge the address set, use partially conflicting tasks/frequencies, increase structural drift, and limit how often the fast selector is allowed to re-probe. Then measure whether the two-timescale system tracks the moving operator family or catastrophically chases stale coordinates.

A second direction is more architectural: remove the explicit per-task lookup table and let the task/query itself generate or search an address. That would move the project from

```text
learned task -> address table + self-carving substrate
```

toward

```text
query -> physical address selection -> addressed operator -> consequence -> substrate rewrite
```

without pretending that the present Gate 6 has already solved that problem.
