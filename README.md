# ObjektiYksi

**Object one. Trying to grasp for it.**

This repo tests one concrete version of the idea that emerged from `ResonantNeuron`, `SighImageSuper`, `FunctionalArbors`, `Dig`, `TheMycelialCortex`, and `DendriteAsIteratedFeedbackOperator`:

> **A fast signal can define a temporary functional topology; a slower material rule can harden part of that topology; the changed topology then changes the next signal.**

The first experiment is deliberately small and attackable. It is not a claim that cortex is an acoustic cavity, that dendrites literally contain water-wave dams, or that a returned signal automatically solves biological credit assignment.

## What happened

GitHub Actions reproduced Gates 0–1 on Python 3.11 and 3.12. Starting from an exactly mirror-symmetric material (`desired/decoy = 1`), the signed forward × receiver-reference rule redistributes a **fixed mean coupling budget**. After fast state is erased, the identical source-only query gives:

```text
desired power / baseline       980.94x
desired / mirror decoy        1366.48x
target / strongest other port   14.84x
```

Mirroring only the learned material swaps the preference exactly (`ratio × mirrored ratio = 1.000000000000011`). Yet a static `1/g` shortest-path calculation actually finds the decoy path cheaper than the desired path. So the learned object is **not a simple conductive road**; its function lives in the frequency-dependent wave operator.

The controls also catch the silence trap. Magnitude-only reference reaches `74.41x` target/decoy while making the target itself slightly weaker; forward-only nearly silences both ports and remains symmetric. The signed phase-sensitive rule is the one that simultaneously increases absolute target transfer and specificity.

See [`GATE0_1_RESULTS.md`](GATE0_1_RESULTS.md) and the compact frozen [`results/gate0_1_summary.json`](results/gate0_1_summary.json).

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

That expression is useful here because it can be read as a **forward field meeting a receiver-launched reference field**. Gate 0 uses it as a positive-control write rule. It is not presented as an emergent biological learning rule.

The slow material update acts in log-coupling space and keeps the mean coupling fixed:

```math
\log g_e \leftarrow \log g_e + \eta\,\widehat q_e,
\qquad
\langle g\rangle=1.
```

So the system cannot win by simply making every edge more conductive. Positive parts of the interference sensitivity become **channels**; negative parts become **dams**. Then the field is solved again through the changed material.

The loop is therefore

```text
fixed material
    ↓
source + receiver-reference fields
    ↓
temporary functional edge pattern q_e
    ↓
slow redistribution of edge permeability
    ↓
new K(g), therefore new H_g
    ↓
state erased
    ↓
source-only query traverses a changed topology
```

This is the missing reverse arrow from the older resonator work:

```text
ResonantNeuron:       TOPOLOGY -> SIGNAL
ObjektiYksi Gate 0:   TOPOLOGY <-> SIGNAL
```

## Why a symmetric task

The source sits on the left. Two readout ports are mirror-symmetric on the right. Before carving, symmetry forces equal transfer to the desired port and the decoy. The desired port is used only as the reference during writing. After writing, **all fast state is discarded** and the source is queried alone.

The strongest observable is therefore not raw amplitude but whether the persistent material now breaks the original symmetry:

```math
R = \frac{|u_{\rm desired}|^2}{|u_{\rm decoy}|^2}.
```

At baseline `R ≈ 1`. A successful persistent write should make `R > 1` without changing the source or readout geometry.

## Controls

`self_carving.py` runs the same fixed material budget under five rules:

- `phase_reference` — signed forward × reciprocal-reference sensitivity above;
- `magnitude_reference` — keeps overlap magnitude but throws away phase/sign;
- `forward_only` — uses only local forward field energy;
- `scrambled_reference` — spatially permutes the signed reference pattern;
- `no_write` — frozen material.

The point is not that the positive control must be biologically plausible. The point is to ask a prior question first:

> **Can a distributed wave interaction define a spatial pattern which, when slowly hardened under a fixed material budget, becomes a persistent topology that changes later routing?**

If that fails, there is no reason to argue about biological implementations.

## Gate 1 — road or resonant topology?

`gate1_route_or_resonance.py` attacks the obvious interpretation that the material merely carved a high-conductance road. It mirrors the learned material, sweeps frequency, ranks all right-side ports, and compares the wave result with ordinary `1/g` shortest paths.

The mirror test says the persistent material really contains the receiver preference. The shortest-path test says **ordinary graph distance does not explain it**. The frequency sweep says the result is a transfer property of the resonant operator.

## What counts as a dam

A nodal line is not automatically a wall. This repo therefore uses `dam` operationally: an edge whose persistent coupling has been driven substantially below the material mean. Likewise a `channel` is an edge driven above the mean. The code measures both rather than calling every low-amplitude wave line a topological barrier.

## Run

```bash
python -m pip install -r requirements.txt
pytest -q
python self_carving.py --out results
python gate1_route_or_resonance.py --out results
```

The experiments write:

```text
results/gate0.json
results/gate0_topology.svg
results/gate1_route_or_resonance.json
```

The SVG is generated directly by the experiment, with no plotting dependency.

## Claim boundary

Gates 0–1 establish only this computational-physics statement in the toy:

> **A reciprocal wave interaction can define a spatial write field which is converted into a slow, resource-constrained material deformation; after fast state is erased, that persistent deformation changes a later receiver-selective wave transfer, and the result is not reducible to a static shortest conductive path.**

They do **not** establish that real dendrites implement this rule, that wave nodes literally become anatomical walls, that the reference signal is a biological adjoint, or that the rule solves free structural credit assignment.

The next gate removes the analytically signed positive-control rule and asks whether a strictly local stateful medium can discover an equivalent carving direction by physical dither + delayed consequence.