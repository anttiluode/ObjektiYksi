# ObjektiYksi

**Object one. Trying to grasp for it.**

This repo tests one concrete version of the idea that emerged from `ResonantNeuron`, `SighImageSuper`, `FunctionalArbors`, `Dig`, `TheMycelialCortex`, and `DendriteAsIteratedFeedbackOperator`:

> **A fast signal can define a temporary functional topology; a slower material rule can harden part of that topology; the changed topology then changes the next signal.**

The first experiment is deliberately small and attackable. It is not a claim that cortex is an acoustic cavity, that dendrites literally contain water-wave dams, or that a returned signal automatically solves biological credit assignment.

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

## What counts as a dam

A nodal line is not automatically a wall. This repo therefore uses `dam` operationally: an edge whose persistent coupling has been driven substantially below the material mean. Likewise a `channel` is an edge driven above the mean. The code measures both rather than calling every low-amplitude wave line a topological barrier.

## Run

```bash
python -m pip install -r requirements.txt
python self_carving.py --out results
pytest -q
```

The experiment writes:

```text
results/gate0.json
results/gate0_topology.svg
```

The SVG is generated directly by the experiment, with no plotting dependency.

## Claim boundary

A positive Gate 0 would establish only this computational-physics statement:

> **In this reciprocal wave-network toy, a forward/reference interaction can be converted into a slow, resource-constrained material deformation which persists after fast state is erased and changes a later source-only transfer.**

It would **not** establish that real dendrites implement this rule, that wave nodes literally become anatomical walls, that the reference signal is a biological adjoint, or that the rule solves free structural credit assignment. Those remain separate questions.

The next gate, if Gate 0 survives controls, is to remove the analytically signed positive-control rule and ask whether a strictly local stateful medium can discover an equivalent carving direction by physical dither + delayed consequence.