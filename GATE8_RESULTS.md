# Gate 8 — Crossed spatial map: attack the native-geometry explanation

Gate 7 killed the joint position-frequency interpretation of Gate 6. Source position alone was sufficient for the three tasks. That left an easier explanation:

> perhaps `upper`, `center`, and `lower` are merely a fixed geometric multiplexer already built into the blank lattice.

Gate 8 attacks that explanation directly.

## Preregistered task

Frequency is fixed at

```text
omega = 1.35
```

and the three source landing positions are tied permanently to three tasks. There is **no selector** that can relabel them.

The native control is

```text
upper source  -> upper target
center source -> center target
lower source  -> lower target
```

The hard mapping is a 3-cycle:

```text
upper source  -> lower target
center source -> upper target
lower source  -> center target
```

Every source therefore has to beat its native spatial preference.

For each fixed source-target relation the bounded utility is

```math
U_i=\log(P_{t_i}+\sigma^2)-\log(\max_{j\ne t_i}P_j+\sigma^2).
```

The material learner receives only

```math
U=\min_i U_i.
```

It is not told which source-target relation produced the minimum, which output was the strongest competitor, or any edge gradient. A proposal moves a fixed quantum of material between two adjacent edges, conserving total material.

All modes use the same mapping-independent observer floor and, for each seed, the same physical proposal schedule.

## Preregistered success rule

The crossed mapping counted as programmable only if all of the following held:

```text
median crossed rank-1 fraction          = 3/3
median crossed worst-relation ratio    >= 1.10x
worst seed crossed ratio                > 1.00x
crossed median / random-drift median    >= 1.10x
```

If the identity map passed while the crossed map failed, the surviving interpretation would have been reduced toward a native geometric multiplexer.

## Result

Both focused Python 3.11 and 3.12 CI runs produced the same deterministic result.

```text
                         blank worst      final worst      final rank-1
CROSSED MAP                 0.109779         1.522495          3/3
IDENTITY MAP                1.148829         1.781158          3/3
RANDOM CROSSED DRIFT        0.109779         0.067250          0/3
NO-WRITE CROSSED            0.109779         0.109779          0/3
```

Across the three crossed seeds:

```text
seed 0    final worst ratio    1.522495
seed 1    final worst ratio    1.717478
seed 2    final worst ratio    1.362632
```

Every crossed relation is rank-1 in every seed.

The median crossed result improves the blank crossed worst relation by

```text
1.522495 / 0.109779 = 13.87x
```

and exceeds the random-drift median by about

```text
1.522495 / 0.067250 = 22.64x.
```

The conservative learner accepts about `31.7%` of proposed physical transfers in the median crossed run. The identity of the current worst relation changes a median `90` times and the identity of its strongest competitor changes a median `95` times. Total material is conserved to at worst `2.58e-16` relative error.

## Preregistered verdict

```text
crossed_success = true
identity_success = true

verdict = crossed_permutation_survives_native_geometry_attack
```

So the simplest native-geometry explanation does **not** survive this gate.

The blank lattice strongly rejects the crossed mapping: its worst requested transfer is only about `0.11x` its strongest competitor and none of the three requested outputs is rank-1. Local consequence-selected restructuring nevertheless produces one shared material state in which all three deliberately non-native source-to-output relations are simultaneously rank-1.

This is stronger than Gate 6/7's stable address chart. The substrate is not limited to amplifying the transfer relation already favored by source position.

## What Gate 8 earns

Within this deterministic resonant lattice:

> **One conserved physical substrate can be reconfigured, using local material transfers plus one worst-case scalar consequence, to implement a non-native three-relation spatial permutation.**

The persistent object is still one material vector `g`; there are not three independently stored route matrices.

This makes the useful picture narrower but stronger:

```text
source coordinate
      +
shared learned material
      ->
address-conditioned transfer relation
```

where the material can change the relation itself rather than merely strengthening a fixed geometric correspondence.

## What Gate 8 does not earn

Gate 8 does **not** establish:

- arbitrary permutation capacity as the number of ports grows;
- a universal self-configuring wave computer;
- efficient learning compared with digital optimization;
- biological dendritic learning;
- that frequency is useful for the present tasks (Gate 7 says it is not needed here);
- autonomous task discovery or semantics;
- a mechanism for biological credit assignment.

The task is still externally specified and the consequence is evaluated globally after every candidate physical perturbation.

## Next kill

The strongest remaining objection from Gate 6/7 is now more precise:

> material can rewrite the source-to-output map, but can learning make the **best address itself move**, or are useful input coordinates always fixed before learning?

A next gate should begin with an intentionally wrong or ambiguous address optimum, hold the task fixed, and require slow material deformation to move the audit-best input location from one physical address to another. The decision must be preregistered before looking at the trajectory.

If best-address movement still refuses to occur, the right description is a programmable transfer substrate with comparatively stable input coordinates—not a co-adapting coordinate manifold.

If it does move reproducibly, then the stronger claim becomes testable: slow learning can deform not only transfer values but the **coordinate chart of the operator family itself**.

## Claim boundary

Gate 8 is a controlled computational-physics result in an engineered reciprocal lattice. It demonstrates non-native spatial remapping under bounded scalar consequence and local conservative structural exploration. It is evidence against one trivial explanation of Gates 6-7, not evidence that cortical dendrites implement this exact mechanism.
