# Gate 7 — Address ablation: kill the joint position-frequency story

Gate 6 gave the selector nine possible physical addresses:

```text
3 source landing rows x 3 carrier frequencies = 9 addresses
```

but its exact blank-material and final oracles selected the same frequency, `omega = 1.35`, for all three tasks. Gate 7 therefore asks the deliberately destructive question:

> **Did Gate 6 actually need joint source-position and frequency addressing, or was task identity already carried by source position alone?**

## Preregistered comparison

Three address spaces were fixed before the result was read:

```text
FULL
    source_dr in {-1, 0, +1}
    omega     in {1.35, 1.55, 1.75}
    9 addresses

POSITION ONLY
    source_dr in {-1, 0, +1}
    omega     = 1.35
    3 addresses

FREQUENCY ONLY
    source_dr = 0
    omega     in {1.35, 1.55, 1.75}
    3 addresses
```

All three conditions use the same bounded observer floor, computed once from the full blank-material Gate-6 address space, and the same slow material proposal schedule per seed. The selector begins with no address utilities, samples unseen addresses first, then uses bounded epsilon-greedy scalar utility probes. Slow learning uses only local material-conserving edge transfers retained when the currently selected worst-task scalar consequence improves.

The match criterion was fixed before reading the result:

> An ablation matches `FULL` only if it reaches the same median rank-1 fraction **and** at least `90%` of `FULL`'s median final worst-task target/strongest-competitor ratio.

The four possible outcomes were also fixed in advance: both dimensions individually sufficient, position-dominated, frequency-dominated, or a genuine joint-space advantage.

## Result

Both Python 3.11 and 3.12 reproduced the same deterministic receipt.

```text
                         FULL       POSITION ONLY     FREQUENCY ONLY
addresses                   9              3                 3

blank oracle
  worst-task ratio       1.148829       1.148829          0.972307
  rank-1 fraction        1.000000       1.000000          0.333333

final learned selector
  worst-task ratio       1.705770       1.702361          0.999996
  worst seed ratio       1.611691       1.623054          0.999968
  rank-1 fraction        1.000000       1.000000          0.666667
  selector/oracle agree  1.000000       1.000000          1.000000
  distinct addresses     3              3                 2

oracle address switches  0              0                 0
median accept fraction   0.190278       0.166667          0.009722
material error max       0              0                 0
```

Relative to the full nine-address system:

```text
POSITION ONLY final worst-task margin / FULL = 0.998002
FREQUENCY ONLY final worst-task margin / FULL = 0.586243
```

`POSITION ONLY` therefore retains **99.80%** of the full median worst-task margin and exactly matches its `3/3` rank-1 performance. `FREQUENCY ONLY` reaches only **58.62%** of the full margin and only `2/3` tasks are rank-1.

## Preregistered verdict

```text
position_matches_full   = true
frequency_matches_full  = false

verdict = position_dominated_joint_not_needed
```

So Gate 7 **kills the joint position-frequency interpretation for this toy**.

The extra frequency choices in Gate 6 were not needed to explain the three-task addressing result. The actual task code was already present in spatial landing coordinate:

```text
task 0 -> upper source row
task 1 -> center source row
task 2 -> lower source row
```

while all three use the same carrier `omega = 1.35`.

## What survives

This does **not** kill the addressed-operator picture itself.

The blank substrate already supplies a stable spatial coordinate chart, and slow conservative material learning improves the worst-task margin from about

```text
1.1488x -> 1.70x
```

while preserving the address identities. One shared material therefore still changes the quality of several spatially addressed transfer relations at once.

The narrower surviving statement is:

> **In the current ObjektiYksi toy, source landing position is a sufficient address for the three tasks; slow material learning reshapes the shared operator family while preserving that spatial coordinate chart.**

That is a weaker and cleaner result than "learned joint position-frequency addressing."

## Important limitations

`POSITION ONLY` has a smaller search space than `FULL`, so Gate 7 establishes representational sufficiency, not that position is intrinsically a better coordinate or that three addresses always learn more efficiently than nine.

The fixed `omega = 1.35` in `POSITION ONLY` is intentionally the frequency chosen by every Gate-6 blank/final oracle task. It is an upper-bound test of whether the frequency dimension was necessary for the existing Gate-6 claim, not a general comparison of spatial and spectral coding.

Likewise, failure of this three-frequency `FREQUENCY ONLY` ablation says nothing about whether richer frequency structure can address computations in other resonant media or biological dendrites.

## Next kill

Gate 6 and Gate 7 share another suspicious feature:

```text
oracle address switches = 0
```

The best spatial address is determined by the blank geometry and remains unchanged while material learning improves margins.

The next attack should therefore ask whether learned material can ever change **which address is best**, rather than merely strengthening a pre-existing coordinate map.

A strong Gate 8 should deliberately construct tasks whose blank-material address preferences are ambiguous or wrong, then allow only the material to learn. If the best address still cannot move, the appropriate description becomes much more mundane:

> **fixed geometric multiplexer + learned transfer gains**

rather than a deformable addressed operator manifold.

If material learning can reproducibly create or move address optima under preregistered controls, then the stronger co-adaptive operator-family interpretation survives.

## Claim boundary

Gate 7 is an ablation of one engineered lattice and one address set. It establishes only that **the current Gate-6 result does not require the frequency coordinate**. It does not test biological dendritic frequency coding and does not establish a general primacy of spatial addressing.
