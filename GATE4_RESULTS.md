# Gate 4 — frozen routing basin

Gate 3 learned at exactly one source node and one carrier frequency. Gate 4 freezes that material completely and asks what, if anything, survives when the query moves.

No write is permitted in this gate.

## Attack

For each of the six Gate-3 trained materials we evaluate:

```text
frequency:  17 points from omega=1.35 to 1.75, source fixed
source:     3x3 neighborhood around the training source, omega fixed at 1.55
joint:      5 frequencies x 9 source positions
```

Every query measures the chosen target against the strongest of the other eight output ports.

This distinguishes two very different interpretations:

```text
A. the material learned a destination-like routing basin
B. the material learned a source-conditioned transfer relation
```

## GitHub Actions receipt

Run `34466485537` reproduced Gates 0–4 on Python 3.11 and 3.12; both jobs passed all 18 tests.

The frozen trained materials give, across six seeds:

```text
frequency perturbation, source fixed
  median contiguous rank-1 width        0.2875 omega units
  minimum contiguous rank-1 width       0.2500
  median held-out rank-1 fraction       0.71875
  median held-out target/strongest      1.58396x

source perturbation, omega fixed
  median held-out rank-1 fraction       0.31250
  median held-out target/strongest      0.61953x

joint source + frequency perturbation
  median held-out rank-1 fraction       0.43182
  median held-out target/strongest      0.86578x
  worst seed/query target/strongest     0.03974x
```

So the result is not a generic target attractor.

The material is surprisingly tolerant to **carrier-frequency** changes while the source remains at the trained landing point: on the sampled grid, the target remains rank 1 across a substantial contiguous frequency interval. But moving the source by only one lattice site often destroys the preference.

The asymmetry is the result:

```text
frequency generalization  >>  source-position generalization
```

## Interpretation

Gate 1 already showed that the learned object is not a shortest conductive road. Gate 4 now says it is also not simply a destination basin independent of where the signal enters.

The more accurate object is a **source-conditioned Green-function relation**:

```math
H_g(t \leftarrow s;\omega).
```

The persistent material makes one family of source-to-target transfers favorable. If the source landing coordinate changes, a different mixture of the global resonant modes is excited, and the carved arrangement can cease to be useful even though the material itself has not changed.

This is important for the original topology/signal intuition. The topology is not only a property of the material. The functional topology is relational:

```text
material + source coordinate + frequency + readout
                    ↓
              effective routing
```

A fixed material can therefore contain many different effective topologies depending on how it is interrogated.

## What Gate 4 kills

It kills the strongest simple reading of Gate 3:

> carve toward a target once, then nearby inputs will naturally flow to the same target.

They do not, in this toy.

It also means that the old `Landing Zones` idea matters again. If the input location is part of the code, then changing the landing point is not necessarily innocuous noise; it can mean asking a different question of the same operator.

## Next attack

Gate 5 should distinguish **source specificity as a limitation** from **source specificity as an address**.

Train one material on a small set of source landing points while still returning only one scalar consequence. A strong version is a worst-case objective:

```math
U = \min_{s\in S}
\left[
\log(P_{target}(s)+\sigma^2)
-\log(\max_{j\ne target}P_j(s)+\sigma^2)
\right].
```

The structural rule need not be told which source or which competing output caused the worst case. It only receives the one scalar after each local physical dither.

Then hold out other source positions.

If one material can carve a basin that generalizes to unseen landing points, spatial specificity was just under-training. If not, the stronger interpretation is that the medium naturally learns **relations between coordinates**, not destination labels.

## Claim boundary

Gate 4 establishes only that the Gate-3 material has a measurable **frequency basin at the trained source** and substantially weaker transfer to nearby unseen source locations.

It does not establish biological place coding, spatial cognition, an anatomical landing-zone mechanism, or a universal invariant topology. The source-position result is a negative result and is kept as such.
