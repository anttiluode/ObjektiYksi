# Gate 2 — local dither + scalar consequence

Gate 0 was a positive control: the medium was handed the exact signed edge sensitivity of the desired receiver. Gate 2 removes that privilege.

## Rule

At each attempt, choose two edges that share a lattice node and move a small amount of coupling from one edge to the other. Only those two edges change:

```text
g_a <- g_a + delta
g_b <- g_b - delta
```

so total material is conserved locally and globally. No receiver-launched reference field and no analytic edge derivative are used by the learning rule.

After the dither, the fast field is solved from the source alone. The observer returns one bounded scalar:

```math
U = \log(P_{target}+\sigma^2)-\log(P_{decoy}+\sigma^2).
```

The noise floor makes `both outputs -> zero` return toward zero utility instead of rewarding the silence trap. `true_consequence` keeps the physical dither iff `U` improves; otherwise it reverts it.

This is derivative-free local search in the ordinary optimization sense. It is not a claim that biology implements this exact accept/revert loop.

## GitHub Actions receipt

Six fixed proposal streams were run on Python 3.11 and 3.12. Both jobs passed all tests and reproduced the experiment.

At baseline the mirror-symmetric ports have `target/decoy = 1`.

| rule | median target/decoy | range | median target power / baseline | median target rank |
|---|---:|---:|---:|---:|
| delayed true consequence | **41.1746x** | 18.1966–77.3677x | **2.2091x** | **1 / 9** |
| random accept | 1.1882x | 0.8817–1.4097x | 0.9635x | 4 / 9 |
| inverted consequence | 0.02136x | 0.00983–0.05255x | 0.04243x | 9 / 9 |
| no write | 1.0000x | exactly symmetric | 1.0000x | 6 / 9 |

The largest relative error in total material across the true-consequence runs was `2.58e-16`.

All six true-consequence seeds placed the target first on the nine-port right readout strip. The accepted local moves were about half of all valid probes (`median 0.5083`), so this is not a hidden exhaustive gradient sweep: random local physical proposals were tested and consequence selected them.

## What Gate 2 earns

In this toy, a persistent receiver preference can be found without the Gate-0 analytic sign:

```text
local material dither
        ↓
source-only field
        ↓
one scalar consequence
        ↓
keep / revert
        ↓
repeated physical search
        ↓
persistent resonant topology
```

After training, all fast field state can be discarded. The altered couplings alone reproduce the receiver preference on the next query.

That is a stronger form of `TOPOLOGY <-> SIGNAL` than Gate 0: the direction of the material write is discovered by perturbing the substrate and observing what happened, rather than supplied from the exact Green-function derivative.

## The attack that matters next

The objective names one mirror decoy. Although the target becomes rank 1 in every true-consequence seed, its margin over the **strongest other** right-side port is modest: only about `1.04x–1.40x` across the six seeds.

So the spectacular `41x` median target/decoy number must not be read as `41x` general receiver selectivity. A large part of it is specifically suppressing the named mirror decoy.

Gate 3 should therefore return one scalar that compares the desired receiver with the strongest competing port, without exposing which edge or which competitor caused the error. If local dither still produces a robust target margin, the result becomes a genuine multiport routing result rather than a two-port discrimination result.

## Claim boundary

Gate 2 shows only that **local conservative structural exploration plus a bounded scalar consequence can discover a persistent useful deformation of this reciprocal resonant toy**.

It does not show autonomous goal formation, biological credit assignment, dendritic growth, a literal fluid dam, or a long-delay eligibility mechanism. The `delay` here is operational: consequence is evaluated only after the physical perturbation has been made and the fast field has been re-solved.
