# Gate 10 — dense operator readback

Gate 10 asks whether the persistent material can be treated as a black-box linear operator rather than only as a router.

For three left input ports and three right output ports at fixed `omega = 1.35`, frozen material exposes

```math
M_g = C\left[K(g)-\omega^2I+i\omega\Gamma\right]^{-1}B.
```

The learner never stores `M_g` explicitly. Only the conserved edge-coupling vector `g` persists.

## Gate 10A — hidden physically reachable teacher

A hidden material state is made by 1000 conservative local transfers. Its 3x3 complex port-transfer matrix is the teacher. Student material starts from all ones.

The student is shown only **three dense random complex input vectors** and their three output vectors. Those three probes have full rank, but none is a basis vector. The hidden material and teacher matrix are never supplied to the update rule.

Every learning proposal is still only

```text
one local pair of adjacent edges
one gets +delta
one gets -delta
```

so total material is conserved. The only acceptance signal is the scalar normalized error on the three dense training probes.

After all structural learning stops, the transient wave state is absent. Only then are the basis vectors `e1,e2,e3` injected to reconstruct the columns of `M_hat`.

### Result

Across three fixed student proposal streams:

```text
blank material matrix error            55.2588% median
learned reconstructed matrix error      1.7686% median
worst learned seed                      2.7142%
blank held-out dense-probe error        54.3629% median
learned held-out dense-probe error       1.5592% median
random-drift matrix error               59.7667% median
no-write matrix error                   55.2588% median
material-sum relative error              0.0 in consequence runs
```

Per learned seed:

```text
seed 0   matrix error 2.7142%   held-out 3.0823%
seed 1   matrix error 1.7686%   held-out 1.4824%
seed 2   matrix error 1.4880%   held-out 1.5592%
```

The result passes the success bar fixed in the Gate-10 code before the CI receipt: median matrix and held-out error below 5%, every matrix-readback seed below 10%, and learned median matrix error below one quarter of the random-material control.

A representative target and recovered operator are visibly dense and phase-bearing. For example, one teacher row contains coefficients such as

```text
-0.005076+0.020449j   -0.033300-0.022832j   +0.040266+0.013706j
```

and the seed-1 recovered row is

```text
-0.005044+0.020309j   -0.033340-0.022834j   +0.039473+0.013468j
```

This is not one-hot routing. The persistent material has been system-identified from ordinary dense examples and later behaves like a learned linear transformation over the three-dimensional input space.

## Gate 10B — arbitrary dense-matrix attack

Success on 10A does **not** imply that this reciprocal resonant lattice can realize every 3x3 matrix. To attack that interpretation, the identical learner is given an unrelated dense complex random target, normalized to the blank operator's Frobenius norm but not generated from any known material state.

That attack fails the same readback criterion:

```text
blank arbitrary-target matrix error     145.6234% median
learned arbitrary-target matrix error    73.1030% median
best learned seed                         63.1225%
worst learned seed                        79.1520%
learned held-out error                    70.7990% median
random-drift matrix error                137.8650% median
```

The local consequence rule does move the physical operator toward the arbitrary target, but it remains far outside the 5% operator-recovery criterion.

## Verdict

```text
physically_reachable_operator_readback_succeeds_but_arbitrary_matrix_claim_fails
```

So the supported statement is:

> **A conserved resonant material can be taught, from dense input/output examples, to reproduce a dense linear operator that lies inside its physical operator family; after learning, that operator can be reconstructed by black-box basis queries.**

The unsupported stronger statement is:

> this material is an arbitrary matrix store.

The result is better understood as **operator storage / system identification on a constrained physical manifold**.

## Why this matters for the repo

Gates 0–9 established that material deformation can alter routing relations and even move which physical source coordinate is optimal. Gate 10 changes the unit of analysis from individual source-target relations to a whole vector-space transformation:

```text
routing relation     -> one element of a transfer operator
Gate 10              -> the 3x3 transfer operator itself
```

It therefore gives a precise version of the phrase "the material holds the matrix":

```text
persistent memory:       g
exposed linear operator: M_g
input vector:            x
computation:             y = M_g x
readback:                query e1,e2,e3 and collect the output columns
```

No claim is made yet about energy advantage, scaling, arbitrary matrices, multiple independently programmed operator addresses, nonlinear AI capability, or biology.

## Reproducibility

The focused Gate-10 workflow passed on Python 3.11 and 3.12. The two uploaded Gate-10 JSON receipts were byte-identical (`sha256 f160165483b7816d3a71b546adc88671d9de6e70fff12a3c556ed6c818db169f`).
