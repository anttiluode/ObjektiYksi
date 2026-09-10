# Gate 12 — collision or optimization?

Gate 11 found a narrow miss when two independently specified, individually learnable dense `3x3` operators were forced into one shared material state `g` at two frequency addresses. The original 2400-proposal joint learner reached a median worst-address matrix error of `11.2311%`, just outside the preregistered `10%` bar.

Gate 12 asks whether that miss was a real local capacity wall or simply insufficient search.

> **Was the Gate-11 penalty a geometric collision in the shared physical operator manifold, or an optimization-budget effect?**

The experiment starts from the exact Gate-11 conflicting endpoints for seeds `0,1,2`. No Gate-11 threshold is relaxed.

## 12A — exact tangent audit

At each address,

```math
M_g(\omega)=C[K(g)-\omega^2I+i\omega\Gamma]^{-1}B.
```

For edge material coordinate `g_e`, the exact derivative is

```math
\frac{\partial M}{\partial g_e}
=-(C H b_e)(b_e^T H B).
```

The real and imaginary coordinates of both complex `3x3` matrices are stacked. That gives `36` real output coordinates. The material Jacobian is projected into the fixed-total-material tangent so the audit cannot cheat by adding total coupling.

Across all three Gate-11 endpoints:

```text
stacked tangent rank                   36 / 36
median gradient cosine                -0.079367
range of gradient cosine              -0.105224 ... +0.076074
median best linearized residual ratio  3.70e-15
worst best-linearized residual ratio   4.06e-15
```

The two address-specific gradients are close to orthogonal on average, not identical. More importantly, each address's own least-squares correction is predicted to worsen the other address: the cross-residual multipliers range from about `1.06x` to `3.00x`.

So the interference is real.

But the joint stacked Jacobian has full real output rank and a joint first-order direction that cancels the complete residual to numerical precision. That is incompatible with interpreting the Gate-11 endpoint as a demonstrated local rank/capacity wall.

Full rank is only a local degree-of-freedom result; by itself it does not prove a globally realizable shared solution.

## 12B — exact-Jacobian oracle diagnostic

To test whether a genuine common material state exists nearby, Gate 12 allows an explicitly audit-only Gauss–Newton diagnostic. It gets the exact joint Jacobian and at most five nonlinear backtracked steps while preserving the fixed material sum and material bounds.

This is **not** presented as the biological or local learning rule.

Result:

```text
median worst matrix error              4.0391e-8
worst seed worst matrix error          5.1115e-8
median worst held-out error            3.9890e-8
```

The oracle therefore finds a common `g` that realizes both independently specified target matrices to essentially numerical precision.

For this pair, the two-target problem is not merely approximately compatible: a shared solution exists.

## 12C — more of the original derivative-free learner

The stronger practical attack gives the original Gate-11 local keep/revert rule a fresh stream of `7200` proposals. It still gets no matrix Jacobian and no oracle update. Every proposal is the same local conservative material transfer used before.

```text
Gate-11 endpoint median worst matrix error   11.2311%
continued-local median worst matrix error     5.2831%
continued-local worst seed                     6.9466%
continued-local median worst held-out error    5.0762%
```

Every seed crosses the original `10%` matrix threshold, and the median held-out error also crosses the original `10%` threshold.

Per-seed final worst matrix errors are approximately:

```text
seed 0   5.2831%
seed 1   6.9466%
seed 2   4.5464%
```

So the exact learner whose earlier miss motivated the collision hypothesis rescues itself when given more search budget.

## Verdict

```text
gate11_near_miss_was_budget_limited_local_search_rescues
```

The Gate-11 result remains useful because it measured genuine cross-address interference: updates that help one exposed operator can hurt another.

But Gate 12 kills the stronger interpretation.

> **The observed Gate-11 miss was not a demonstrated representational collision. For this tested pair, one shared material can realize both independently specified dense operators, and the original derivative-free local learner reaches the preregistered accuracy bar with more search.**

This means the present evidence supports a more interesting picture than either extreme:

```text
addressed operators are coupled
        !=
addressed operators are necessarily incompatible
```

A single material edit deforms the family together, so interference is real. But the shared operator manifold can still contain joint solutions.

## Claim boundary

Gate 12 does **not** establish that arbitrary matrix pairs are jointly realizable, that an unlimited number of addressed operators can coexist, that this scales to useful matrix sizes, or that physical storage is more efficient than digital weights.

It establishes one narrower result:

```text
two separately specified reachable 3x3 operators
at two frequency addresses
+ one conserved shared material
+ dense-example local derivative-free learning
-> both recovered below the original 10% bar
```

The oracle additionally shows that this tested pair admits an essentially exact common material solution.

The next scientifically useful question would be where compatibility finally breaks as matrix size, number of addresses, target independence, or material constraints increase. This repository pauses here rather than declaring such a capacity law before measuring it.
