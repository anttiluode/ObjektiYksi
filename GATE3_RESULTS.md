# Gate 3 — strongest-competitor multiport carving

Gate 2 could make the chosen target beat one named mirror decoy by a large factor, but its margin over the strongest of all other output ports was only about `1.04x–1.40x`. Gate 3 attacks exactly that weakness.

## What the structural rule is allowed to know

The physical proposal mechanism is unchanged: choose two edges that meet at one lattice node and transfer a fixed amount of coupling from one to the other. The proposal is local and conserves total material exactly.

After the perturbation, the source-only field is re-solved. The structural rule receives one number:

```math
U_{multi}
= \log(P_{target}+\sigma^2)
- \log(\max_{j \ne target} P_j+\sigma^2).
```

It is **not told which output port is currently strongest**, where that competitor is, or which edge should change. It sees only whether the scalar became better or worse and keeps/reverts the local dither.

## GitHub Actions receipt

Six fixed proposal streams, 650 attempts each, were reproduced on Python 3.11 and 3.12. The Python 3.12 job passed all 15 tests and the complete Gates 0–3 stack; the paired 3.11 job reproduced the same deterministic Gate 3 output.

The baseline target is only rank `6/9` on the right-side output strip and has

```text
target / strongest other = 0.737698x
```

After consequence-selected local dither:

| rule | median target / strongest other | range | median target power / baseline | all seeds rank 1? |
|---|---:|---:|---:|---:|
| **true multiport consequence** | **2.05223x** | **1.67773–2.19309x** | **2.80903x** | **yes** |
| Gate-2 two-port consequence | 1.22740x | 0.90161–1.49376x | 3.32306x | no |
| random accept | 0.58733x | 0.49036–0.64297x | 0.80799x | no |
| inverted multiport consequence | `1.04e-5` | `1.44e-6–3.47e-5` | `2.80e-5x` | no |
| no write | 0.737698x | fixed | 1.00000x | no |

The true multiport rule accepts only about `23.9%` of valid random physical proposals. Total material is conserved to at least `1.29e-16` relative error.

The strongest competing port changes identity a median **59 times** on accepted steps. That matters: the objective is not simply suppressing one pre-selected decoy. As one competitor falls, another becomes the active boundary, and the same one-number consequence keeps reshaping the material against the moving frontier.

## What changed from Gate 2

Gate 2 showed:

```text
local physical experiment + one scalar consequence
    -> persistent two-port preference
```

Gate 3 now shows:

```text
local physical experiment
    ↓
source-only field across many outputs
    ↓
one scalar: target versus whoever is currently strongest
    ↓
keep / revert
    ↓
competitor identity changes repeatedly
    ↓
persistent target rank 1 against every output port
```

This is a stronger routing result. The medium never receives an edge gradient or the address of the current mistake.

## The important limitation

The scalar consequence still contains the **externally specified goal**: which output is the target. Gate 3 therefore does not solve autonomous credit assignment or goal formation. It shows only that once a bounded global consequence exists, spatially local structural search can discover a distributed resonant material arrangement that satisfies it.

There is also no long eligibility delay here. `Delayed` means the material is perturbed first, the fast field is then re-solved, and only afterward is the scalar consequence evaluated.

## Next attack

The result is trained and tested at the same carrier frequency and same source location. Gate 4 should ask whether the carved topology is a brittle frequency-specific trick or a more persistent routing object.

The clean attack is to train at one carrier and evaluate a held-out band and small source-position perturbations **without further writing**. If the preference collapses immediately off-frequency, the object is a narrow resonant filter. If a useful target rank survives a finite band or source neighborhood, then the carved topology has a genuine basin of routing behavior.
