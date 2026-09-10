# Gate 9 — The best physical address moves

Gate 6 learned a task-to-address selector, but the exact best addresses never moved while material changed. Gate 7 then showed that frequency was unnecessary for those tasks, and Gate 8 showed that the material could learn a deliberately non-native spatial 3-cycle.

Gate 9 asks the remaining hard question without introducing any new learning rule:

> **After Gate-8 crossed learning, did the audit-best physical source address for each target actually move?**

## Protocol

Gate 9 reruns the exact Gate-8 conditions. The learner is unchanged. Only after learning is complete do we scan all three physical source rows for each of the three target ports.

For each target `t` and source address `s`, the audit utility is the same bounded target-versus-strongest-competitor quantity used throughout these gates:

```math
U(t,s;g)=\log(P_t+\sigma^2)-\log(\max_{j\ne t}P_j+\sigma^2).
```

The audit-best source is

```math
s^*(t;g)=\arg\max_s U(t,s;g).
```

Crucially, this all-source scan is **audit only**. Gate-8 learning never sees it. Gate 8 sees only the three fixed source-target relations of the requested mapping and one scalar: the current worst relation.

The blank native mapping is

```text
target upper   <- upper source
target center  <- center source
target lower   <- lower source
```

The Gate-8 crossed assignment is

```text
target upper   <- center source
target center  <- lower source
target lower   <- upper source
```

Every crossed-assigned source differs from the blank native source.

Controls rerun the Gate-8 identity-trained, random-crossed, and no-write-crossed materials.

## Preregistered success rule

Before reading the Gate-9 result, address movement was counted only if:

```text
blank assigned-address fraction                 = 0
blank native-address fraction                   = 1
median crossed final assigned-address fraction  = 1
EVERY crossed seed assigned fraction           >= 2/3
median minimum assigned-vs-native utility       >= +0.05
crossed assigned fraction exceeds random        >= 1/3
crossed assigned fraction exceeds no-write      >= 1/3
```

This prevents calling ordinary transfer improvement an address shift after the fact.

## Result

Focused Python 3.11 and 3.12 CI reproduced the same deterministic receipt.

```text
                         blank assigned   blank native   final assigned   final native
CROSSED                       0/3             3/3            3/3             0/3
IDENTITY                      0/3             3/3            0/3             3/3
RANDOM CROSSED                0/3             3/3            0/3             3/3
NO-WRITE CROSSED              0/3             3/3            0/3             3/3
```

For the crossed material, **every target chooses its crossed-assigned source as the exact audit optimum in every seed**.

The source optima move as follows:

```text
upper target:   upper  -> center
center target:  center -> lower
lower target:   lower  -> upper
```

Per-seed assigned-minus-old-native utility advantages are:

```text
seed 0
  upper target     +1.00487
  center target    +1.11568
  lower target     +1.03375

seed 1
  upper target     +1.61209
  center target    +1.11207
  lower target     +1.34084

seed 2
  upper target     +1.26566
  center target    +0.71042
  lower target     +0.62322
```

Summary:

```text
median final assigned-address fraction          1.000
minimum across seeds assigned-address fraction  1.000
median minimum assigned-vs-native utility      +1.00487
worst observed assigned-vs-native utility      +0.62322
median final native-address fraction             0.000
```

The controls do the opposite:

```text
IDENTITY
  median assigned fraction       0
  median native fraction         1
  min assigned-vs-native         -2.34

RANDOM CROSSED
  median assigned fraction       0
  median native fraction         1
  min assigned-vs-native         -2.78

NO WRITE
  median assigned fraction       0
  median native fraction         1
  min assigned-vs-native         -2.33
```

## Preregistered verdict

```text
blank_is_native = true
crossed_moves   = true

verdict = address_optima_move_with_learned_material
```

Gate 9 therefore passes the stronger test that Gate 6 failed to demonstrate.

## What changed conceptually

Gate 8 established that material can make a fixed non-native source produce a requested output. Gate 9 adds something stronger:

```text
before learning
    target t has best physical source s_native

slow material deformation
    g -> g'

after learning
    the same target t has best physical source s_crossed
```

or

```math
\arg\max_s U(t,s;g)
\ne
\arg\max_s U(t,s;g').
```

So the source coordinate is not merely a permanent label attached to a fixed transfer gain. The learned material can reshape the **address landscape** itself.

A useful description of the current toy is now:

> **The substrate supplies a physical coordinate system, but learning can deform which coordinates best expose a requested transfer relation.**

That is closer to a deformable family of addressed operators than to a fixed spatial multiplexer plus learned scalar gains.

## The important caveat

This result is not magical and should not be oversold. Gate 8 directly trains the utility of each crossed-assigned source-target relation. Gate 9 then asks whether those trained sources overtake the previously preferred alternatives. They do, decisively, but the requested destination of the movement was externally specified by the Gate-8 task.

Thus Gate 9 demonstrates **programmable address-optimum movement**, not autonomous discovery of where an address should move.

A stronger future test would separate the source of the task consequence from the address audit more aggressively—for example, train under sparse or partial observations and ask whether the unobserved address landscape reorganizes in a predictable way, or require a newly useful address to emerge without directly rewarding that address on every structural proposal.

## Relation to the dendritic hypothesis

The biological analogy should remain conditional. In a dendritic arbor, synaptic landing position is physically fixed on short timescales, but spine turnover, branch remodeling, synapse formation/elimination, channel redistribution, and network rewiring can alter which spatial/temporal inputs most effectively drive an AIS event.

Gate 9 does **not** show that neurons implement this exact search. It establishes only the computational-physics possibility relevant to the hypothesis:

> **changing one shared physical substrate can change not only transfer strength but which input coordinate is functionally optimal for a downstream event.**

The missing biological arrow remains the same one exposed by `DendriteAsIteratedFeedbackOperator` Gate 17: how local activity, event-related return, and delayed consequence identify useful structural changes without the globally evaluated keep/revert oracle used here.

## Claim boundary

Established in this engineered lattice:

- blank best source addresses are the native vertically aligned sources;
- Gate-8 crossed material makes all three deliberately non-native assigned sources become the exact audit-best sources;
- the address switch occurs in all three targets and all three seeds;
- identity, random, and no-write controls retain the native optima;
- the audit does not participate in learning.

Not established:

- autonomous address discovery;
- arbitrary or scalable coordinate transformations;
- biological dendritic address migration;
- local biological credit assignment;
- hardware or learning-efficiency advantage.

The result earns **deformable address landscape** in this toy, and nothing stronger.
