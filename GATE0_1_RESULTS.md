# Gates 0–1 — the first object

These numbers are frozen from GitHub Actions run `34464499224` on the `gate0-self-carving-topology` branch. Both Python 3.11 and 3.12 passed the tests and reproduced the experiment.

## Gate 0 — signal -> material -> changed signal

The uncarved 11x11 reciprocal lattice is mirror-symmetric:

```text
desired power               8.8909085e-4
decoy power                 8.8909085e-4
desired / decoy             1.000000
mean edge coupling          1.000000
```

The signed forward × receiver-reference rule was checked against a central finite difference on an edge:

```text
analytic dP/dg              3.9903149891e-5
finite difference           3.9903149090e-5
relative error              2.01e-8
```

After 70 slow material updates with the **mean edge coupling held at 1**, all fast field state is discarded and the source is queried alone:

```text
phase-reference rule
--------------------
desired power               0.87214750      (980.94x baseline)
decoy power                 0.00063825
desired / decoy             1366.48x
material std                0.5030
material min / max          0.1521 / 3.8583
corr(initial write field,
     final log material)    +0.4625
mean g in initial + fifth   1.1863
mean g in initial - fifth   0.8507
```

So a temporary signal-defined edge field became a persistent material asymmetry, and the persistent material changed a later source-only transfer by three orders of magnitude in target/decoy contrast.

### Controls

The controls are informative because **contrast alone can cheat**:

| write rule | desired / decoy | desired power / baseline | interpretation |
|---|---:|---:|---|
| signed phase reference | **1366.48x** | **980.94x** | target becomes both selective and much stronger |
| magnitude-only reference | 74.41x | 0.830x | contrast mostly comes from suppressing the decoy; target itself is slightly weaker |
| forward-only | 1.000x | 0.00762x | preserves symmetry while nearly silencing both readouts |
| scrambled signed reference | 1.238x | 0.159x | weak asymmetry, large loss of target power |
| no write | 1.000x | 1.000x | exact baseline |

This repeats the old `SighImageSuper` warning in a new substrate: a clean ratio is not enough if the useful signal is disappearing. The signed phase information is what prevents the positive control from winning only by silence.

## Gate 1 — it did **not** carve a simple static road

At the trained angular frequency `omega = 1.55`, the carved material gives:

```text
desired power                         0.87214750
desired fraction of total field      12.5744%
target rank on right readout strip   1 / 9
target / strongest other port        14.8449x
```

The uncarved target was only `0.0986%` of total field power and ranked `6 / 9` on the same strip.

Mirroring **only the persistent material** across the horizontal midline swaps the learned preference essentially exactly:

```text
original desired / decoy             1366.4767
mirrored desired / decoy             0.000731809
product                               1.000000000000011
```

So the preference lives in the material, not in a hidden source/readout asymmetry.

But a naive static-road interpretation fails. If edge cost is taken as `1/g`, the cheapest path after carving is actually cheaper to the **decoy**:

```text
source -> desired shortest cost       8.7864
source -> decoy shortest cost         7.4477
```

Despite that, the wave transfer favors the desired port by `1366x`.

That is the important correction:

> **The learned object is not adequately described as a high-conductance path. It is a frequency-dependent transfer topology.**

The carved target response peaks near `omega = 1.575` in the sampled sweep, with target power `0.8793`. Around the training band the target remains the strongest of the nine right-side ports; at `omega = 1.55` it beats the strongest other port by `14.84x`.

This is closer to the Berglund intuition than a Physarum-like road: some edges open, some close, but their computational effect is determined by how the entire changed network supports phase and resonance, not by static shortest-path conductance.

## What survived

Gate 0 earns the narrow statement:

```text
signal-defined functional edge field
        ↓
slow fixed-budget material deformation
        ↓
fast state erased
        ↓
persistent changed wave operator
        ↓
changed receiver-selective future
```

Gate 1 sharpens it:

```text
persistent topology != ordinary graph road
persistent topology = material + phase + resonance + frequency
```

What is **not** earned: a biological learning rule, spontaneous target discovery, free credit assignment, or a claim that nodal lines literally become walls.

## Next attack

The current signed phase-reference rule is an analytic positive control. Gate 2 should remove that privilege. A strictly local medium gets only small physical dithers plus delayed scalar consequence. The question is whether repeated perturb/keep/reject cycles can discover a material deformation that approaches the signed-reference result without ever being handed its gradient.