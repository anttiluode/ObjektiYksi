# Addressed Physical Operators as a Hypothesis for Dendritic Computation and Structural Plasticity

## Abstract

Artificial neural networks usually represent learning as changes in stored scalar or matrix-valued parameters. Biological neurons instead compute in an extended physical substrate whose morphology, membrane kinetics, synapse location, channel distributions, and axon initial segment jointly determine how signals propagate. This paper develops a deliberately narrow hypothesis: a dendritic arbor may be usefully modeled as an **addressed physical operator**. Input location and temporal structure select a particular transfer response of the arbor; the axon initial segment (AIS) converts the resulting continuous state into an output event; backpropagating activity and local dendritic state provide an event-related return signal; and slower plasticity can modify not only scalar synaptic efficacy but the physical operator itself through local conductance and structural change.

The hypothesis is motivated by two executable projects. `DendriteAsIteratedFeedbackOperator` shows, in phenomenological compartmental models including one reconstructed human L2/3 morphology, that dendritic transfer can exhibit non-zero resonant mode selectivity, that local physical edits can induce structured global operator changes, and that geometry sensitivity can organize into collective directions with finite observer-grounded optima. Importantly, its strongest attempted event-conditioned receipt fails a spatial-specificity control, so credit assignment is not established. `ObjektiYksi` shows in a separate reciprocal resonant medium that one conserved material substrate can generate a family of source- and frequency-conditioned effective operators, and that strictly local material dithers retained by one bounded scalar consequence can improve several addressed transfer relations simultaneously.

The proposal therefore does **not** claim that dendrites are acoustic cavities, that backpropagating action potentials carry an error gradient, or that the brain performs Hessian optimization. The proposed bridge is weaker and testable: **local activity identifies participation; an AIS-related return identifies that an output event occurred; a delayed third factor supplies usefulness; and small physical exploration can supply the sign of structural change.** The paper separates established biology, executable toy results, and novel speculation, and defines falsification tests intended to kill the hypothesis if simpler explanations suffice.

---

## 1. The problem

A central difficulty in biological learning is not merely how a synapse can detect pre- and postsynaptic activity. It is how a local site can distinguish three questions that are logically different:

1. **Did I participate in the state that drove the neuron?**
2. **Did the neuron subsequently commit an output event?**
3. **Was that event useful enough that the participating structure should be retained or changed?**

Conventional Hebbian and spike-timing-dependent rules address the first two only partially. Modern three-factor and eligibility-trace theories make the separation explicit: local co-activity creates an eligibility trace, while a later modulatory factor associated with reward, punishment, surprise, novelty, or another salient event determines whether plasticity is consolidated [1,2].

The remaining question considered here is what the plastic object must be. Most formal models write

```math
\Delta w_j = f(\text{pre}_j,\text{post},c),
```

where `w_j` is a scalar efficacy. But real dendrites are spatially extended dynamical systems. A change in channel density, spine-neck geometry, branch diameter, local membrane properties, or branch geometry can alter a transfer function rather than merely multiplying an input by one number.

The working hypothesis of this paper is therefore:

> **A neuron can learn by modifying a shared physical operator whose effective computation depends on where and how it is interrogated.**

That statement has three components that must be kept separate:

- **addressing:** location and temporal structure select a response of the dendritic operator;
- **commitment and consequence:** the AIS converts a continuous state into an output event, while backpropagating and modulatory signals provide locally available post-event information;
- **operator plasticity:** slower physical changes modify the future family of transfer responses.

None of these components is novel by itself. The proposed contribution is the specific synthesis and, more importantly, its executable falsification program.

---

## 2. Biological ground that already exists

### 2.1 Dendrites already implement location- and frequency-dependent transfer

The strongest prior art for the "address" concept is not metaphorical. Dendritic impedance and transfer impedance depend on membrane properties, geometry, input location, and frequency. Laudanski et al. showed that dendritic resonance is spatially distributed and can support spatiotemporal filtering: distinct dendritic locations can preferentially transmit different temporal modulation rates to the soma [3]. Narayanan and Johnston experimentally found location-dependent resonance in CA1 pyramidal neurons and showed that long-term potentiation can alter intrinsic resonant properties [4]. Earlier theoretical work likewise showed that non-uniform active conductances can produce strongly different somatic filtering for dendritic versus somatic inputs [5].

This motivates an operator notation. Let `theta` denote morphology and membrane state. A linearized frequency-domain transfer may be written schematically as

```math
H_\theta(\omega)=
\left[G(\theta)+i\omega C(\theta)+Y_{\rm active}(\omega;\theta)\right]^{-1}.
```

If a synaptic population enters at location `s` and the soma/AIS is the observed port, then the relevant response is a selected element or projection of this resolvent:

```math
M_{\rm dend}(s,\omega;\theta)
= P_{\rm AIS} H_\theta(\omega) P_{\rm in}(s).
```

This is the biological analogue of the address-conditioned operator used in `ObjektiYksi`. It is not necessary to claim that a biological neuron literally computes a Fourier transform or maintains a narrow-band carrier. Temporal pattern, burst rate, oscillatory context, receptor kinetics, and input synchrony are all possible physical coordinates of the same general idea.

### 2.2 Geometry is not a decorative container

Dendritic morphology materially affects neuronal dynamics. Mainen and Sejnowski showed in reconstructed neocortical model neurons that changing dendritic geometry while keeping a common ion-channel distribution could reproduce a wide range of firing patterns [6]. Cuntz, Borst, and Segev developed optimization principles for dendritic structure, treating morphology as constrained by competing wiring and signal-transfer objectives [7].

Structural plasticity is also a real learning-associated phenomenon. Motor learning can induce rapid formation and long-term stabilization of dendritic spines [8], and repeated learning can produce clustered spine formation [9]. Branch-specific calcium events can regulate persistent synaptic changes [10]. These results do not show that an arbor optimizes a global transfer objective, but they establish that the physical structure of dendrites is dynamic and functionally relevant.

### 2.3 The AIS is a genuine state-to-event boundary

The AIS is the principal site of action-potential initiation in many neurons and forms a specialized boundary between somatodendritic and axonal compartments [11,12]. Its location, length, and molecular composition are themselves plastic and can change with activity [13]. Thus it is reasonable to distinguish two computational stages:

```text
extended somatodendritic state  ->  AIS commitment  ->  axonal event
```

This is more precise than saying that the AIS "reads an eigenmode." The AIS does not need access to a symbolic mode identity. It simply transforms a voltage trajectory conditioned by the whole somatodendritic operator into spike timing and probability.

### 2.4 Backpropagating action potentials are event signals, not reward signals

Action potentials initiated near the AIS can backpropagate into dendrites and have long been implicated in associative plasticity [14]. Critically, a backpropagating action potential (bAP) should not be interpreted as carrying a behavioral error gradient. It provides information about the neuron's firing state and strongly affects local voltage and calcium dynamics.

Recent review work emphasizes that bAPs are not uniform broadcast pulses. Their amplitude and propagation can vary with branch, frequency, activity state, and trial, including branch-specific failures [15]. This variability weakens any simple "same receipt everywhere" picture but supports a more local formulation:

```math
r_j = r_j(\text{AIS event},\text{branch state},\text{history}).
```

A local branch can therefore receive an event-related signal that is shaped by the same physical arbor through which the forward activity propagated.

### 2.5 Credit assignment already has dendritic and three-factor theories

Urbanczik and Senn proposed a learning rule in which local dendritic state predicts somatic spiking; discrepancies between local dendritic prediction and somatic output influence synaptic plasticity [16]. Richards and Lillicrap reviewed dendritic mechanisms as possible solutions to credit assignment, emphasizing spatial separation of ordinary feedforward signals and feedback/credit-related signals [17]. Eligibility-trace and three-factor rules provide a well-developed mechanism for bridging local activity and delayed reward or novelty signals [1,2].

The present hypothesis should therefore not be presented as "solving biological credit assignment." It asks a narrower question:

> **Can the plastic variable be promoted from a scalar synaptic weight to a physical transfer operator, while retaining only local participation, an event-related return, delayed consequence, and small physical exploration?**

---

## 3. Executable results that motivate the hypothesis

### 3.1 `DendriteAsIteratedFeedbackOperator`

The companion repository tests the dendritic side directly in compartmental models. Its claim boundary is intentionally narrow.

A passive cable control performs ordinary diffusive modal selection. A phenomenological restorative quasi-active membrane introduces stable complex poles and non-zero pass bands. In a five-tone toy, tuned branches can strongly enrich one non-zero input component at a bounded somatic/AIS readout. This is an operational "mode purifier" only in the sense of selective filtering; it does not create information and does not imply a hidden nonlinear winner-take-all process.

Gate 11 then closes a toy causal loop:

```text
PURIFY -> AIS-like event -> state-dependent return -> local write -> replay
```

but explicitly does **not** claim that the return is an adjoint or gradient. Gate 17 is more important for the present paper because it attacks geometry and receipt credit separately.

On one reconstructed human L2/3 morphology, log-length perturbations along a 1.242 mm path were represented in 16 smooth orthonormal modes. The first four modes contained 97.88% of measured first-order gradient energy, but 96.73% of the total was the uniform mode alone: much of the apparent low-dimensional structure was simply global shortening. In the first eight smooth modes, however, the measured Hessian was negative in all eight directions, with effective curvature rank about 5.23. A Newton candidate computed in that subspace survived direct nonlinear replay: the counted full step improved the bounded objective by 17.33% and reduced the measured local gradient norm to 19.26% of baseline. The strongest warranted conclusion is therefore not that each branch segment has a "natural length," but that **a collection of locally sensitive geometric coordinates can possess a finite observer-grounded optimum in a collective subspace**.

Gate 17 simultaneously tested an event-conditioned receipt. A seemingly attractive correlation of about 0.774 between receipt-derived mode projections and the measured geometry gradient failed a circular-shift null: shifted receipts were typically at least as predictive, with empirical `p = 0.953`. A reciprocal-return control performed similarly. Thus

```text
local sensitivity != collective structural curvature != credit assignment.
```

The geometry result survives; the tested receipt-credit mechanism does not.

See: https://github.com/anttiluode/DendriteAsIteratedFeedbackOperator/blob/main/GATE17_GEOMETRY_MODES.md

### 3.2 `ObjektiYksi`

`ObjektiYksi` tests the complementary idea in a deliberately non-biological resonant lattice. Its persistent state is a conserved material vector `g`. For source coordinate `s`, carrier `omega`, and selected ports,

```math
H_g(\omega)=\left[K(g)-\omega^2I+i\omega\Gamma\right]^{-1},
```

and

```math
M_{\rm eff}(s,\omega;g)
=P_{\rm out}H_g(\omega)P_{\rm in}(s).
```

`M_eff` is not stored. One material substrate generates a family of effective operators depending on how it is interrogated.

The important progression is:

- Gate 0: an analytic positive-control sensitivity carves a resonant transfer preference while conserving mean coupling.
- Gate 1: the learned function is not reducible to an ordinary shortest conductive road.
- Gate 2: the analytic edge-gradient sign is removed; strictly local conservative material dithers are kept or reverted using one bounded scalar consequence.
- Gate 3: the same impoverished rule learns against whichever output is currently the strongest competitor, without receiving the competitor identity.
- Gate 4: frozen learned material generalizes more strongly across frequency than across source position, showing that the query coordinate is part of the effective topology.
- Gate 5: a long-budget worst-case learner sculpts one material so that several source-conditioned transfer relations become useful simultaneously.
- Gate 6: a tiny online selector learns which physical address to use for each task while the shared material changes.

Gate 6 initially looked like evidence that both source position and frequency become learned computational coordinates. The receipt contains an important correction. On the blank substrate, the audit-only exact oracle already gives all three tasks rank-1 using three distinct addresses. In the online runs, the initial and final oracle choices are

```text
task 0 -> source_dr = -1, omega = 1.35
task 1 -> source_dr =  0, omega = 1.35
task 2 -> source_dr = +1, omega = 1.35
```

for all three seeds. Thus the task-specific address code in the current Gate 6 is entirely in **source position**; the frequency coordinate is not used to distinguish the tasks. The selector learns to discover a coordinate code already present in the blank geometry. Slow material learning then improves the worst selected target/competitor ratio from about `1.1488x` to `1.7492x` while preserving the best addresses.

This negative clarification is central. Gate 6 supports

> **a stable physical coordinate chart over a slowly deforming operator family**

more strongly than it supports

> **the spontaneous learning of a new joint position-frequency code.**

That distinction motivates Gate 7 below.

---

## 4. Hypothesis: the dendrite as an addressed physical operator

Let `theta` collect the slowly persistent physical state of a dendritic arbor: geometry, local channel densities, spine-neck parameters, and other structural or quasi-structural variables. Let `a` denote an input address, broadly construed:

```math
a=(s,\omega,\phi,\text{burst pattern},\text{receptor state},\ldots).
```

The fast computation is

```math
x \mapsto M(a;\theta)x.
```

The AIS transforms the resulting somatodendritic trajectory into an event `z`:

```math
z=\mathcal A_{\rm AIS}[M(a;\theta)x,\text{context}].
```

Forward activity leaves a local eligibility trace `e_j`. The AIS event and current dendritic state produce a local return quantity `r_j`. A delayed modulatory or behavioral consequence supplies a third factor `c`.

The minimal plasticity architecture is therefore

```math
e_j \;\;\times\;\; r_j \;\;\times\;\; c.
```

But this product is **not assumed to contain a gradient sign**. The companion experiments repeatedly suggest a safer division of labor:

```text
eligibility / receipt  ->  WHERE to explore
small local dither     ->  WHICH WAY
later consequence      ->  WHETHER to keep it
```

A local physical parameter `theta_j` can therefore execute

```math
\theta_j' = \theta_j + \delta,
```

observe a later scalar consequence, and either retain or reverse the perturbation. In the simplest two-sided form,

```math
\theta_j\in\{\theta_j-\delta,\theta_j,\theta_j+\delta\}.
```

This turns biological noise, channel turnover, spine motility, or microstructural remodeling from nuisance into a possible zeroth-order search process.

The resulting learning loop is

```text
spatiotemporal input
        ↓
address-conditioned dendritic operator
        ↓
AIS commitment
        ↓
branch/state-dependent event return
        +
local eligibility
        +
delayed third factor
        ↓
small physical exploration
        ↓
retain / reverse
        ↓
changed dendritic operator
        ↓
next input sees a different family of responses
```

This is the core hypothesis.

---

## 5. What is and is not new

The proposal is **not** novel in any of the following isolated claims:

- dendrites filter signals in a location- and frequency-dependent way [3-5];
- morphology affects neuronal dynamics [6,7];
- bAPs influence dendritic plasticity [14,15];
- dendrites may participate in credit assignment [16,17];
- eligibility traces plus third factors can bridge delayed consequences [1,2];
- learning changes synaptic and dendritic structure [8-10];
- local feedback can configure physical wave systems.

The potentially distinct synthesis is:

> **The persistent object is a deformable physical operator rather than a table of scalar weights; input coordinates address different effective responses of that operator; output commitment creates a branch-state-dependent post-event signal; and local physical exploration plus delayed consequence can reshape the operator without requiring an explicit gradient.**

This should be treated as a hypothesis until it makes predictions that distinguish it from simpler models.

---

## 6. Predictions that can kill the hypothesis

A useful theory should lose when its special machinery is unnecessary. The following are direct falsifiers.

### Prediction 1 — location-dependent operator structure must matter beyond scalar attenuation

If responses generated by changing dendritic input location can be reproduced by a single scalar gain applied to one common kernel, then "addressed operator" is unnecessary. A strong test should compare complex transfer shape, phase, latency, frequency response, and nonlinear event probability after matching somatic RMS or peak amplitude.

**Kill condition:** matched-amplitude inputs at different locations produce indistinguishable AIS behavior and no meaningful non-scalar transfer differences.

### Prediction 2 — a branch-dependent event return must carry information beyond generic postsynaptic firing

If the only biologically useful information in `r_j` is a global binary "the neuron spiked," then the proposed branch-state-dependent receipt adds nothing to ordinary three-factor rules.

**Kill condition:** replacing the actual local bAP/return waveform with a spatially uniform postsynaptic-event flag preserves structural learning equally well across relevant tasks and branches.

The negative Gate 17 receipt result already pushes in this direction: the first phenomenological event-conditioned receipt failed its spatial null.

### Prediction 3 — shape-only collective geometry must survive removal of global size

Gate 17's first-order geometry structure is dominated by uniform shortening. The stronger collective-shape claim therefore requires a zero-mean geometry test.

```math
\langle \delta\log L,1\rangle_L=0.
```

**Kill condition:** once global path scale is held fixed, collective curvature and finite interior optima disappear or fail to replicate across paths/cells.

This is the planned Gate 18 in `DendriteAsIteratedFeedbackOperator`.

### Prediction 4 — physical exploration must outperform blind structural drift

A consequence-gated local dither mechanism is only interesting if the consequence actually selects useful deformations.

**Kill condition:** matched random retain/revert controls produce comparable operator improvement, robustness, and replay performance.

`ObjektiYksi` Gates 2-5 already reject the simplest random-drift controls in the lattice toy, but a dendritic version remains open.

### Prediction 5 — addressed computation must require more than a trivial geometric coordinate code

Gate 6 exposes the immediate danger. Its three tasks already have three distinct rank-1 addresses on the blank material, and all three chosen addresses share the same frequency. The "learned joint address" interpretation is therefore too strong.

**Kill condition:** a position-only address space matches the full position × frequency address space, while frequency-only addressing adds no task-specific capacity.

This is Gate 7 of `ObjektiYksi`.

### Prediction 6 — slow operator deformation should eventually be able to change the best address

A truly co-adaptive addressed substrate should admit regimes where changing the substrate changes which address is best. Gate 6 did not demonstrate this: despite repeated damage to inactive tasks, the audit-best addresses stayed fixed.

**Kill condition:** across broad material changes and more asymmetric tasks, the best address remains determined almost entirely by static geometry and never meaningfully depends on learned material.

If this happens, the correct interpretation is not a learned operator manifold but a fixed coordinate multiplexer with tunable gains.

---

## 7. Gate 7 preregistration — kill the frequency-address story first

The first new falsification gate is intentionally simple because the existing receipt already points to it.

### Observation to attack

Gate 6 exposes nine candidate addresses:

```text
3 source landing rows x 3 frequencies = 9 addresses.
```

Yet the blank and learned exact oracle use only

```text
(-1, 1.35), (0, 1.35), (+1, 1.35).
```

Frequency does not distinguish the three tasks.

### Gate 7 comparison

Run the same bounded online-selector + slow conservative material-learning protocol with three address spaces:

```text
FULL
    source_dr in {-1,0,+1}
    omega     in {1.35,1.55,1.75}

POSITION ONLY
    source_dr in {-1,0,+1}
    omega     = 1.35

FREQUENCY ONLY
    source_dr = 0
    omega     in {1.35,1.55,1.75}
```

Record for each condition:

- blank-material oracle worst-task ratio;
- blank-material rank-1 fraction;
- final selected worst-task ratio;
- final selected rank-1 fraction;
- selector/oracle agreement;
- number of distinct final addresses;
- material conservation and acceptance rate.

### Interpretation fixed in advance

If `POSITION ONLY` matches `FULL`, Gate 6 has **not** earned a joint position-frequency addressing claim. Position is the operative task coordinate in this toy.

If `FREQUENCY ONLY` performs comparably, then frequency is an interchangeable coordinate rather than a uniquely learned dimension.

If `FULL` substantially outperforms both ablations, then a genuinely joint address begins to be supported.

No post-hoc relabeling of the outcome will change these criteria.

---

## 8. Relation to artificial learning systems

The architectural contrast with a conventional network is easiest to state as

```math
y=W_qx
```

versus

```math
y=M(a;\theta)x.
```

In the first expression, a task may select or generate a stored matrix `W_q`. In the second, one shared physical substrate `theta` generates a family of transformations and a low-dimensional address selects which part of that family is exposed.

Slow learning changes `theta`, thereby changing many effective operators at once:

```math
\theta\rightarrow\theta'
\quad\Rightarrow\quad
\{M(a;\theta)\}_{a\in\mathcal A}
\rightarrow
\{M(a;\theta')\}_{a\in\mathcal A}.
```

This coupling is both the attraction and the danger. It offers parameter sharing and physical multiplexing, but every local edit can alter many addressed behaviors. That immediately reconnects the idea to continual-learning and interference problems: preserving one response while a shared operator changes is not automatic.

The strongest engineering question is therefore not whether such a system is "brain-like." It is whether a shared physical operator plus cheap addressing can offer useful memory, routing, adaptation, or robustness per degree of persistent state compared with explicitly stored matrices.

No such efficiency advantage is currently established.

---

## 9. Limitations

The current evidence is several steps away from biology.

First, `ObjektiYksi` is a reciprocal damped resonant lattice, not a neuron. Its frequency coordinate is an explicit sinusoidal carrier and its tasks are externally named output ports. Gate 6 uses a small digital value table for address selection. It demonstrates an operator-family concept, not autonomous semantics.

Second, `DendriteAsIteratedFeedbackOperator` uses phenomenological membrane kinetics and observer objectives. The reconstructed morphology is biological, but the objective used to derive collective geometry modes is imposed by the experimenter. A finite optimum in that model does not imply that the real cell developed by optimizing the same objective.

Third, the strongest attempted receipt-credit mapping has already failed a spatial-null test. This is evidence against, not for, the claim that an AIS-conditioned return by itself solves credit assignment.

Fourth, structural changes in real neurons span very different timescales and mechanisms: synaptic receptor trafficking, channel redistribution, spine-head and spine-neck changes, spine addition/removal, dendritic branch remodeling, and AIS plasticity should not be collapsed into one variable `theta` without experimental justification.

Finally, novelty has not been exhaustively established. The literature already contains strong precedents for dendritic spatiotemporal filtering, morphology optimization, dendritic credit assignment, three-factor learning, bAP-dependent plasticity, and structural plasticity. The claim here is a synthesis plus a set of executable predictions, not a declaration that no related theory exists.

---

## 10. Conclusion

The most defensible form of the idea is not that a dendrite "learns an eigenmode" or that an AIS spike carries an error backward. It is this:

> **A dendritic arbor is an extended physical transfer operator. Where and how an input arrives can select different responses of that operator. The AIS converts the resulting state into an output event. Local activity, event-related backpropagation, and delayed modulatory consequence provide distinct pieces of information. Slow physical change can alter the operator itself, and small local exploration can in principle discover useful structural directions without an explicit gradient.**

The synthesis is biologically plausible enough to test and specific enough to fail.

The next correct move is therefore not to make the story larger. It is to remove its easiest explanations one at a time.

Gate 7 starts with the simplest one: **does frequency actually contribute to the learned address, or did we merely build a three-position router and give it six unused extra coordinates?**

---

## References

1. Gerstner W, Lehmann M, Liakoni V, Corneil D, Brea J. Eligibility Traces and Plasticity on Behavioral Time Scales: Experimental Support of NeoHebbian Three-Factor Learning Rules. *Front Neural Circuits*. 2018;12:53. https://doi.org/10.3389/fncir.2018.00053
2. Frémaux N, Gerstner W. Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules. *Front Neural Circuits*. 2016;9:85. https://doi.org/10.3389/fncir.2015.00085
3. Laudanski J, Torben-Nielsen B, Segev I, Shamma S. Spatially Distributed Dendritic Resonance Selectively Filters Synaptic Input. *PLoS Comput Biol*. 2014;10(8):e1003775. https://doi.org/10.1371/journal.pcbi.1003775
4. Narayanan R, Johnston D. Long-term potentiation in rat hippocampal neurons is accompanied by spatially widespread changes in intrinsic oscillatory dynamics and excitability. *Neuron*. 2007;56(6):1061-1075. https://doi.org/10.1016/j.neuron.2007.10.033
5. Zhuchkova E, Remme MWH, Schreiber S. Somatic versus Dendritic Resonance: Differential Filtering of Inputs through Non-Uniform Distributions of Active Conductances. *PLoS One*. 2013;8(11):e78908. https://doi.org/10.1371/journal.pone.0078908
6. Mainen ZF, Sejnowski TJ. Influence of dendritic structure on firing pattern in model neocortical neurons. *Nature*. 1996;382:363-366. https://doi.org/10.1038/382363a0
7. Cuntz H, Borst A, Segev I. Optimization principles of dendritic structure. *Theor Biol Med Model*. 2007;4:21. https://doi.org/10.1186/1742-4682-4-21
8. Xu T, Yu X, Perlik AJ, et al. Rapid formation and selective stabilization of synapses for enduring motor memories. *Nature*. 2009;462:915-919. https://doi.org/10.1038/nature08389
9. Fu M, Yu X, Lu J, Zuo Y. Repetitive motor learning induces coordinated formation of clustered dendritic spines in vivo. *Nature*. 2012;483:92-95. https://doi.org/10.1038/nature10844
10. Cichon J, Gan W-B. Branch-specific dendritic Ca2+ spikes cause persistent synaptic plasticity. *Nature*. 2015;520:180-185. https://doi.org/10.1038/nature14251
11. Kole MHP, Stuart GJ. Signal processing in the axon initial segment. *Neuron*. 2012;73:235-247. https://doi.org/10.1016/j.neuron.2012.01.007
12. Leterrier C. The Axon Initial Segment: An Updated Viewpoint. *J Neurosci*. 2018;38:2135-2145. https://doi.org/10.1523/JNEUROSCI.1922-17.2018
13. Yamada R, Kuba H. Structural and Functional Plasticity at the Axon Initial Segment. *Front Cell Neurosci*. 2016;10:250. https://doi.org/10.3389/fncel.2016.00250
14. Sjöström PJ, Rancz EA, Roth A, Häusser M. Dendritic excitability and synaptic plasticity. *Physiol Rev*. 2008;88:769-840. https://doi.org/10.1152/physrev.00016.2007
15. Stochasticity in action potential backpropagation: consequences for neuronal computation. *Review*, 2026. PubMed PMID: 41948461. https://pubmed.ncbi.nlm.nih.gov/41948461/
16. Urbanczik R, Senn W. Learning by the dendritic prediction of somatic spiking. *Neuron*. 2014;81(3):521-528. https://doi.org/10.1016/j.neuron.2013.11.030
17. Richards BA, Lillicrap TP. Dendritic solutions to the credit assignment problem. *Curr Opin Neurobiol*. 2019;54:28-36. https://doi.org/10.1016/j.conb.2018.08.003
