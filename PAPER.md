# Addressed Physical Operators as a Hypothesis for Dendritic Computation and Structural Plasticity

## Abstract

Artificial neural networks usually represent learning as changes in stored scalar or matrix-valued parameters. Biological neurons instead compute in an extended physical substrate whose morphology, membrane kinetics, synapse locations, channel distributions, and axon initial segment jointly determine how signals propagate. We develop a deliberately narrow hypothesis: a dendritic arbor may be usefully modeled as an **addressed physical operator**. Input location and temporal structure select a response of the arbor; the axon initial segment (AIS) converts the resulting continuous state into an output event; backpropagating activity and local dendritic state provide event-related information; and slower plasticity can modify not only synaptic efficacy but the physical operator itself.

The hypothesis is motivated by two executable projects. `DendriteAsIteratedFeedbackOperator` shows, in phenomenological compartmental models including a reconstructed human L2/3 morphology, that dendritic transfer can exhibit non-zero resonant selectivity, that local physical edits can induce structured global operator changes, and that geometry sensitivity can organize into collective directions with finite observer-grounded optima. Its strongest tested event-conditioned receipt, however, fails a spatial-specificity control, so biological credit assignment is **not** established. `ObjektiYksi` tests the complementary computational-physics object in a reciprocal resonant lattice. One conserved material substrate generates a family of input-conditioned transfer operators and can be modified by strictly local material-conserving perturbations retained or reverted using one bounded scalar consequence.

Falsification changes the interpretation substantially. Gate 7 shows that the current three-task result does **not** require joint position-frequency addressing: source position alone retains 99.80% of the full model's median worst-task margin, whereas frequency alone fails. Gate 8 then attacks the trivial explanation that position merely reads out native geometry. A deliberately wrong three-cycle source-to-output mapping improves from a worst relation of 0.1098x to a median 1.5225x, with all three requested relations rank-1 in every seed. Gate 9 audits all candidate source positions after the same learning and finds that the exact best physical source for every target moves from its blank native location to the source imposed by the crossed mapping in every seed. Thus, in the toy, slow physical change can deform not merely transfer amplitudes but the **address landscape** of the operator family.

The proposal does **not** claim that dendrites are acoustic cavities, that backpropagating action potentials carry an error gradient, that the brain performs Hessian optimization, or that the lattice learning rule is biologically local. The proposed bridge is weaker: local activity can identify participation; an AIS-related event can provide postsynaptic consequence information; a delayed third factor can report usefulness; and small physical exploration could in principle supply the sign of structural change. The decisive unresolved problem is the physical map from these locally available quantities to useful structural edits. The paper therefore treats the idea as a falsifiable synthesis rather than an established mechanism.

---

## 1. The problem: what is the thing that learns?

A central difficulty in biological learning is not merely how a synapse can detect pre- and postsynaptic activity. A local site must distinguish at least three logically different questions:

1. **Did I participate in the state that drove the neuron?**
2. **Did the neuron subsequently commit an output event?**
3. **Was that event useful enough that participating structure should be retained or changed?**

Modern eligibility-trace and three-factor theories make this separation explicit: local activity creates an eligibility trace, while a later modulatory factor associated with reward, punishment, novelty, surprise, or another salient event determines whether plasticity is consolidated [1,2].

The question considered here is different: **what is the plastic object?** Many formal models write

```math
\Delta w_j=f(\mathrm{pre}_j,\mathrm{post},c),
```

where `w_j` is a scalar efficacy. Real dendrites are spatially extended dynamical systems. Changes in channel density, membrane kinetics, spine-neck geometry, branch diameter, path length, branching, synapse placement, or AIS properties can alter a transfer function rather than merely multiply one input by one number.

The working hypothesis is therefore:

> **A neuron may learn partly by modifying a shared physical operator whose effective computation depends on where and how it is interrogated.**

The hypothesis contains three separable pieces:

- **addressing:** spatial and temporal properties of incoming activity select a response of the dendritic operator;
- **commitment and consequence:** the AIS converts an extended somatodendritic state into discrete output events, while backpropagating and modulatory signals provide post-event information;
- **operator plasticity:** slower physical changes modify the future family of transfer responses.

None of these components is novel in isolation. The proposed contribution is their synthesis together with an executable program designed to remove the synthesis if simpler explanations suffice.

---

## 2. Biological ground that already exists

### 2.1 Dendrites already implement location- and frequency-dependent transfer

The strongest prior art for the word "address" is ordinary dendritic biophysics. Transfer impedance depends on geometry, membrane state, input location, and frequency. Laudanski et al. showed that spatially distributed dendritic resonance can selectively filter synaptic temporal structure: changing the dendritic input location changes the resonant transfer seen at the soma [3]. Narayanan and Johnston experimentally found spatially distributed changes in intrinsic resonance and excitability associated with long-term potentiation [4]. Zhuchkova, Remme, and Schreiber showed theoretically that non-uniform active conductances can produce different somatic and dendritic resonance profiles [5].

Let `theta` denote morphology and membrane state. A linearized frequency-domain description can be written schematically as

```math
H_\theta(\omega)=
\left[G(\theta)+i\omega C(\theta)+Y_{\rm active}(\omega;\theta)\right]^{-1}.
```

If input enters at spatial location `s` and the soma/AIS is the observed port, then

```math
M_{\rm dend}(s,\omega;\theta)
=P_{\rm AIS}H_\theta(\omega)P_{\rm in}(s)
```

is an address-conditioned transfer response. This notation does not imply that a neuron computes an explicit Fourier transform or stores a symbolic frequency coordinate. Temporal pattern, burst rate, synchrony, oscillatory state, receptor kinetics, and local voltage state can all influence which dynamical response is exposed.

Gate 7 of `ObjektiYksi` is an important warning against overgeneralizing this biological precedent: although frequency is physically meaningful, the current lattice's three-task address code does **not** need frequency. A coordinate can affect the operator without carrying task identity in a particular experiment.

### 2.2 Geometry is part of the computation

Dendritic morphology materially affects neuronal dynamics. Mainen and Sejnowski showed that reconstructed neocortical geometries, under a common channel-density model, can generate markedly different firing patterns [6]. Cuntz, Borst, and Segev developed optimization principles in which dendritic structure reflects competing wiring and signal-transfer objectives [7].

Structural plasticity is also experimentally real. Motor learning can rapidly form and selectively stabilize new dendritic spines [8]; repeated learning can produce clustered spine formation [9]; and branch-specific calcium events can support persistent synaptic plasticity [10]. These observations do not show that real dendrites optimize the objective used in our simulations. They establish the narrower premise that physical structure is dynamic and computationally relevant.

### 2.3 The AIS is a genuine state-to-event boundary

The AIS is a principal site of action-potential initiation and a specialized boundary between somatodendritic and axonal compartments [11,12]. AIS length, position, and molecular composition can themselves change with activity [13]. It is therefore useful to distinguish

```text
extended somatodendritic state
              ->
        AIS commitment
              ->
          axonal event
```

without claiming that the AIS "reads an eigenmode." The AIS simply transforms a voltage trajectory produced by the whole cell into spike timing and probability.

### 2.4 Backpropagating action potentials are event signals, not reward signals

Action potentials initiated near the AIS can backpropagate into dendrites and strongly influence local voltage, calcium, and plasticity [14]. A bAP should not be interpreted as a behavioral error gradient. At most, it supplies information related to the neuron's own output event.

Recent review work emphasizes that backpropagation is not a spatially uniform broadcast: attenuation, amplification, branch failure, frequency dependence, and trial-to-trial variability can make the same somatic event appear differently in different dendritic regions [15]. A more realistic abstract quantity is therefore

```math
r_j=r_j(\text{AIS event},\text{branch state},\text{history}).
```

This makes a branch-conditioned event signal plausible. It does **not** prove that the signal contains useful credit.

### 2.5 Existing dendritic credit-assignment theories are close intellectual ancestors

Urbanczik and Senn proposed a rule in which dendritic state predicts somatic spiking and prediction mismatch drives synaptic plasticity [16]. Richards and Lillicrap reviewed dendritic mechanisms as possible solutions to credit assignment, emphasizing spatially separated feedforward and feedback signals [17]. Eligibility-trace and three-factor rules already provide a principled distinction between local participation and delayed consequence [1,2].

The present hypothesis should therefore not be advertised as "the solution" to biological credit assignment. Its narrower question is:

> **Can the plastic variable be promoted from a scalar synaptic weight to a physical transfer operator while retaining only locally available participation, postsynaptic event information, delayed consequence, and small physical exploration?**

---

## 3. Executable results motivating the hypothesis

### 3.1 `DendriteAsIteratedFeedbackOperator`

The companion repository tests the dendritic side directly in compartmental models and maintains passive controls.

A passive cable performs ordinary diffusive modal selection. A phenomenological restorative quasi-active extension introduces stable complex poles and non-zero pass bands. In tuned five-tone examples, one input component can become strongly enriched at a bounded somatic/AIS readout. "Eigenmode purifier" is used only operationally here: the distributed operator preferentially transmits one non-zero temporal mode. No information is created and no hidden nonlinear winner-take-all process is implied.

Gate 11 closes a toy causal loop:

```text
mixed cue
 -> dendritic filtering
 -> AIS-like event
 -> event-conditioned return
 -> local persistent edit
 -> erase fast state
 -> replay through changed operator
```

but the event return is explicitly not an adjoint or gradient.

Gates 13-17 move to one reconstructed human L2/3 morphology. Gate 17 represents log-length perturbations along a 1.242 mm soma-to-tip path using smooth orthonormal geometry modes. The first four modes contain 97.88% of measured first-order gradient energy, but 96.73% is the uniform mode alone, so the dominant first-order effect is largely global shortening. In the first eight smooth modes, all measured Hessian eigenvalues are negative and the curvature has effective rank about 5.23 rather than one. A Newton candidate computed in that subspace survives direct nonlinear replay: the counted full step improves the bounded objective by 17.33% and reduces the measured local gradient norm to 19.26% of baseline.

The warranted conclusion is therefore modest but nontrivial:

> **Many locally sensitive geometry coordinates can organize into collective directions, and a task- and observer-relative finite optimum can exist in a collective geometry subspace.**

The same Gate 17 attacks the proposed event-conditioned receipt as structural credit. A raw correlation of about 0.774 between receipt-derived mode projections and the measured geometry gradient initially looks attractive. A circular-shift null kills the interpretation: shifted receipts are typically at least as predictive, with empirical `p=0.953`, and an ordinary reciprocal-return control performs similarly.

Thus the repo establishes an important separation:

```text
local geometry sensitivity
        !=
collective structural curvature
        !=
credit assignment
```

The geometry result survives. The tested receipt-credit mechanism does not.

See `DendriteAsIteratedFeedbackOperator/GATE17_GEOMETRY_MODES.md` for the complete protocol and claim boundary.

### 3.2 `ObjektiYksi`: one material, many effective operators

`ObjektiYksi` tests a deliberately non-biological resonant lattice. Its persistent state is a vector of positive edge couplings `g`. At angular frequency `omega`,

```math
H_g(\omega)=\left[K(g)-\omega^2I+i\omega\Gamma\right]^{-1}.
```

For source position `s` and chosen ports,

```math
M_{\rm eff}(s,\omega;g)
=P_{\rm out}H_g(\omega)P_{\rm in}(s).
```

`M_eff` is not stored. One material state generates different effective transfers depending on how it is interrogated.

The sequence through Gate 6 is:

- **Gate 0:** analytic positive-control sensitivity carves a strong resonant receiver preference while mean material is held fixed;
- **Gate 1:** the resulting function is not captured by an ordinary `1/g` shortest path;
- **Gate 2:** the analytic edge-gradient sign is removed; strictly local conservative material dithers are retained or reverted using one bounded scalar consequence;
- **Gate 3:** the same impoverished rule learns against whichever output is currently strongest, without receiving competitor identity;
- **Gate 4:** frozen material generalizes much better over carrier frequency than over source landing position;
- **Gate 5:** long-budget worst-case learning makes one material support several source-conditioned relations simultaneously;
- **Gate 6:** a small online selector discovers task-specific physical addresses while the shared material is slowly changing.

Gate 6 initially looked like evidence for learned joint position-frequency addressing. Its audit already contained the correction: all three tasks use the same oracle frequency `omega=1.35`; the distinct code is in source position.

### 3.3 Gate 7 kills the joint position-frequency claim

Gate 7 preregistered three address spaces before reading the result:

```text
FULL:            3 source rows x 3 frequencies = 9 addresses
POSITION ONLY:   3 source rows at omega=1.35
FREQUENCY ONLY:  center source x 3 frequencies
```

All conditions use the same bounded observer floor and matched slow material proposal schedules. An ablation counts as matching `FULL` only if it reaches the same median rank-1 fraction and at least 90% of the full model's median final worst-task target/strongest-competitor ratio.

The result is decisive:

```text
                         FULL      POSITION ONLY   FREQUENCY ONLY
final worst ratio       1.70577       1.70236         0.999996
rank-1 fraction         1.00000       1.00000         0.666667
relative to FULL        1.00000       0.99800         0.586243
```

Therefore

```text
verdict = position_dominated_joint_not_needed
```

Gate 7 removes a preferred interpretation rather than rescuing it. Frequency remains a physical parameter of the operator, but it is unnecessary for the current three-task address code.

### 3.4 Gate 8 kills the fixed native-multiplexer explanation

After Gate 7, the next simple explanation is that upper, center, and lower source positions merely select their geometrically aligned outputs. Gate 8 therefore removes the selector and fixes frequency at `omega=1.35`. The three sources are permanently assigned a deliberately non-native three-cycle:

```text
upper source  -> lower target
center source -> upper target
lower source  -> center target
```

Each candidate material edit remains a transfer of a fixed amount of material between adjacent edges. The structural learner receives only the worst of the three bounded target-versus-strongest-competitor utilities. It does not receive the identity of the worst relation, competitor identity, or an edge gradient.

The crossed map begins strongly wrong:

```text
blank crossed worst ratio    0.109779
blank crossed rank-1         0/3
```

After 1,200 proposals across three seeds:

```text
median crossed worst ratio   1.522495
worst seed                   1.362632
crossed rank-1               3/3 in every seed
random crossed drift         0.067250, 0/3 rank-1
no write                     0.109779, 0/3 rank-1
native identity control      1.781158
```

The crossed relation improves by 13.87x over its blank value and beats matched random drift by about 22.64x. The identity of the currently worst relation changes about 90 times and its strongest competitor about 95 times in the median run.

The preregistered verdict is

```text
crossed_permutation_survives_native_geometry_attack
```

Thus the source coordinate is not merely a fixed upper/center/lower multiplexer. Learned material can rewrite what a particular source location does downstream.

### 3.5 Gate 9 shows that the best physical address itself moves

Gate 8 changes output behavior at fixed sources. Gate 9 asks a stronger audit question without modifying the learning rule. After rerunning Gate 8, all three candidate source locations are scanned for each target. Define

```math
s^*(t;g)=\arg\max_s U(t,s;g),
```

where `U` is the same bounded target-versus-strongest-output-competitor utility.

On blank material, every target prefers its vertically native source and none prefers the source assigned by the crossed mapping. After crossed learning, the exact optimum moves:

```text
upper target:   upper source  -> center source
center target:  center source -> lower source
lower target:   lower source  -> upper source
```

This happens for **all three targets in all three seeds**. The median minimum assigned-versus-old-native utility advantage is `+1.00487`; the weakest target in the weakest seed remains `+0.62322`. Identity-trained, random-drift, and no-write controls retain the native optimum for all three targets.

The preregistered verdict is

```text
address_optima_move_with_learned_material
```

So in this toy

```math
\arg\max_s U(t,s;g_{\rm blank})
\ne
\arg\max_s U(t,s;g_{\rm learned}).
```

This is the strongest current evidence for the phrase **deformable address landscape**. The substrate supplies physical coordinates, but learning can change which coordinate best exposes a requested relation.

The caveat is important: Gate 8 directly rewards the three crossed source-target relations. Gate 9 therefore demonstrates programmable movement of address optima, not autonomous discovery of where an address ought to move.

---

## 4. Hypothesis: the dendrite as an addressed physical operator

Let `theta` collect slowly persistent physical variables of a dendritic arbor: local conductances, channel densities, spine-neck parameters, branch geometry, and related structural state. Let `a` denote an input address broadly construed:

```math
a=(s,\text{temporal pattern},\text{oscillatory state},\text{synchrony},\ldots).
```

A fast computation can be written abstractly as

```math
x\mapsto M(a;\theta)x.
```

The AIS then converts the resulting somatodendritic trajectory into an event:

```math
z=\mathcal A_{\rm AIS}[M(a;\theta)x,\text{context}].
```

Forward activity can leave a local eligibility trace `e_j`. The output event, filtered by the current branch state, can contribute a local event-related quantity `r_j`. A delayed modulatory or behavioral consequence supplies a third factor `c`.

A minimal local coincidence is therefore

```math
e_j\,r_j\,c.
```

But this product is **not assumed to contain a gradient sign**. The experiments motivate a weaker division of labor:

```text
eligibility / event information  -> WHERE change is plausible
small physical fluctuation       -> WHICH WAY was tried
later consequence                -> WHETHER it should persist
```

For a physical degree of freedom `theta_j`, the local substrate could explore

```math
\theta'_j=\theta_j+\delta
```

and later retain or reverse that perturbation. Biological candidates for such exploration include stochastic receptor/channel turnover, spine motility, cytoskeletal remodeling, local growth/retraction, and other small fluctuations. The paper does **not** claim that any one of these implements the lattice algorithm.

The complete hypothetical loop is therefore

```text
spatiotemporal input
        ↓
address-conditioned dendritic operator
        ↓
AIS commitment
        ↓
local eligibility + branch-conditioned event signal
        +
delayed third factor
        ↓
small physical exploration
        ↓
retain / reverse / consolidate
        ↓
changed physical operator
        ↓
future inputs see a changed family of responses
```

`ObjektiYksi` demonstrates that the last three computational-physics arrows are possible under a global scalar oracle. `DendriteAsIteratedFeedbackOperator` demonstrates that morphology and membrane state really do define rich transfer operators in a neuron-like model. Neither repo currently supplies the missing biological credit mechanism connecting the two.

---

## 5. What is and is not new

The proposal is **not** novel in any isolated claim that:

- dendrites filter signals in a location- and frequency-dependent manner [3-5];
- morphology affects neuronal dynamics [6,7];
- bAPs influence dendritic plasticity [14,15];
- dendrites may participate in credit assignment [16,17];
- eligibility traces plus third factors can bridge delayed consequences [1,2];
- learning changes synaptic and dendritic structure [8-10];
- local feedback can configure physical wave systems.

The potentially distinct synthesis is:

> **The persistent object is a deformable physical operator rather than only a collection of scalar synaptic efficacies; input coordinates expose different responses of that operator; output commitment and local state provide event-related information; and consequence-selected physical exploration can reshape both transfer relations and the landscape of useful input coordinates.**

The word "potentially" matters. Novelty has not been exhaustively established, and the biological mechanism required to make the synthesis local remains open.

---

## 6. Falsifiers and current status

The project is useful only if preferred interpretations are allowed to die.

### 6.1 Does location matter beyond scalar attenuation?

If responses from different dendritic locations reduce to one common temporal kernel multiplied by a scalar, an addressed-operator description adds little.

**Kill condition:** after matching somatic amplitude, different input locations produce indistinguishable transfer shape, phase, latency, frequency response, and AIS event statistics.

**Status:** still open biologically; the dendritic toy already shows non-scalar branch kernels.

### 6.2 Does a branch-conditioned event return add information beyond a global spike flag?

If replacing local bAP/return waveforms with a spatially uniform postsynaptic-event signal gives equivalent learning, the special "receipt" machinery is unnecessary.

**Kill condition:** uniform event notification performs as well as the full local return under matched eligibility and third-factor signals.

**Status:** the first receipt candidate took a serious loss. Gate 17's event-conditioned spatial pattern fails its shift-null control. The theory must not cite that receipt as solved credit assignment.

### 6.3 Do collective shape modes survive removal of global scale?

Gate 17 is dominated by global shortening. A stronger geometry claim requires zero-mean shape perturbations such as

```math
\langle\delta\log L,1\rangle_L=0.
```

**Kill condition:** once global scale is fixed, finite collective optima disappear or fail across cells/paths.

**Status:** planned as the next hard attack in `DendriteAsIteratedFeedbackOperator`.

### 6.4 Does consequence selection beat blind structural drift?

**Kill condition:** matched random retain/revert produces comparable improvements.

**Status:** rejected in the `ObjektiYksi` lattice through multiple gates, including Gate 8 where random crossed drift falls while consequence-selected carving produces all three crossed relations. Still open for a biologically local dendritic mechanism.

### 6.5 Is joint position-frequency addressing actually necessary?

**Kill condition:** position-only matches the full position-frequency address space.

**Status:** **killed.** Gate 7 finds position-only retains 99.80% of the full margin and identical rank-1 performance. The paper no longer claims joint position-frequency task addressing for the current toy.

### 6.6 Is spatial addressing merely a native geometric multiplexer?

**Kill condition:** native source-output alignment learns, but a preregistered crossed spatial permutation cannot.

**Status:** **survives the attack.** Gate 8 learns the deliberately wrong three-cycle from a 0.1098x worst relation to 1.5225x median, all relations rank-1 in every seed.

### 6.7 Can learned material move the best physical address?

**Kill condition:** transfer values change, but the audit-best source remains determined by blank geometry.

**Status:** **survives the attack.** Gate 9 moves all three audit optima from native sources to crossed-assigned sources in every seed; controls retain native optima.

### 6.8 The remaining central kill: can credit be made genuinely local?

The current lattice learner evaluates a globally defined consequence after each candidate physical change. That is intentionally an oracle compared with biological plasticity.

A biologically interesting version must replace

```text
try local change
-> globally evaluate whole task
-> keep / revert
```

with locally available variables whose causal sufficiency can be attacked against strong controls.

**Kill condition:** no locally available combination of eligibility, postsynaptic event information, delayed modulatory consequence, and realistic structural fluctuations can outperform simpler three-factor synaptic rules or blind drift when the task requires operator-level structural change.

This is now the main theoretical bottleneck.

---

## 7. Relation to artificial learning systems

A conventional task-conditioned model can be written schematically as

```math
y=W_qx.
```

The physical-operator alternative is

```math
y=M(a;\theta)x,
```

where `theta` is one shared persistent substrate and `a` is a low-dimensional way of interrogating it.

Slow learning changes `theta` and therefore changes many effective operators simultaneously:

```math
\theta\rightarrow\theta'
\quad\Rightarrow\quad
\{M(a;\theta)\}_{a\in\mathcal A}
\rightarrow
\{M(a;\theta')\}_{a\in\mathcal A}.
```

Gates 8-9 make the distinction sharper. Learning can alter not only the transfer associated with a fixed coordinate but also

```math
\arg\max_a U(q,a;\theta),
```

the address that best exposes a requested relation.

This creates both opportunity and danger. One physical substrate may implicitly provide many transformations without storing a dense matrix for each address, but any local material change can interfere with many behaviors at once. Continual-learning, collision, and observability problems therefore arise naturally.

No memory-density, energy-efficiency, learning-speed, or hardware advantage over digital neural networks is currently established. Those are engineering questions, not consequences of the mathematical framing.

---

## 8. Limitations

`ObjektiYksi` is a reciprocal damped resonant lattice, not a neuron. Its source ports and task utilities are externally specified. It repeatedly solves the full physical field after every candidate structural perturbation, and a global scalar decides keep/revert. Calling the material update "local" refers to **where the physical perturbation occurs**, not to complete biological locality of the information used to accept it.

`DendriteAsIteratedFeedbackOperator` contains a biological morphology but phenomenological channel models and experimenter-defined observer objectives. A finite geometry optimum in that model does not imply that the real cell developed by optimizing the same quantity.

The strongest attempted receipt-credit mechanism is negative. This should remain prominent because it prevents a common conceptual slide from

```text
bAP reaches dendrite
```

to

```text
therefore dendrite has a useful error signal.
```

That inference has not been earned.

Structural variables also span very different mechanisms and timescales. Receptor trafficking, ion-channel redistribution, spine-neck changes, spine birth/death, branch remodeling, axonal/AIS plasticity, and circuit rewiring should not be collapsed into one `theta` in a biological claim.

Finally, the current lattice demonstrations are very small. Gate 8 proves one three-relation permutation, not arbitrary routing or scalable universal computation. Gate 9 proves externally driven address-optimum movement, not autonomous discovery of a task coordinate system.

---

## 9. Conclusion

The strongest defensible version of the hypothesis is no longer the original picturesque claim that a dendrite "purifies an eigenmode" and then receives an error back from the AIS.

It is more precise:

> **A dendritic arbor is an extended physical transfer operator. Spatial and temporal properties of incoming activity can expose different responses of that operator. The AIS converts the resulting state into output events. Slow changes in morphology and membrane state can change the operator itself. If local eligibility, event-related information, delayed consequence, and physical exploration can jointly select useful structural changes, then learning need not be represented solely as updates to scalar synaptic weights.**

The computational-physics half is becoming concrete. `ObjektiYksi` now shows that one conserved substrate can be locally restructured into a non-native multi-relation mapping and that the restructuring can move which physical input coordinate is optimal for each requested output. `DendriteAsIteratedFeedbackOperator` shows that real morphology can support nontrivial, collective operator geometry in a compartmental model.

The biological bridge remains unresolved. The first proposed receipt failed. That failure focuses the question rather than weakening it into vagueness:

```text
What locally available biological information can select
useful changes to an extended dendritic operator?
```

If no such information survives appropriate controls, the structural credit-assignment theory dies and the operator framing remains only a descriptive account of dendritic computation. If a local mechanism does survive, the result would connect three bodies of work that are usually treated separately: dendritic filtering, three-factor credit assignment, and structural plasticity.

That is now the experiment worth trying to kill.

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
