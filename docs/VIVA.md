# Viva preparation: likely questions and answers

**Why model traffic as a graph instead of rows of flows?**
Many attacks are only visible in relationships. One DDoS request or one scan probe looks normal on its own;
hundreds of sources converging on one host, or one host touching hundreds of ports, does not. A row-based
classifier has to be given those relationships as hand-made features; a GNN aggregates them from the
neighbourhood. In our results the flow-only MLP/Random Forest lose most on DDoS, DoS and Botnet - exactly
the families defined by structure.

**What are the nodes and edges?** Nodes are IP addresses active in a 20-second window; edges are individual
flows (directed, source → destination) carrying 27 flow features. Nodes carry 11 statistics of the window.

**Why edge classification and not node classification?** Analysts investigate flows/connections, and one
host can send benign and malicious traffic at the same time (an infected laptop still browses the web).
This is the E-GraphSAGE formulation used by Baahmed et al. (2025) and Luša et al. (2025).

**What is different in your E-GraphSAGE?** It aggregates inbound and outbound flows separately, so the model
knows direction: a DDoS victim has large fan-*in*, a scanner large fan-*out*. Edge features are part of
every message, and there is a residual connection with LayerNorm.

**Why a chronological split?** Random splits put nearly identical flows from the same attack burst in both
train and test, which inflates accuracy. We train on the past and test on the future, which is how an IDS is
deployed.

**Why synthetic data? Isn't that cheating?** The generator exists so the whole system - including CI and the
hosted demo - runs without downloading gigabytes. It deliberately makes single flows overlap with benign
traffic and adds hard negatives (a benign flash crowd, periodic IoT telemetry). The same pipeline reads
CIC-IDS2017, CSE-CIC-IDS2018, CICDDoS2019 and the NF-* NetFlow datasets (`docs/DATASETS.md`). Absolute numbers
on synthetic data should not be compared with papers; the *relative* comparisons (graph vs. no graph,
explainer vs. random, in-domain vs. cross-domain) are the point.

**How do you know the explanations are correct?** We measure them: fidelity+ (removing the top-5 features
the explainer selected should drop the prediction), fidelity- (keeping only them should preserve it),
sparsity, stability under small noise, and runtime - always against a random explainer. A unit test also
checks that the computation subgraph used for explaining gives exactly the full-graph prediction.

**IG vs SHAP vs GNNExplainer - why three?** They answer different questions. IG and SHAP: which properties
of *this flow* mattered (IG is ~100x faster; SHAP has game-theoretic guarantees). GNNExplainer: which *other
flows* mattered - the structural part only a graph model has. Counterfactuals answer "what would have to
change", which analysts find most actionable.

**What is a counterfactual explanation here?** Two kinds. Feature: the smallest change to the flow's
mutable properties (rates, sizes, timing) that flips the prediction to benign - e.g. "benign if it had
received a response". Structural: how many of the most important neighbouring flows would have to vanish
for the flow to look benign - showing whether the decision rests on context or on the flow itself.

**How does it handle new (zero-day) attacks?** Leave-one-attack-out experiment: the family is never
labelled in training. A flow counts as detected if it is classified as any attack or the model is uncertain
(confidence below the 5th percentile seen on validation). Structurally loud attacks are mostly caught under
a different label; the table in RESULTS.md shows which are not.

**Is it robust to an attacker who knows the model?** White-box PGD perturbs what an attacker controls in
their own flows (rates, sizes, timing). Detection falls as the budget grows - for all models - but the
graph models degrade more slowly because the attacker cannot hide the fan-in/fan-out structure by changing
their own flow features.

**What happens when the network changes?** The drift monitor computes the Population Stability Index of
benign-looking traffic per window vs. training. On the IoT network it jumps far above 0.25, telling the
operator to retrain. The cross-domain table quantifies the accuracy loss.

**Is it real-time?** Inference on a 20-second window takes milliseconds on a laptop CPU (latency and
throughput are reported per model), far below the window length. Explanations are computed on demand
for the alert an analyst opens; IG takes tens of milliseconds, SHAP up to a few seconds.

**Can it block attacks?** It recommends iptables/nftables rules per incident (rate limit + ipset for DDoS,
quarantine for infected internal hosts), but never applies them automatically - a false positive would
otherwise become an outage. This is the same human-in-the-loop design commercial NDR tools use.

**Limitations (say them before you are asked).**
- Synthetic headline numbers; real-dataset numbers depend on labelling quality of those datasets.
- IP-level nodes: NAT and IP spoofing blur identities; encrypted payloads are not inspected (only metadata).
- Window-based: very slow attacks spread over hours need longer windows or a temporal memory model.
- Adversarial test is feature-space; real attackers could also change structure (e.g. fewer sources).
- Explanations are post-hoc approximations; we measure their fidelity but they can still mislead.

**Future work.** Temporal GNNs with memory (TGN), self-supervised pre-training on unlabelled traffic
(as in Xu et al. 2024), federated training across sites (Wang et al. 2025), analyst studies of
explanation usefulness.
