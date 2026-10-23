# Architecture and method

```
packets ──► flow meter ──┐
                         ├─► canonical flows ─► temporal graph builder ─► edge-level GNN ─► predictions
flow CSVs ─► loaders ────┘     (27 features)      (20 s windows,                │               │
                                                   hosts = nodes,               │               ├─► incidents + MITRE + firewall rules
                                                   flows = edges)               │               └─► drift monitor (PSI)
                                                                                └─► explainers: IG · SHAP · GNNExplainer · counterfactuals · evidence
```

## 1. Flow representation

Every source is normalised to one schema (`xgnn_ids/data/features.py`): endpoints, ports, protocol, duration,
packets and bytes in each direction, TCP flag counts and mean inter-arrival time. From it 27 edge features are
derived - log-scaled volumes and rates, mean packet sizes, flag counts, a *no response* indicator, the
response/request byte ratio, protocol one-hots and destination-port categories (web, remote login, DNS, SMB,
well-known, registered, dynamic).

## 2. Temporal graphs

Flows are cut into windows of `window_s = 20` seconds per capture segment. Each window is a directed
multigraph: one node per IP address, one edge per flow. Nodes carry 11 window-level statistics
(flows sent/received, distinct peers in each direction, distinct ports used/targeted, bytes, share of
half-open inbound connections, share of unanswered outbound connections, private-address flag). They are
computed only from the window's own flows, so there is no label leakage.

Splitting is **chronological within each segment** (60/20/20 by time), so the model is always evaluated on
traffic recorded after its training period - random splits leak near-duplicate flows between train and test
and inflate scores (see the literature review, Luša et al. 2025).

## 3. Models (`xgnn_ids/models/gnn.py`)

All are edge classifiers `f(x, edge_index, edge_attr, edge_weight) -> logits[E, 8]`, written in plain
PyTorch (no PyTorch Geometric needed), with a shared head `MLP([h_src || h_dst || e_flow])`.

| Model | Message passing |
|---|---|
| Flow-MLP | none - ablation showing what the flow alone gives |
| GCN | symmetric-normalised convolution on the undirected host graph; node inputs include mean in/out flow features |
| GAT | GATv2-style multi-head attention; the flow's features enter the attention score and the message |
| **E-GraphSAGE (directed)** | separate mean aggregation over *inbound* and *outbound* flows (`[h_u ‖ e_uv]` messages), residual + LayerNorm. Direction matters: a DDoS victim has huge fan-in, a scanner huge fan-out |
| E-GraphSAGE (no node stats) | same, but nodes start from a constant vector - the model must discover the structure itself |

Training: AdamW, cosine schedule, class-weighted cross-entropy (weights ∝ 1/√frequency), gradient clipping,
early stopping on validation macro-F1. Several windows are packed into one disjoint-union batch.

Baselines: Random Forest and Gradient Boosting on flow features, and Random Forest with the same per-host
graph statistics appended (a strong *hand-crafted graph features* baseline).

## 4. Explainability (`xgnn_ids/explain/`)

An L-layer GNN's prediction for flow (u→v) only depends on the (L+1)-hop neighbourhood of u and v. All
explainers run on that **computation subgraph** - a unit test checks the prediction there equals the
full-graph prediction exactly. To evaluate many feature variants at once (SHAP), the subgraph is replicated
B times as a disjoint union and run in one forward pass.

| Explainer | Question it answers | Cost |
|---|---|---|
| Integrated Gradients | which properties of this flow pushed it to the class | 1 batched pass, ~10-50 ms |
| SHAP (KernelSHAP) | same, with Shapley guarantees | hundreds of model evaluations, ~0.5-5 s |
| GNNExplainer | which neighbouring flows the decision relied on (edge mask) | 100 optimisation steps |
| Feature counterfactual | smallest change to the flow's mutable features that makes it benign | gradient search |
| Structural counterfactual | how many important neighbouring flows must disappear for the flow to look benign | a few passes |
| Graph evidence | plain-language facts: fan-in, fan-out, port spread, repetition, beacon regularity, payload size | statistics |

Explanation **quality** is measured, not assumed (`explain/quality.py`): fidelity+ (prediction drop when the
top-k items are removed), fidelity- (drop when only the top-k are kept), sparsity (share of items not needed
for 80% of the attribution mass), stability (top-k Jaccard under small input noise) and runtime - each
compared with a random explainer, and for the structural case with a *same-destination* heuristic.

## 5. From alerts to action (`xgnn_ids/analysis/`)

* **Incident correlation** groups alerts by family and the entity that ties a campaign together
  (victim for DDoS, scanner for scans, C2 address for botnets, internal host for infiltration) and splits
  on quiet gaps > 60 s.
* Each incident carries MITRE ATT&CK techniques, a response recommendation and **suggested iptables /
  nftables rules** (rate limits, ipsets, quarantine of internal hosts) - for review, never applied
  automatically.
* **Drift monitor**: Population Stability Index of the benign-predicted traffic of each window against the
  training benign traffic. PSI > 0.25 means the model is operating outside its training distribution.

## 6. Robustness experiments (`xgnn_ids/training/robustness.py`)

* **Zero-day**: leave-one-attack-out. The hidden family's flows stay in the training graphs but are
  unlabelled; at test time a flow counts as detected if classified as any attack or if the model's
  confidence falls below the 5th percentile of validation confidences.
* **Adversarial evasion**: white-box PGD on the features an attacker controls in their own flows (timing,
  volume, packet sizes) with an L∞ budget ε in standard-deviation units, targeted at the BENIGN class.
* **Cross-domain**: trained on the enterprise network, tested on an IoT network with different addressing,
  services, benign behaviour (periodic MQTT telemetry that resembles beaconing) and attack variants.

## 7. The synthetic data generator (`xgnn_ids/data/synthetic.py`)

Built because public datasets are gigabytes and cannot ship with a demo or run in CI. It reproduces the
*relational* signatures of each family while making single flows overlap with benign traffic
(HTTP-flood requests look like page visits, brute-force logins look like SSH sessions), includes hard
negatives (a benign flash crowd hitting the web server, periodic IoT telemetry), and places episodes of every
family in all three chronological splits. Real datasets plug into the identical pipeline (`docs/DATASETS.md`).
