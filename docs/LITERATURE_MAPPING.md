# How the implementation maps to the literature review

| Reviewed study | Idea used here | Gap it left that this project addresses |
|---|---|---|
| Zhong et al. (2024), GNN-IDS survey | GCN, GAT, GraphSAGE families compared on the same graphs; flows as edges | Explainability and real-time deployment named as open challenges → explainers + latency/throughput for every model |
| Baahmed et al. (2025), E-GraphSAGE + GNNExplainer | Edge-level classification, GNNExplainer for edge predictions | Single dataset, static graphs, no explanation metrics → temporal windows, cross-domain test, fidelity/sparsity/stability/runtime |
| Khan et al. (2024), X-CBA (SHAP + LIME) | Several explanation perspectives side by side | Consistency and cost between explainers unknown → IG vs SHAP vs GNNExplainer measured on the same alerts with runtimes |
| Azeez et al. (2026), GNN + MHA + SHAP for 5G-SDN | Multi-head attention (GAT), reporting latency and throughput | Domain-specific → enterprise and IoT profiles, cross-domain evaluation |
| Luša et al. (2025), TE-G-SAGE | Temporal edge-aware GraphSAGE, chronological evaluation, inductive learning | Single dataset, SHAP overhead → direction-aware E-GraphSAGE, IG as lightweight alternative, drift monitoring |
| Mohammad et al. (2026), adversarially robust GNN-IDS | Adversarial evaluation of GNN detectors | No explanation-quality validation; controlled adversaries → PGD on attacker-controllable features + measured explanations |
| Xu et al. (2024), NEGSC self-supervised | Detecting attacks with reduced labels | Not explainable → zero-day (leave-one-attack-out) test with uncertainty flagging; self-supervision listed as future work |
| Wang et al. (2025), FedSTGCN | Spatio-temporal graphs, per-class reporting (minority classes) | High aggregate scores hid weak minority classes → per-class P/R/F1 and macro-F1 reported everywhere |

Additional practical components not present in the reviewed papers: incident correlation, MITRE ATT&CK
mapping, firewall-rule suggestions, counterfactual explanations, live capture and a deployable analyst console.
