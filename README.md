# XGNN-IDS: Explainable Graph Neural Network Intrusion Detection

[![CI](https://github.com/sai-140307/XGNN-IDS/actions/workflows/ci.yml/badge.svg)](https://github.com/sai-140307/XGNN-IDS/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-ee4c2c)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Network traffic is turned into a graph every 20 seconds - **hosts are nodes, flows are edges** - and a graph
neural network classifies every flow as benign or one of seven attack families (DDoS, DoS, port scan,
brute force, botnet, web attack, infiltration). Every alert comes with an explanation an analyst can act on:
which properties of the flow mattered, which *other* flows in the network made it suspicious, what would
have had to change for it to look benign, and a suggested response.

*BCSE308P Computer Networks project, Vellore Institute of Technology - guided by Dr. Aaseegha M D (SCORE).
Team: Sai Chandra Reddy (24BYB0011), N Rishi Pranav (24BYB0082).*

![The XGNN-IDS console: a DDoS replayed on the perimeter map](docs/screenshots/console-map.png)

<!-- RESULTS_START -->
## Results at a glance

Measured on a chronological split (train on the past, test on the future) of 63,476 flows from the built-in
enterprise-network simulation; full tables, method and figures in [results/RESULTS.md](results/RESULTS.md).

| | Flow-only model | Graph model (GCN) |
|---|---|---|
| Macro-F1 across 8 classes | 0.939 (Random Forest) | **0.999** |
| DDoS F1 | 0.900 | **0.997** |
| False-positive rate on benign traffic | 4.9% | **0.10%** |
| Attacks still detected under white-box evasion (ε = 1 std) | 62% (Flow-MLP) | **96%** |
| Inference time per 20 s window (CPU) | 25.7 ms | **1.0 ms** |

- **Explanations are measured, not assumed.** Integrated Gradients over the flow and its two hosts:
  fidelity+ 5.8 log-odds vs 0.4 for a random explainer, 94% stability, ~60 ms per alert. For *which
  neighbouring flows mattered*, edge Integrated Gradients reaches fidelity+ 3.3 and 90% stability, while
  GNNExplainer (the usual choice in GNN-IDS papers) was no better than random (0.18) - so it is kept only
  for comparison.
- **Zero-day:** with one family never labelled, 6 of 7 families are still flagged (≥96% of their flows);
  low-volume botnet beaconing is the one that slips through.
- **New network:** on an unseen IoT network every model loses accuracy and the drift monitor fires
  (PSI 0.62). Fine-tuning on ~4 minutes of local traffic cuts GCN false alarms from 72% to 0.1% while keeping
  office-network accuracy at 0.996.

![per-class F1](results/per_class_f1.png)

Example of an explanation the console produces (Botnet alert):

> **Botnet** detected with 100% confidence on flow `192.168.10.159:54747 -> 66.137.26.58:6667/TCP`.
> - **Network context:** 66.137.26.58 received 12 flows from 8 distinct sources in this 20s window.
> - **Neighbouring flows:** 9 of 299 neighbouring flows carry 80% of the structural evidence; 100% of it
>   comes from flows towards the same destination.
> - Removing the 13 most important neighbouring flows from message passing would make it look BENIGN.
> - **MITRE ATT&CK:** T1071 · **Response:** isolate the internal hosts, block the C2 address.

More: [results/example_explanations.md](results/example_explanations.md).

*Numbers come from synthetic traffic and should not be compared with papers that use CIC-IDS2017 directly;
the same pipeline trains on real datasets (see [docs/DATASETS.md](docs/DATASETS.md)).*
<!-- RESULTS_END -->

## What it does

| Capability | Details |
|---|---|
| Graph-based detection | Directed temporal flow graphs; edge-level **E-GraphSAGE** (direction-aware), **GAT** (edge-feature attention), **GCN**; flow-only MLP, Random Forest and Gradient Boosting baselines |
| Explainability | **Integrated Gradients** and **SHAP** over the flow and its two hosts, **edge Integrated Gradients** and **GNNExplainer** (which neighbouring flows matter), **counterfactuals** (what would make it benign), plain-language **network evidence** |
| Explanation quality, measured | Fidelity+, fidelity-, sparsity, stability and runtime for every explainer, against a random baseline |
| From alerts to action | **Incident correlation** (thousands of alerts → a few incidents), **MITRE ATT&CK** mapping, suggested **iptables / nftables** rules |
| Robustness | **Zero-day** leave-one-attack-out test, **adversarial** evasion (white-box PGD), **cross-domain** test on an unseen IoT network with **few-shot adaptation**, **concept-drift** monitor (PSI) |
| Data sources | Built-in realistic traffic generator, CIC-IDS2017 / CSE-CIC-IDS2018 / CICDDoS2019, NF-* NetFlow datasets, pcap files, **live capture** |
| Interfaces | Custom analyst console (perimeter map, replay, incident case files, attack simulator, capture upload), REST API (FastAPI), command line, live sensor, Docker |

## The console

The interface is built around one idea: show the network, not tables about it. Internal hosts sit inside a
drawn perimeter, the internet sits on the outer ring, and every flow is a line between them, coloured only if
it is an attack. You replay a capture window by window like a recording; a DDoS shows as hundreds of outside
hosts converging on one server, a botnet as several inside machines reaching out to one rare address.

- **Incidents** on the right group thousands of alerts into a few campaigns, each with a MITRE ATT&CK
  technique, a response recommendation and firewall rules you can download.
- **Case files** explain any flow you click: plain-language network evidence, which properties of the flow
  and its two hosts mattered, which neighbouring flows carried the evidence (highlighted on the map), and
  what would have had to change for it to look benign.
- **Change scenario** injects any mix of the seven attack families into an enterprise or IoT network, or
  loads your own CICFlowMeter / NetFlow CSV or pcap.
- **How well it works** and **How it works** are designed report pages built from the latest training run.

| Incident view | Case file |
|---|---|
| ![incident](docs/screenshots/console-incident.png) | ![explanation](docs/screenshots/console-explanation.png) |

## Architecture

```mermaid
flowchart LR
    A[Packets<br/>live NIC / pcap] --> B[Flow meter]
    C[Flow CSVs<br/>CIC / NetFlow] --> D
    B --> D[Canonical flows<br/>27 features]
    D --> E[Temporal graph builder<br/>hosts = nodes, flows = edges]
    E --> F[Edge-level GNN<br/>E-GraphSAGE / GAT / GCN]
    F --> G[Incidents + MITRE ATT&CK<br/>+ firewall rules]
    F --> H[Explainers<br/>IG, SHAP, GNNExplainer,<br/>counterfactuals, evidence]
    E --> I[Drift monitor PSI]
    G --> J[Analyst console / REST API]
    H --> J
    I --> J
```

Method details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). How each part answers a gap from the literature
review: [docs/LITERATURE_MAPPING.md](docs/LITERATURE_MAPPING.md).

## Quick start

```bash
git clone https://github.com/sai-140307/XGNN-IDS.git && cd XGNN-IDS
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app/streamlit_app.py      # analyst console (trained models are included in models/)
```

Command line:

```bash
python -m xgnn_ids train                       # full experiment -> models/ and results/RESULTS.md
python -m xgnn_ids train --quick               # ~3 minute version
python -m xgnn_ids train --data day1.csv day2.csv   # real CIC / NetFlow CSVs (see docs/DATASETS.md)
python -m xgnn_ids detect capture.pcap         # alerts.csv + incidents.json
python -m xgnn_ids explain flows.csv           # explain the most confident alert
sudo python -m xgnn_ids live --iface eth0      # live sensor
uvicorn api.main:app --port 8000               # REST API, docs at /docs
pytest                                         # 57 tests
```

## Project layout

```
xgnn_ids/
  data/        synthetic traffic generator, dataset loaders, feature engineering
  graph/       temporal graph construction, chronological split
  models/      GCN, GAT, E-GraphSAGE, Flow-MLP (plain PyTorch)
  training/    training loop, baselines, metrics, robustness experiments, report
  explain/     IG, SHAP, GNNExplainer, counterfactuals, evidence, quality metrics
  analysis/    incident correlation, firewall rules, drift monitor
  capture/     flow meter, pcap reader, live sniffer
  inference.py Detector API used by the app, API and CLI
app/           Streamlit analyst console
api/           FastAPI service
tests/         pytest suite (models, explainers, loaders, API, dashboard)
docs/          architecture, datasets, deployment, lab demo, viva notes
results/       metrics, figures and example explanations from the last training run
models/        trained weights + preprocessing artifacts
```

## Documentation

- [Architecture and method](docs/ARCHITECTURE.md)
- [Using real datasets](docs/DATASETS.md) (CIC-IDS2017, CICDDoS2019, NetFlow)
- [Deployment](docs/DEPLOYMENT.md) (Streamlit Cloud, Docker, API)
- [Live lab demonstration](docs/LAB_DEMO.md)
- [Viva preparation](docs/VIVA.md)

## References

1. Zhong, M. et al. (2024). A Survey on Graph Neural Networks for Intrusion Detection Systems: Methods, Trends and Challenges.
2. Azeez, S. D., Ilyas, M., Ahmed, S. R. (2026). Graph Neural Networks with Multi-Head Attention and SHAP-Based Explainability for Robust, Interpretable, and High-Throughput Intrusion Detection in 5G-Enabled Software Defined Networks. https://doi.org/10.32604/cmc.2026.074930
3. Xu, R. et al. (2024). Applying Self-supervised Learning to Network Intrusion Detection for Network Flows with Graph Neural Network. https://arxiv.org/abs/2403.01501
4. Baahmed, M., Benaddi, F., Bouhorma, M. (2025). Using Graph Neural Networks for the Detection and Explanation of Network Intrusions. https://hal.science/hal-04909504
5. Khan, M. A., Alhazmi, F., Alhazmi, A. (2024). X-CBA: Explainability Aided CatBoosted Anomal-E for Intrusion Detection System. https://arxiv.org/abs/2402.00839
6. Mohammad, U. G. et al. (2026). An optimized graph neural network approach for robust and explainable IoT intrusion detection against adversarial attacks. https://doi.org/10.1038/s41598-026-48715-9
7. Wang, Y. et al. (2025). FedSTGCN: a novel federated spatiotemporal graph learning-based network intrusion detection method for the Internet of Things. https://doi.org/10.1631/FITEE.2400932
8. Graph-based federated learning approach for intrusion detection in IoT networks (2025). https://doi.org/10.1038/s41598-025-25175-1
9. Luša, R., Pintar, D., Vranić, M. (2025). TE-G-SAGE: Explainable Edge-Aware Graph Neural Networks for Network Intrusion Detection. https://www.mdpi.com/2673-3951/6/4/165

Methods implemented from: Lo et al. (2022) E-GraphSAGE; Ying et al. (2019) GNNExplainer; Sundararajan et al.
(2017) Integrated Gradients; Lundberg & Lee (2017) SHAP; Brody et al. (2022) GATv2; Kipf & Welling (2017) GCN;
Madry et al. (2018) PGD.

## License

MIT
