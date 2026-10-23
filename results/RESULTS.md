# Results

Data: **synthetic enterprise network (seed 0)** - 63,476 flows in 90 graph windows (20s each). Chronological split: train 38,404 / val 11,173 / test 13,899 flows.

Environment: Python 3.11.17, PyTorch 2.14.1+cpu, CPU (2 threads). Total run time 6.5 min.

## 1. Detection performance (test split)

| Model | Accuracy | Macro-F1 | Weighted-F1 | DDoS F1 | FPR | Detection rate | Latency ms/window | Flows/s |
|---|---|---|---|---|---|---|---|---|
| Flow-MLP (no graph) | 0.8649 | 0.7537 | 0.8563 | 0.8395 | 0.0391 | 0.9119 | 0.2 | 4,238,364 |
| GCN | 0.9969 | 0.9976 | 0.9969 | 0.9951 | 0.0012 | 0.9955 | 1.0 | 738,782 |
| GAT (edge-attention) | 0.9968 | 0.9985 | 0.9968 | 0.9942 | 0.0019 | 0.9958 | 2.1 | 359,870 |
| E-GraphSAGE (directed) **(default)** | 0.9956 | 0.9969 | 0.9956 | 0.9924 | 0.0012 | 0.9933 | 1.5 | 498,780 |
| E-GraphSAGE (no node stats) | 0.9745 | 0.8593 | 0.9670 | 0.9880 | 0.0050 | 0.9991 | 1.6 | 473,650 |
| Random Forest (flow features) | 0.9339 | 0.9385 | 0.9343 | 0.9001 | 0.0485 | 0.9702 | 25.7 | 30,092 |
| Random Forest + graph statistics | 0.9953 | 0.9971 | 0.9953 | 0.9911 | 0.0003 | 0.9922 | 15.9 | 48,552 |
| Gradient Boosting (flow features) | 0.9316 | 0.9377 | 0.9328 | 0.9005 | 0.0622 | 0.9791 | 7.1 | 109,342 |

### Per-class results - E-GraphSAGE (directed)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| BENIGN | 0.9908 | 0.9988 | 0.9948 | 5850 |
| DDoS | 0.9994 | 0.9855 | 0.9924 | 3450 |
| DoS | 1.0000 | 1.0000 | 1.0000 | 1739 |
| PortScan | 1.0000 | 1.0000 | 1.0000 | 1119 |
| BruteForce | 1.0000 | 1.0000 | 1.0000 | 224 |
| Botnet | 1.0000 | 1.0000 | 1.0000 | 165 |
| WebAttack | 0.9957 | 0.9831 | 0.9894 | 237 |
| Infiltration | 0.9964 | 1.0000 | 0.9982 | 1115 |

![per-class F1](per_class_f1.png)

![confusion matrix](confusion_matrix.png)

## 2. Cross-domain generalisation (enterprise -> IoT, no retraining)

Benign-traffic drift (PSI vs. training): in-domain test 0.006, IoT network 0.622 (>0.25 = significant drift).

| Model | In-domain macro-F1 | IoT macro-F1 | IoT DDoS F1 | IoT FPR | IoT detection rate |
|---|---|---|---|---|---|
| Flow-MLP (no graph) | 0.7537 | 0.5439 | 0.8631 | 0.7288 | 0.9353 |
| GCN | 0.9976 | 0.6821 | 0.9882 | 0.7172 | 0.9981 |
| GAT (edge-attention) | 0.9985 | 0.6874 | 0.9943 | 0.6677 | 0.9926 |
| E-GraphSAGE (directed) | 0.9969 | 0.6855 | 0.9777 | 0.6606 | 0.9991 |
| E-GraphSAGE (no node stats) | 0.8593 | 0.3372 | 0.1989 | 0.6800 | 0.8849 |
| Random Forest (flow features) | 0.9385 | 0.7319 | 0.8869 | 0.5406 | 0.9772 |
| Random Forest + graph statistics | 0.9971 | 0.8809 | 0.9836 | 0.0010 | 0.9421 |
| Gradient Boosting (flow features) | 0.9377 | 0.7160 | 0.8887 | 0.6647 | 0.9798 |

## 3. Zero-day test (leave-one-attack-out)

Model: E-GraphSAGE (directed). The hidden family is present in the training traffic but never labelled. *Detected* = classified as any attack or flagged as uncertain.

| Unseen family | Test flows | Flagged as attack | Uncertain | Detected (either) | Most common label | Benign false alarms |
|---|---|---|---|---|---|---|
| DDoS | 3,450 | 0.000 | 0.757 | 0.757 | BENIGN | 0.010 |
| DoS | 1,739 | 1.000 | 1.000 | 1.000 | WebAttack | 0.095 |
| PortScan | 1,119 | 1.000 | 0.925 | 1.000 | Infiltration | 0.012 |
| BruteForce | 224 | 0.871 | 1.000 | 1.000 | DoS | 0.017 |
| Botnet | 165 | 0.000 | 0.000 | 0.000 | BENIGN | 0.016 |
| WebAttack | 237 | 0.983 | 0.983 | 0.983 | DoS | 0.032 |
| Infiltration | 1,115 | 0.944 | 0.787 | 0.998 | PortScan | 0.029 |

## 4. Adversarial robustness

White-box PGD (10 steps) on attacker-controllable features (timing, volume, packet sizes); ε in standard deviations. Values = share of attack flows still detected.

| Model | ε=0.0 | ε=0.25 | ε=0.5 | ε=1.0 | ε=2.0 |
|---|---|---|---|---|---|
| Flow-MLP (no graph) | 0.912 | 0.829 | 0.742 | 0.592 | 0.463 |
| GCN | 0.996 | 0.990 | 0.982 | 0.963 | 0.787 |
| GAT (edge-attention) | 0.996 | 0.990 | 0.981 | 0.962 | 0.684 |
| E-GraphSAGE (directed) | 0.993 | 0.985 | 0.977 | 0.928 | 0.671 |

![robustness](adversarial_robustness.png)

## 5. Explanation quality

56 correctly detected attack flows, top-5 features. Fidelity+ (higher = the highlighted features are necessary), Fidelity- (lower = they are sufficient), sparsity (higher = more concise), stability (Jaccard of top-k under noise).

### Feature attribution

| Method | Fidelity+ | Fidelity- | Sparsity | Stability | Runtime ms |
|---|---|---|---|---|---|
| Integrated Gradients | 0.0194 | -0.0030 | 0.689 | 0.856 | 81.6 |
| SHAP | 0.0188 | -0.0025 | 0.805 | 0.747 | 1582.2 |
| GNNExplainer (feature mask) | 0.0135 | 0.0026 | 0.325 | 0.911 | 512.8 |
| Random | -0.0008 | 0.0198 | 0.464 | 0.112 | 0.0 |

### Structural explanation (which neighbouring flows matter, top-10 flows)

| Method | Fidelity+ | Fidelity- | Sparsity | Stability | Runtime ms |
|---|---|---|---|---|---|
| GNNExplainer | -0.0000 | 0.1161 | 0.949 | 0.363 | 506.3 |
| Same-destination heuristic | 0.0052 | 0.1125 | nan | nan | 0.0 |
| Random | 0.0009 | 0.1143 | nan | nan | 0.0 |

Example narratives: [example_explanations.md](example_explanations.md)

## Training curves

![training curves](training_curves.png)
