# Results

Data: **synthetic enterprise network (seed 0)** - 63,476 flows in 90 graph windows (20s each). Chronological split: train 38,404 / val 11,173 / test 13,899 flows.

Environment: Python 3.11.17, PyTorch 2.14.1+cpu, CPU (2 threads). Total run time 5.9 min.

## 1. Detection performance (test split)

| Model | Accuracy | Macro-F1 | Weighted-F1 | DDoS F1 | FPR | Detection rate | Latency ms/window | Flows/s |
|---|---|---|---|---|---|---|---|---|
| Flow-MLP (no graph) | 0.8912 | 0.7967 | 0.8848 | 0.8554 | 0.0497 | 0.9467 | 0.2 | 4,202,367 |
| GCN **(default)** | 0.9982 | 0.9991 | 0.9982 | 0.9969 | 0.0010 | 0.9976 | 1.0 | 747,440 |
| GAT (edge-attention) | 0.9976 | 0.9986 | 0.9976 | 0.9959 | 0.0021 | 0.9973 | 2.1 | 370,211 |
| E-GraphSAGE (directed) | 0.9961 | 0.9971 | 0.9961 | 0.9934 | 0.0012 | 0.9942 | 1.6 | 496,649 |
| E-GraphSAGE (no node stats) | 0.9862 | 0.9513 | 0.9849 | 0.9915 | 0.0022 | 0.9920 | 1.5 | 498,932 |
| Random Forest (flow features) | 0.9339 | 0.9385 | 0.9343 | 0.9001 | 0.0485 | 0.9702 | 25.7 | 30,090 |
| Random Forest + graph statistics | 0.9953 | 0.9971 | 0.9953 | 0.9911 | 0.0003 | 0.9922 | 12.7 | 60,963 |
| Gradient Boosting (flow features) | 0.9316 | 0.9377 | 0.9328 | 0.9005 | 0.0622 | 0.9791 | 8.2 | 94,274 |

### Per-class results - GCN

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| BENIGN | 0.9968 | 0.9990 | 0.9979 | 5850 |
| DDoS | 0.9994 | 0.9945 | 0.9969 | 3450 |
| DoS | 1.0000 | 1.0000 | 1.0000 | 1739 |
| PortScan | 1.0000 | 1.0000 | 1.0000 | 1119 |
| BruteForce | 1.0000 | 1.0000 | 1.0000 | 224 |
| Botnet | 1.0000 | 1.0000 | 1.0000 | 165 |
| WebAttack | 1.0000 | 1.0000 | 1.0000 | 237 |
| Infiltration | 0.9964 | 1.0000 | 0.9982 | 1115 |

![per-class F1](per_class_f1.png)

![confusion matrix](confusion_matrix.png)

## 2. Cross-domain generalisation (enterprise -> IoT, no retraining)

Benign-traffic drift (PSI vs. training): in-domain test 0.006, IoT network 0.622 (>0.25 = significant drift).

| Model | In-domain macro-F1 | IoT macro-F1 | IoT DDoS F1 | IoT FPR | IoT detection rate |
|---|---|---|---|---|---|
| Flow-MLP (no graph) | 0.7967 | 0.5546 | 0.8646 | 0.7460 | 0.9594 |
| GCN | 0.9991 | 0.6712 | 0.9962 | 0.7218 | 0.9983 |
| GAT (edge-attention) | 0.9986 | 0.6908 | 0.9952 | 0.6716 | 0.9941 |
| E-GraphSAGE (directed) | 0.9971 | 0.6897 | 0.9739 | 0.6606 | 0.9985 |
| E-GraphSAGE (no node stats) | 0.9513 | 0.3447 | 0.2275 | 0.6773 | 0.8873 |
| Random Forest (flow features) | 0.9385 | 0.7319 | 0.8869 | 0.5406 | 0.9772 |
| Random Forest + graph statistics | 0.9971 | 0.8809 | 0.9836 | 0.0010 | 0.9421 |
| Gradient Boosting (flow features) | 0.9377 | 0.7160 | 0.8887 | 0.6647 | 0.9798 |

### Few-shot adaptation to the new network

Fine-tuned on the first 7,762 IoT flows (12 windows, about 4 min of labelled traffic) plus a replay sample of the original training windows; evaluated on the last 28,063 IoT flows.

| Model | IoT macro-F1 before | IoT FPR before | IoT macro-F1 after | IoT FPR after | Enterprise macro-F1 after |
|---|---|---|---|---|---|
| GCN | 0.6735 | 0.7186 | 0.8274 | 0.0011 | 0.9959 |
| Flow-MLP (no graph) | 0.5404 | 0.7461 | 0.6476 | 0.1226 | 0.6929 |
| Random Forest + graph statistics | 0.8470 | 0.0009 | 0.8941 | 0.0004 | 0.9981 |

## 3. Zero-day test (leave-one-attack-out)

Model: GCN. The hidden family is present in the training traffic but never labelled. *Detected* = classified as any attack or flagged as uncertain.

| Unseen family | Test flows | Flagged as attack | Uncertain | Detected (either) | Most common label | Benign false alarms |
|---|---|---|---|---|---|---|
| DDoS | 3,450 | 0.714 | 0.965 | 0.965 | DoS | 0.033 |
| DoS | 1,739 | 1.000 | 1.000 | 1.000 | WebAttack | 0.137 |
| PortScan | 1,119 | 1.000 | 1.000 | 1.000 | DDoS | 0.029 |
| BruteForce | 224 | 1.000 | 1.000 | 1.000 | DoS | 0.034 |
| Botnet | 165 | 0.000 | 0.000 | 0.000 | BENIGN | 0.034 |
| WebAttack | 237 | 0.983 | 0.717 | 1.000 | DoS | 0.053 |
| Infiltration | 1,115 | 0.691 | 0.985 | 0.985 | PortScan | 0.033 |

## 4. Adversarial robustness

White-box PGD (10 steps) on attacker-controllable features (timing, volume, packet sizes); ε in standard deviations. Values = share of attack flows still detected.

| Model | ε=0.0 | ε=0.25 | ε=0.5 | ε=1.0 | ε=2.0 |
|---|---|---|---|---|---|
| Flow-MLP (no graph) | 0.947 | 0.881 | 0.789 | 0.622 | 0.454 |
| GCN | 0.998 | 0.992 | 0.985 | 0.963 | 0.789 |
| GAT (edge-attention) | 0.997 | 0.993 | 0.984 | 0.969 | 0.712 |
| E-GraphSAGE (directed) | 0.994 | 0.986 | 0.978 | 0.939 | 0.683 |

![robustness](adversarial_robustness.png)

## 5. Explanation quality

56 correctly detected attack flows. Feature explainers rank the 5 most important items of the explanation space (27 flow features + 11 statistics of each endpoint host). Fidelity is measured in log-odds of the explained class: Fidelity+ = drop when the top items are reset to the training mean (higher = they are necessary), Fidelity- = drop when *only* they are kept (lower = they are sufficient). Flip rate = share of alerts whose class changes when the top items are reset. Sparsity = share of items not needed for 80% of the attribution mass. Stability = top-k overlap (Jaccard) under small input noise.

### Feature attribution

| Method | Fidelity+ (log-odds) | Fidelity- (log-odds) | Flip rate | Sparsity | Stability | Runtime ms |
|---|---|---|---|---|---|---|
| Integrated Gradients | 5.806 | 1.424 | 0.29 | 0.792 | 0.942 | 61.5 |
| SHAP | 4.295 | 2.535 | 0.16 | 0.871 | 0.631 | 1313.9 |
| GNNExplainer (feature mask) | 3.225 | 3.018 | 0.16 | 0.429 | 0.922 | 388.7 |
| Random | 0.441 | 6.130 | 0.00 | 0.477 | 0.064 | 0.1 |

### Structural explanation (top 20% of neighbouring flows removed / kept)

| Method | Fidelity+ (log-odds) | Fidelity- (log-odds) | Sparsity | Stability | Runtime ms |
|---|---|---|---|---|---|
| Edge Integrated Gradients | 3.253 | 1.973 | 0.659 | 0.904 | 68.0 |
| GNNExplainer | 0.181 | 3.443 | 0.565 | 0.103 | 389.9 |
| Same-destination heuristic | 2.351 | 1.539 | nan | nan | 0.0 |
| Random | 0.215 | 2.913 | nan | nan | 0.0 |

Example narratives: [example_explanations.md](example_explanations.md)

## Training curves

![training curves](training_curves.png)
