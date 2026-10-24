# Demo capture verdicts (GAT (edge-attention), alert threshold 0.9)

## 1_safe.csv
flows 2207, windows 3, alerts 0
incidents: none

| true label | flows | alerted as (conf >= 0.9) |
|---|---|---|
| BENIGN | 2207 | {} |

## 2_medium_risk.csv
flows 2044, windows 3, alerts 420
incidents: BruteForce (80 flows), PortScan (340 flows)

| true label | flows | alerted as (conf >= 0.9) |
|---|---|---|
| BENIGN | 1624 | {} |
| BruteForce | 80 | {'BruteForce': 80} |
| PortScan | 340 | {'PortScan': 340} |

## 3_high_risk.csv
flows 4706, windows 3, alerts 2297
incidents: DDoS (1898 flows), Infiltration (356 flows), Botnet (6 flows), WebAttack (35 flows)

| true label | flows | alerted as (conf >= 0.9) |
|---|---|---|
| BENIGN | 1704 | {'DDoS': 1} |
| Botnet | 6 | {'Botnet': 6} |
| DDoS | 1901 | {'DDoS': 1897} |
| DoS | 704 | {'DoS': 2} |
| Infiltration | 356 | {'Infiltration': 356} |
| WebAttack | 35 | {'WebAttack': 35} |

