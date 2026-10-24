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
flows 4414, windows 3, alerts 2657
incidents: DDoS (2227 flows), Infiltration (381 flows), Botnet (7 flows), WebAttack (42 flows)

| true label | flows | alerted as (conf >= 0.9) |
|---|---|---|
| BENIGN | 1704 | {'DDoS': 2} |
| Botnet | 7 | {'Botnet': 7} |
| DDoS | 2278 | {'DDoS': 2225} |
| Infiltration | 383 | {'Infiltration': 381} |
| WebAttack | 42 | {'WebAttack': 42} |

