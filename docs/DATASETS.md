# Using real datasets

The pipeline reads any of these formats directly; the format is detected from the column names.

| Dataset | Files to download | Notes |
|---|---|---|
| **CIC-IDS2017** | `GeneratedLabelledFlows.zip` (a.k.a. *TrafficLabelling*) | Use these, **not** `MachineLearningCSV.zip` - the ML version drops IP addresses and timestamps, so no graph can be built. |
| **CSE-CIC-IDS2018** | per-day CSVs from the AWS bucket (`Processed Traffic Data for ML Algorithms`) | Some days lack IPs; the days with `Src IP`/`Dst IP` columns work. |
| **CICDDoS2019** | `CSV-01-12.zip`, `CSV-03-11.zip` | All reflection/exploitation DDoS variants map to the `DDoS` class. |
| **NF-UNSW-NB15 / NF-ToN-IoT / NF-BoT-IoT / NF-CSE-CIC-IDS2018 (v1-v3)** | UQ NetFlow datasets | `Attack` column is mapped to families; `TCP_FLAGS` is decoded. |
| Your own captures | `.pcap` / `.pcapng`, or CICFlowMeter CSVs | pcaps go through the built-in flow meter; labels are `UNKNOWN`. |

Download links: CIC datasets at <https://www.unb.ca/cic/datasets/> (free registration form), NetFlow
datasets at <https://staff.itee.uq.edu.au/marius/NIDS_datasets/>.

## Label mapping

Raw labels are mapped to eight families (`xgnn_ids/data/loaders.py`, `LABEL_RULES`):

| Family | Example raw labels |
|---|---|
| BENIGN | BENIGN, Benign, Normal |
| DDoS | DDoS, DrDoS_DNS/LDAP/NTP/SNMP/SSDP/MSSQL/NetBIOS, Syn, UDP-lag, TFTP, Portmap, LOIC, HOIC |
| DoS | DoS Hulk, GoldenEye, slowloris, Slowhttptest, Heartbleed |
| PortScan | PortScan, scanning, Reconnaissance |
| BruteForce | FTP-Patator, SSH-Patator, password, Bruteforce |
| Botnet | Bot, backdoor |
| WebAttack | Web Attack - Brute Force / XSS / Sql Injection, injection, xss |
| Infiltration | Infiltration, exploits, shellcode, worms, ransomware, mitm |

Labels that do not match (e.g. UNSW `Fuzzers`, `Generic`) become `UNKNOWN` and are ignored by the loss
and the metrics, but the flows still take part in the graph.

## Training on CIC-IDS2017

```bash
# each CSV becomes its own time segment (one capture day)
python -m xgnn_ids train --data data/raw/TrafficLabelling/*.csv --max-rows 400000
```

The split is chronological *within each file*: the first 60% of every day trains, the next 20% validates,
the last 20% tests. This keeps every day's attacks represented while still testing on traffic recorded after
the training period. Note that in CIC-IDS2017 some attacks occur once per day (e.g. Infiltration on
Thursday afternoon), so a chronological split can leave a class absent from one split; the per-class table
in `results/RESULTS.md` shows supports so this is visible.

CIC-IDS2017 timestamps have one-minute resolution in some files; flows sharing a minute keep their file
order, and windows larger than `max_edges` (default 6000 flows) are split into consecutive chunks.

## Cross-dataset evaluation

Train on one dataset and test on another without retraining:

```python
from xgnn_ids.inference import Detector
from xgnn_ids.data.loaders import load_flows
from xgnn_ids.training.metrics import classification_metrics
from xgnn_ids.config import CLASSES

det = Detector.load("models")
res = det.detect(load_flows("data/raw/CICDDoS2019/DrDoS_DNS.csv"))
y = res.df["label"].map({c: i for i, c in enumerate(CLASSES)}).fillna(-1).astype(int).to_numpy()
p = res.df["pred_label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
print(classification_metrics(y, p)["per_class"]["DDoS"])
print("benign-traffic drift:", res.windows["drift_psi"].median())
```
