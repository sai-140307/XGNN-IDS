"""Export the exact flow data the shipped models were trained and tested on.

    python scripts/export_training_data.py      # -> data/export/*.csv

training_data.csv  office LAN (seed 0, 30 min) + home Wi-Fi (seed 1, 20 min), with the
                   chronological split column (train / val / test) used in results/RESULTS.md
iot_test_data.csv  unseen IoT network (seed 7, 20 min), used only for the cross-network test
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xgnn_ids.config import GraphConfig  # noqa: E402
from xgnn_ids.data.synthetic import generate_traffic  # noqa: E402
from xgnn_ids.graph.builder import build_graphs, chronological_split  # noqa: E402

OUT = ROOT / "data" / "export"
D = 1800  # same as `python -m xgnn_ids train` (xgnn_ids/training/experiment.py)

df = pd.concat([generate_traffic("enterprise", duration_s=D, seed=0),
                generate_traffic("home", duration_s=max(D * 2 // 3, 240), seed=1, episodes_per_attack=3)],
               ignore_index=True)
graphs = build_graphs(df, GraphConfig())
split = np.array([""] * len(df), dtype=object)
window = np.full(len(df), -1)
for name, part in zip(["train", "val", "test"], chronological_split(graphs, (0.6, 0.2, 0.2))):
    for g in part:
        split[g.flow_rows] = name
        window[g.flow_rows] = g.window_id
df["split"], df["window"] = split, window

OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "training_data.csv", index=False)
iot = generate_traffic("iot", duration_s=1200, seed=7)
iot.to_csv(OUT / "iot_test_data.csv", index=False)

print(f"training_data.csv: {len(df):,} flows, {len(graphs)} windows")
print(df.groupby("split")["label"].value_counts().unstack(fill_value=0).loc[["train", "val", "test"]].to_string())
print(f"iot_test_data.csv: {len(iot):,} flows")
