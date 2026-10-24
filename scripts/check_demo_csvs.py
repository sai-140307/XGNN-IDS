"""Run the shipped default model on data/demo/*.csv and write what the console would show to data/demo/VERDICTS.md."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xgnn_ids.inference import Detector  # noqa: E402

MIN_CONF = 0.9
det = Detector.load(ROOT / "models")
lines = [f"# Demo capture verdicts ({det.display_name}, alert threshold {MIN_CONF})", ""]
for f in sorted((ROOT / "data" / "demo").glob("*.csv")):
    res = det.detect(pd.read_csv(f))
    d = res.df
    alerts = d[(d.pred_label != "BENIGN") & (d.confidence >= MIN_CONF)]
    inc = det.incidents(res, MIN_CONF)
    lines += [f"## {f.name}", f"flows {len(d)}, windows {len(res.graphs)}, alerts {len(alerts)}",
              "incidents: " + (", ".join(f"{i['class']} ({i['flows']} flows)" for i in inc) or "none"),
              "", "| true label | flows | alerted as (conf >= 0.9) |", "|---|---|---|"]
    for lab, grp in d.groupby("label"):
        a = grp[(grp.pred_label != "BENIGN") & (grp.confidence >= MIN_CONF)]
        lines.append(f"| {lab} | {len(grp)} | {a.pred_label.value_counts().to_dict()} |")
    lines.append("")
(ROOT / "data" / "demo" / "VERDICTS.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
