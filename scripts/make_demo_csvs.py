"""Write three ready-to-upload demo captures to data/demo/ (5 minutes of office traffic each).

    python scripts/make_demo_csvs.py

1_safe.csv          normal traffic only                         -> the console should say "No attack"
2_medium_risk.csv   reconnaissance and a password-guessing run  -> PortScan + BruteForce incidents
3_high_risk.csv     flood, denial of service, botnet, intrusion -> DDoS, DoS, Botnet, WebAttack, Infiltration
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xgnn_ids.data.synthetic import ATTACKS, generate_traffic, inject_attack  # noqa: E402

OUT = ROOT / "data" / "demo"
DURATION = 300.0
SCENARIOS = {
    "1_safe.csv": dict(seed=31, attacks=[], intensity=1.0),
    "2_medium_risk.csv": dict(seed=32, attacks=["PortScan", "BruteForce"], intensity=0.8),
    "3_high_risk.csv": dict(seed=33, attacks=["DDoS", "Botnet", "WebAttack", "DoS", "Infiltration"], intensity=1.2),
}


def build(seed: int, attacks: list[str], intensity: float) -> "pd.DataFrame":  # noqa: F821
    df = generate_traffic("enterprise", DURATION, seed=seed, attacks=[])
    for i, a in enumerate(attacks):                      # same placement as the console's simulator
        lo, hi = ATTACKS[a][1]
        length = min((lo + hi) / 2, DURATION / (len(attacks) + 1) * 1.5, DURATION * 0.5)
        t0 = DURATION * (i + 1) / (len(attacks) + 1) - length / 2
        df = inject_attack(df, a, max(t0, 1.0), length, "enterprise", seed, intensity, attack_seed=seed * 100 + i)
    df["segment"] = "demo"
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cfg in SCENARIOS.items():
        df = build(**cfg)
        df.to_csv(OUT / name, index=False)
        print(f"{name}: {len(df):,} flows, labels {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
