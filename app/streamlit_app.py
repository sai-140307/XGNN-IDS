"""XGNN-IDS console host.

The interface itself is a custom web app in ``app/console`` (HTML/CSS/D3). This
script hides Streamlit's own chrome, serves that console full-screen as a
bidirectional component, and acts as its backend: it runs the attack
simulator, parses uploaded captures, runs the GNN, and computes explanations
on request.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))

from payload import build_payload, eval_payload, explanation_payload  # noqa: E402
from xgnn_ids.config import ATTACK_CLASSES  # noqa: E402
from xgnn_ids.data.loaders import normalise  # noqa: E402
from xgnn_ids.data.synthetic import ATTACKS, generate_traffic, inject_attack  # noqa: E402
from xgnn_ids.inference import Detector  # noqa: E402

MODELS_DIR = Path(os.environ.get("XGNN_MODELS_DIR", ROOT / "models"))
RESULTS = Path(os.environ.get("XGNN_RESULTS_DIR", ROOT / "results")) / "metrics.json"
MIN_CONF = 0.5
MAX_UPLOAD_ROWS = 300_000
DEFAULT_SCENARIO = {"profile": "enterprise", "minutes": 10, "attacks": ["DDoS", "PortScan", "Botnet", "BruteForce"],
                    "intensity": 1.0, "seed": 42}

st.set_page_config(page_title="XGNN-IDS", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stSidebar"], [data-testid="collapsedControl"], footer, #MainMenu { display: none !important; }
.stApp, [data-testid="stAppViewContainer"] { background: #EEF1EE; }
[data-testid="stMain"], section.main { padding: 0 !important; }
.block-container, [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stElementContainer"]:has(iframe), .element-container:has(iframe) { margin: 0 !important; }
iframe { display: block; width: 100% !important; height: 100vh !important; border: 0; }
</style>""", unsafe_allow_html=True)

console = components.declare_component("xgnn_console", path=str(APP / "console"))
NAMES = {"mlp": "Flow-MLP (no graph)", "gcn": "GCN", "gat": "GAT (edge-attention)", "egraphsage": "E-GraphSAGE (directed)",
         "egraphsage_nostats": "E-GraphSAGE (no node stats)", "rf": "Random Forest (flow features)",
         "rf_graphstats": "Random Forest + graph statistics", "hgb": "Gradient Boosting (flow features)"}


@st.cache_resource(show_spinner=False)
def load_detector(name: str) -> Detector:
    return Detector.load(MODELS_DIR, name)


@st.cache_data(show_spinner=False)
def simulate(profile: str, minutes: int, attacks: tuple, intensity: float, seed: int) -> pd.DataFrame:
    duration = minutes * 60.0
    df = generate_traffic(profile, duration, seed=seed, attacks=[])
    for i, a in enumerate(attacks):
        lo, hi = ATTACKS[a][1]
        length = min((lo + hi) / 2, duration / (len(attacks) + 1) * 1.5, duration * 0.5)
        t0 = duration * (i + 1) / (len(attacks) + 1) - length / 2
        df = inject_attack(df, a, max(t0, 1.0), length, profile, seed, intensity, attack_seed=seed * 100 + i)
    return df


@st.cache_data(show_spinner=False)
def evaluation(mtime: float) -> dict | None:
    if not RESULTS.exists():
        return None
    return eval_payload(json.loads(RESULTS.read_text()), NAMES)


def parse_upload(name: str, b64: str) -> pd.DataFrame:
    raw = base64.b64decode(b64)
    if name.lower().endswith((".pcap", ".pcapng", ".cap")):
        from xgnn_ids.capture.flowmeter import pcap_to_flows

        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as fh:
            fh.write(raw)
        try:
            df = pcap_to_flows(fh.name, max_packets=2_000_000)
        finally:
            os.unlink(fh.name)
        if df.empty:
            raise ValueError("no IPv4 TCP/UDP/ICMP packets were found in that capture")
        return df
    return normalise(pd.read_csv(io.BytesIO(raw), low_memory=False, nrows=MAX_UPLOAD_ROWS), segment=Path(name).stem)


def ensure_model() -> None:
    if Detector.available_models(MODELS_DIR):
        return
    st.markdown("### No trained model found")
    st.write("Train a quick model here (2-3 minutes on a laptop CPU), or run `python -m xgnn_ids train`.")
    if st.button("Train a quick model now", type="primary"):
        from xgnn_ids.training.experiment import run_experiment

        with st.spinner("Training"):
            run_experiment(None, str(RESULTS.parent), str(MODELS_DIR), quick=True, models=["gcn", "egraphsage", "mlp"],
                           baselines=["rf"], cross_domain=False, zero_day=False, adversarial=False, xai=False,
                           log=lambda *_: None)
        st.cache_resource.clear()
        st.rerun()
    st.stop()


ensure_model()
ss = st.session_state
art = Detector.read_artifacts(MODELS_DIR)
ss.setdefault("scenario", dict(DEFAULT_SCENARIO))
ss.setdefault("model", art["default_model"])
ss.setdefault("source", "sim")
ss.setdefault("explanations", {})
ss.setdefault("explain_req", None)
ss.setdefault("last_nonce", None)
ss.setdefault("error", None)

# ------------------------------------------------------------------ handle the console's last request
ev = ss.get("console")
if isinstance(ev, dict) and ev.get("nonce") and ev["nonce"] != ss.last_nonce:
    ss.last_nonce = ev["nonce"]
    ss.error = None
    action = ev.get("action")
    try:
        if action == "simulate":
            attacks = [a for a in ev.get("attacks", []) if a in ATTACK_CLASSES]
            ss.scenario = {"profile": ev.get("profile") if ev.get("profile") in ("enterprise", "iot") else "enterprise",
                           "minutes": int(min(max(int(ev.get("minutes", 10)), 3), 30)), "attacks": attacks,
                           "intensity": float(min(max(float(ev.get("intensity", 1.0)), 0.3), 2.0)),
                           "seed": int(ev.get("seed", 42)) % 100000}
            ss.source = "sim"
            ss.explain_req = None
        elif action == "upload":
            df = parse_upload(str(ev.get("name", "capture.csv")), ev.get("data", ""))
            ss.upload_df, ss.upload_name = df, str(ev.get("name"))
            ss.upload_id = hashlib.sha1(ev.get("data", "")[:200_000].encode()).hexdigest()[:12]
            ss.source = "upload"
            ss.explain_req = None
        elif action == "model":
            if ev.get("name") in art["models"]:
                ss.model = ev["name"]
                ss.explain_req = None
        elif action == "explain":
            ss.explain_req = {"row": int(ev["row"]), "shap": bool(ev.get("shap"))}
    except Exception as exc:  # shown inside the console
        ss.error = f"Could not complete that request: {exc}"

# ------------------------------------------------------------------ detection (cached per model + data)
det = load_detector(ss.model)
if ss.source == "upload" and "upload_df" in ss:
    data_key = ("upload", ss.upload_id)
    source_meta = {"source": ss.upload_name, "scenario": None}
    df_in = ss.upload_df
else:
    sc = ss.scenario
    data_key = ("sim", sc["profile"], sc["minutes"], tuple(sc["attacks"]), sc["intensity"], sc["seed"])
    source_meta = {"source": "Simulated " + ("IoT network" if sc["profile"] == "iot" else "enterprise LAN"),
                   "scenario": sc, "profile": sc["profile"]}
    df_in = simulate(sc["profile"], sc["minutes"], tuple(sc["attacks"]), sc["intensity"], sc["seed"])

full_key = (ss.model, data_key)
if ss.get("base_key") != full_key:
    res = det.detect(df_in)
    meta = {"model": det.display_name, "model_name": det.name, "window_s": det.graph_cfg.window_s,
            "uncertainty_threshold": float(det.threshold),
            "models": [{"name": k, "display": v["display"]} for k, v in art["models"].items()],
            "key": hashlib.sha1(repr(full_key).encode()).hexdigest()[:16], **source_meta}
    ss.res = res
    ss.base_payload = build_payload(res, det.incidents(res, MIN_CONF), MIN_CONF, meta)
    # drift over the whole capture: all flows the model considers benign, compared with training benign traffic
    if det.drift is not None:
        import numpy as np

        benign = [g.edge_attr[res.df["pred_label"].to_numpy()[g.flow_rows] == "BENIGN"] for g in res.graphs]
        benign = np.vstack(benign) if benign else np.zeros((0, 1))
        if len(benign) >= 200:
            ss.base_payload["meta"]["drift"] = round(float(det.drift.psi(benign)[0]), 4)
    ss.base_key = full_key
    ss.explanations = {}

# ------------------------------------------------------------------ explanation on request
explanation = None
req = ss.explain_req
if req is not None:
    ck = (full_key, req["row"], req["shap"])
    if ck not in ss.explanations and 0 <= req["row"] < len(ss.res.df):
        try:
            methods = ("ig", "edge_ig", "counterfactual") + (("shap",) if req["shap"] else ())
            ss.explanations[ck] = explanation_payload(det.explain(ss.res, req["row"], methods), req["row"], req["shap"])
        except Exception as exc:
            ss.error = f"Could not explain that flow: {exc}"
    explanation = ss.explanations.get(ck)

payload = {**ss.base_payload, "explanation": explanation, "nonce": ss.last_nonce, "error": ss.error,
           "eval": evaluation(RESULTS.stat().st_mtime if RESULTS.exists() else 0.0)}
console(payload=payload, key="console", default=None)
