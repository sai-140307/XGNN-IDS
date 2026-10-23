"""XGNN-IDS analyst console (Streamlit).

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xgnn_ids.config import ATTACK_CLASSES, CLASSES  # noqa: E402
from xgnn_ids.data.features import is_private_ip  # noqa: E402
from xgnn_ids.data.loaders import normalise  # noqa: E402
from xgnn_ids.data.synthetic import ATTACKS, generate_traffic, inject_attack  # noqa: E402
from xgnn_ids.inference import Detector  # noqa: E402
from xgnn_ids.training.metrics import classification_metrics  # noqa: E402

import os  # noqa: E402

MODELS_DIR = Path(os.environ.get("XGNN_MODELS_DIR", ROOT / "models"))
RESULTS = Path(os.environ.get("XGNN_RESULTS_DIR", ROOT / "results")) / "metrics.json"

COLORS = {
    "BENIGN": "#A3AEB9", "DDoS": "#D64545", "DoS": "#E8833A", "PortScan": "#C9A227",
    "BruteForce": "#8A63D2", "Botnet": "#C2408F", "WebAttack": "#1F9E8F", "Infiltration": "#2F6FDE",
}
INK, MUTED, LINE, PANEL = "#1B2733", "#5F6B78", "#D9DFE5", "#FFFFFF"

st.set_page_config(page_title="XGNN-IDS", page_icon="🛡️", layout="wide")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select { font-family: 'IBM Plex Sans', sans-serif; }
code, .mono { font-family: 'IBM Plex Mono', monospace !important; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
h1 { font-weight: 700; letter-spacing: -0.02em; font-size: 2.0rem !important; margin-bottom: 0 !important; }
h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
.subtitle { color: #5F6B78; font-size: 1.0rem; margin: 0.2rem 0 1.2rem 0; max-width: 75ch; }
.kpi { background: #FFFFFF; border: 1px solid #D9DFE5; border-radius: 6px; padding: 0.75rem 1rem; }
.kpi .v { font-size: 1.6rem; font-weight: 600; color: #1B2733; line-height: 1.2; }
.kpi .l { font-size: 0.85rem; color: #5F6B78; }
.kpi.alert { border-left: 4px solid #D64545; }
.kpi.ok { border-left: 4px solid #3AA675; }
.kpi.warn { border-left: 4px solid #E8833A; }
.incident { border: 1px solid #D9DFE5; border-left: 4px solid var(--c); border-radius: 6px; padding: 0.6rem 0.9rem;
            margin-bottom: 0.5rem; background: #FFFFFF; }
.incident .t { font-weight: 600; color: #1B2733; }
.incident .m { color: #5F6B78; font-size: 0.88rem; }
.chip { display: inline-block; padding: 0.05rem 0.5rem; border-radius: 999px; font-size: 0.8rem; font-weight: 500;
        color: white; background: var(--c); margin-right: 0.3rem; }
.evidence { border-left: 3px solid #2F6FDE; padding: 0.2rem 0 0.2rem 0.8rem; margin: 0.4rem 0; }
.evidence b { color: #1B2733; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ helpers


def kpi(col, value, label, kind=""):
    col.markdown(f'<div class="kpi {kind}"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def chip(label: str) -> str:
    return f'<span class="chip" style="--c:{COLORS.get(label, MUTED)}">{label}</span>'


def fig_layout(fig, height=320, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=12), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    fig.update_xaxes(gridcolor="#EEF1F4", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF1F4", zeroline=False)
    return fig


@st.cache_resource(show_spinner=False)
def load_detector(name: str | None):
    return Detector.load(MODELS_DIR, name)


@st.cache_data(show_spinner=False)
def simulate(profile: str, minutes: int, attacks: tuple, intensity: float, seed: int) -> pd.DataFrame:
    duration = minutes * 60.0
    df = generate_traffic(profile, duration, seed=seed, attacks=[])
    slots = len(attacks)
    for i, a in enumerate(attacks):
        lo, hi = ATTACKS[a][1]
        length = min((lo + hi) / 2, duration / (slots + 1) * 1.5, duration * 0.5)
        t0 = duration * (i + 1) / (slots + 1) - length / 2
        df = inject_attack(df, a, max(t0, 1.0), length, profile, seed, intensity, attack_seed=seed * 100 + i)
    return df


def ensure_model() -> bool:
    if Detector.available_models(MODELS_DIR):
        return True
    st.warning("No trained model was found in `models/`. Train a quick one here (about 2-3 minutes on a laptop CPU) "
               "or run `python -m xgnn_ids train` from a terminal for the full experiment.")
    if st.button("Train a quick model now", type="primary"):
        from xgnn_ids.training.experiment import run_experiment

        log = st.empty()
        lines: list[str] = []

        def _log(msg):
            lines.append(str(msg))
            log.code("\n".join(lines[-14:]))

        run_experiment(None, str(ROOT / "results"), str(MODELS_DIR), quick=True, models=["egraphsage", "gat", "mlp"],
                       baselines=["rf"], cross_domain=False, zero_day=False, adversarial=False, xai=False, log=_log)
        st.cache_resource.clear()
        st.rerun()
    return False


# ------------------------------------------------------------------ sidebar

st.sidebar.markdown("## XGNN-IDS")
st.sidebar.caption("Explainable graph neural network intrusion detection")
if not ensure_model():
    st.stop()

art = Detector.read_artifacts(MODELS_DIR)
model_names = list(art["models"].keys())
default_idx = model_names.index(art["default_model"]) if art["default_model"] in model_names else 0
model_name = st.sidebar.selectbox("Detection model", model_names, index=default_idx,
                                  format_func=lambda n: art["models"][n]["display"])
det = load_detector(model_name)

source = st.sidebar.radio("Traffic source", ["Attack simulator", "Upload flow CSV", "Upload packet capture (pcap)"])
df_in, data_key = None, None
if source == "Attack simulator":
    profile = st.sidebar.selectbox("Network", ["enterprise", "iot"],
                                   format_func=lambda p: {"enterprise": "Enterprise LAN (training domain)",
                                                          "iot": "IoT network (never seen in training)"}[p])
    minutes = st.sidebar.slider("Capture length (minutes)", 5, 30, 10)
    attacks = st.sidebar.multiselect("Attacks to inject", ATTACK_CLASSES, default=["DDoS", "PortScan", "Botnet", "BruteForce"])
    intensity = st.sidebar.slider("Attack intensity", 0.3, 2.0, 1.0, 0.1)
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
    df_in = simulate(profile, minutes, tuple(attacks), intensity, int(seed))
    data_key = ("sim", profile, minutes, tuple(attacks), intensity, int(seed))
elif source == "Upload flow CSV":
    up = st.sidebar.file_uploader("CICFlowMeter, NetFlow (NF-*) or XGNN CSV", type=["csv"])
    st.sidebar.caption("CIC-IDS2017 users: use the TrafficLabelling CSVs, which keep IP addresses.")
    if up is not None:
        try:
            df_in = normalise(pd.read_csv(up, low_memory=False, nrows=300_000), segment=up.name)
            data_key = ("csv", up.name, up.size)
        except ValueError as exc:
            st.error(str(exc))
else:
    up = st.sidebar.file_uploader("pcap / pcapng", type=["pcap", "pcapng", "cap"])
    if up is not None:
        import tempfile

        from xgnn_ids.capture.flowmeter import pcap_to_flows

        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as fh:
            fh.write(up.getvalue())
        df_in = pcap_to_flows(fh.name, max_packets=2_000_000)
        data_key = ("pcap", up.name, up.size)

min_conf = st.sidebar.slider("Alert confidence threshold", 0.3, 0.99, 0.5, 0.01)

st.markdown("# Network intrusion console")
st.markdown('<p class="subtitle">Traffic is turned into a graph every few seconds - hosts are nodes, flows are '
            'edges - and a graph neural network classifies every flow. Each alert comes with the reasons '
            'behind it.</p>', unsafe_allow_html=True)

if df_in is None or len(df_in) == 0:
    st.info("Upload a capture in the sidebar, or switch to the attack simulator.")
    st.stop()

key = (model_name, data_key)
if st.session_state.get("key") != key:
    with st.spinner("Building graphs and classifying flows..."):
        t = time.perf_counter()
        st.session_state["res"] = det.detect(df_in)
        st.session_state["elapsed"] = time.perf_counter() - t
        st.session_state["key"] = key
        st.session_state.pop("explanations", None)
res = st.session_state["res"]
df = res.df
alerts = df[(df["pred_label"] != "BENIGN") & (df["confidence"] >= min_conf)]
incidents = det.incidents(res, min_conf)
has_truth = df["label"].isin(CLASSES).mean() > 0.95

tab_mon, tab_inv, tab_sim, tab_eval, tab_how = st.tabs(
    ["Monitor", "Investigate an alert", "Detection scorecard", "Model evaluation", "How it works"])

# ------------------------------------------------------------------ monitor
with tab_mon:
    c = st.columns(6)
    kpi(c[0], f"{len(df):,}", "flows analysed")
    kpi(c[1], f"{df['src_ip'].nunique() + df['dst_ip'].nunique():,}", "host addresses")
    kpi(c[2], f"{len(alerts):,}", "malicious flows", "alert" if len(alerts) else "ok")
    kpi(c[3], f"{len(incidents)}", "incidents", "alert" if incidents else "ok")
    kpi(c[4], f"{res.windows['latency_ms'].mean():.0f} ms",
        f"per {det.graph_cfg.window_s:.0f}s window ({len(df) / max(st.session_state['elapsed'], 1e-6):,.0f} flows/s end-to-end)")
    psi = res.windows["drift_psi"].dropna()
    drift_score = float(psi.median()) if len(psi) else float("nan")
    drift_text = "n/a" if np.isnan(drift_score) else f"{drift_score:.2f}"
    kind = "ok" if np.isnan(drift_score) or drift_score < 0.1 else "warn" if drift_score < 0.25 else "alert"
    kpi(c[5], drift_text, "traffic drift (PSI)" + ("" if kind == "ok" else " - check model"), kind)

    st.markdown("### Alerts over time")
    w = res.windows.copy()
    w["minute"] = w["t0"] / 60
    fig = go.Figure()
    for cname in ATTACK_CLASSES:
        if w[cname].sum() > 0:
            fig.add_trace(go.Bar(x=w["minute"], y=w[cname], name=cname, marker_color=COLORS[cname]))
    fig.update_layout(barmode="stack", xaxis_title="time (minutes)", yaxis_title="malicious flows per window")
    st.plotly_chart(fig_layout(fig, 260), width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Network graph of one window")
        busiest = int(w["alerts"].idxmax()) if w["alerts"].max() > 0 else 0
        gi = st.slider("Window", 0, len(res.graphs) - 1, busiest,
                       format="%d", help="Each window is one graph snapshot; it opens on the busiest attack window.")
        g = res.graphs[gi]
        wdf = df.iloc[g.flow_rows]
        src, dst = g.edge_index
        pred = wdf["pred_label"].to_numpy()
        conf = wdf["confidence"].to_numpy()
        deg = np.bincount(src, minlength=g.num_nodes) + np.bincount(dst, minlength=g.num_nodes)
        bad_nodes = np.unique(np.concatenate([src[pred != "BENIGN"], dst[pred != "BENIGN"]]))
        keep = set(np.argsort(-deg)[:180].tolist()) | set(bad_nodes[:120].tolist())
        pairs: dict[tuple[int, int], list] = {}
        for s, d, p, cf in zip(src, dst, pred, conf):
            if s in keep and d in keep:
                rec = pairs.setdefault((int(s), int(d)), [0, "BENIGN"])
                rec[0] += 1
                if p != "BENIGN" and cf >= min_conf:
                    rec[1] = p
        G = nx.Graph()
        G.add_nodes_from(keep)
        G.add_edges_from(pairs.keys())
        pos = nx.spring_layout(G, seed=3, k=1.2 / np.sqrt(max(len(G), 1)), iterations=40)
        fig = go.Figure()
        for cname in CLASSES:
            xs, ys = [], []
            for (s, d), (n, lab) in pairs.items():
                if lab == cname:
                    xs += [pos[s][0], pos[d][0], None]
                    ys += [pos[s][1], pos[d][1], None]
            if xs:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=cname, hoverinfo="skip",
                                         line=dict(color=COLORS[cname], width=0.6 if cname == "BENIGN" else 1.4),
                                         opacity=0.45 if cname == "BENIGN" else 0.9))
        nodes = sorted(keep)
        ips = g.node_ips[nodes]
        internal = np.array([is_private_ip(ip) for ip in ips])
        attacked = np.isin(nodes, bad_nodes)
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", name="hosts",
            marker=dict(size=5 + 3 * np.log1p(deg[nodes]), color=np.where(internal, INK, "#FFFFFF"),
                        line=dict(color=np.where(attacked, "#D64545", INK), width=np.where(attacked, 2, 1))),
            text=[f"{ip}<br>{'internal' if i else 'external'} - {deg[n]} flows" for ip, i, n in zip(ips, internal, nodes)],
            hoverinfo="text", showlegend=False))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(fig_layout(fig, 520), width="stretch")
        st.caption(f"Window {gi}: {g.num_nodes:,} hosts, {g.num_edges:,} flows "
                   f"(showing {len(keep)} busiest hosts). Filled dots are internal hosts, hollow dots external; "
                   "red outline = involved in an alert.")
    with right:
        st.markdown("### Incidents")
        if not incidents:
            st.success("No incidents above the confidence threshold.")
        for inc in incidents[:12]:
            colr = COLORS[inc["class"]]
            st.markdown(
                f'<div class="incident" style="--c:{colr}"><div class="t">{chip(inc["class"])} {inc["title"]}</div>'
                f'<div class="m">{inc["id"]} - {inc["severity"]} - {inc["flows"]:,} flows between '
                f'{inc["start"] / 60:.1f} and {inc["end"] / 60:.1f} min - confidence {inc["mean_confidence"]:.2f}<br>'
                f'MITRE ATT&amp;CK {inc["mitre"]}</div></div>', unsafe_allow_html=True)
            with st.expander(f"Response plan for {inc['id']}"):
                st.write(inc["response"])
                st.code("\n".join(inc["firewall"]["iptables"]), language="bash")
        if incidents:
            rules = "\n\n".join("\n".join(i["firewall"]["iptables"]) for i in incidents)
            d1, d2, d3 = st.columns(3)
            d1.download_button("Firewall rules", "#!/bin/sh\n" + rules, "xgnn_firewall_rules.sh", "text/x-sh")
            d2.download_button("Incidents (JSON)", json.dumps(incidents, indent=1), "incidents.json", "application/json")
            d3.download_button("Alerts (CSV)", alerts.to_csv(index=False), "alerts.csv", "text/csv")

    with st.expander("Replay this capture as a live feed"):
        st.caption("Plays the windows back one by one, the way the live sensor (`python -m xgnn_ids live`) reports them.")
        speed = st.slider("Seconds per window", 0.1, 1.5, 0.3)
        if st.button("Start replay"):
            ph_chart, ph_log = st.empty(), st.empty()
            log_lines: list[str] = []
            for i in range(len(res.windows)):
                part = w.iloc[: i + 1]
                f2 = go.Figure()
                for cname in ATTACK_CLASSES:
                    if part[cname].sum() > 0:
                        f2.add_trace(go.Bar(x=part["minute"], y=part[cname], name=cname, marker_color=COLORS[cname]))
                f2.update_layout(barmode="stack", xaxis=dict(range=[0, w["minute"].max() + 0.5]))
                ph_chart.plotly_chart(fig_layout(f2, 240), width="stretch", key=f"replay{i}")
                row = w.iloc[i]
                found = [f"{c}: {int(row[c])}" for c in ATTACK_CLASSES if row[c] > 0]
                stamp = f"t={row['minute']:5.1f} min  {int(row['flows']):5d} flows  "
                log_lines.append(stamp + ("ALERT  " + ", ".join(found) if found else "ok"))
                ph_log.code("\n".join(log_lines[-12:]))
                time.sleep(speed)

# ------------------------------------------------------------------ investigate
with tab_inv:
    if alerts.empty:
        st.info("No alerts to investigate at this confidence threshold.")
    else:
        f1, f2 = st.columns([2, 1])
        cls_filter = f1.multiselect("Attack type", sorted(alerts["pred_label"].unique()),
                                    default=sorted(alerts["pred_label"].unique()))
        order = f2.selectbox("Sort by", ["confidence", "time"])
        view = alerts[alerts["pred_label"].isin(cls_filter)]
        # one representative alert per (class, src, dst) so the list is readable
        view = view.sort_values("confidence", ascending=False).drop_duplicates(["pred_label", "src_ip", "dst_ip"])
        view = view.sort_values("ts" if order == "time" else "confidence", ascending=order == "time").head(400)
        table = view[["ts", "pred_label", "confidence", "src_ip", "src_port", "dst_ip", "dst_port"]].copy()
        table["ts"] = (table["ts"] / 60).round(2)
        table.columns = ["minute", "prediction", "confidence", "source", "src port", "destination", "dst port"]
        st.caption("One row per attack type and host pair. Select a row to explain it.")
        event = st.dataframe(table, hide_index=True, width="stretch", height=260, on_select="rerun",
                             selection_mode="single-row",
                             column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        sel = event.selection.rows if event is not None and hasattr(event, "selection") else []
        row = int(view.index[sel[0]]) if sel else int(view.index[0])
        use_shap = st.toggle("Also compute SHAP (slower, a few seconds)", value=False)

        cache = st.session_state.setdefault("explanations", {})
        ck = (row, use_shap)
        if ck not in cache:
            with st.spinner("Explaining the prediction (Integrated Gradients, GNNExplainer, counterfactuals)..."):
                methods = ("ig", "gnnexplainer", "counterfactual") + (("shap",) if use_shap else ())
                cache[ck] = det.explain(res, row, methods)
        exp = cache[ck]
        p = exp["prediction"]
        fl = exp["flow"]

        st.markdown(f"### {chip(p['label'])} `{fl['src_ip']}:{fl['src_port']}` to `{fl['dst_ip']}:{fl['dst_port']}` "
                    f"({fl['protocol']})", unsafe_allow_html=True)
        if has_truth:
            ok = fl["true_label"] == p["label"]
            st.caption(f"Ground truth in this capture: {fl['true_label']} - "
                       f"{'the model is correct' if ok else 'the model is wrong here'}.")
        a, b = st.columns([3, 2])
        with a:
            st.markdown(exp["narrative"])
        with b:
            probs = pd.Series(p["probs"]).sort_values()
            fig = go.Figure(go.Bar(x=probs.values, y=probs.index, orientation="h",
                                   marker_color=[COLORS[c] for c in probs.index]))
            fig.update_layout(title="Class probabilities", xaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_layout(fig, 280, legend=False), width="stretch")
            if exp.get("evidence"):
                st.markdown("**Network evidence**")
                for ev in exp["evidence"]:
                    st.markdown(f'<div class="evidence"><b>{ev["title"]}</b><br>{ev["text"]}</div>',
                                unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**What the model looked at: the flow and its two hosts**")
            method = st.radio("Attribution method", list(exp["features"].keys()), horizontal=True,
                              label_visibility="collapsed")
            feats = pd.DataFrame(exp["features"][method]).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=feats["attribution"], y=[f"{l} = {v}" for l, v in zip(feats["label"], feats["value"])],
                orientation="h", marker_color=np.where(feats["attribution"] > 0, COLORS[p["label"]], "#A3AEB9")))
            fig.update_layout(xaxis_title=f"contribution to P({p['label']})")
            st.plotly_chart(fig_layout(fig, 330, legend=False), width="stretch")
            st.caption("Coloured bars push towards the predicted attack, grey bars push away. "
                       + f"Computed in {exp['timings_ms'].get(method, 0):.0f} ms.")
        with c2:
            st.markdown("**Which other flows mattered (GNNExplainer)**")
            stc = exp.get("structure")
            if not stc:
                st.info("This flow has no neighbouring flows in its window.")
            else:
                g = res.graphs[int(res.window_of[row])]
                mask = np.asarray(stc["edge_mask"])
                sub = np.asarray(stc["sub_edges"])
                tgt_local = int(np.flatnonzero(sub == int(res.edge_of[row]))[0])
                top = np.argsort(-mask)[:150]
                G = nx.Graph()
                for i in top:
                    s, d = g.edge_index[:, sub[i]]
                    G.add_edge(int(s), int(d))
                pos = nx.spring_layout(G, seed=5, iterations=50)
                fig = go.Figure()
                for i in sorted(top, key=lambda i: mask[i]):
                    s, d = (int(v) for v in g.edge_index[:, sub[i]])
                    is_t = i == tgt_local
                    fig.add_trace(go.Scatter(
                        x=[pos[s][0], pos[d][0]], y=[pos[s][1], pos[d][1]], mode="lines", hoverinfo="skip",
                        showlegend=False, opacity=1.0 if is_t else 0.25 + 0.75 * float(mask[i]),
                        line=dict(color="#D64545" if is_t else "#2F6FDE", width=5 if is_t else 0.5 + 3 * float(mask[i]))))
                nn_ = list(G.nodes)
                fig.add_trace(go.Scatter(x=[pos[n][0] for n in nn_], y=[pos[n][1] for n in nn_], mode="markers",
                                         marker=dict(size=7, color=INK), text=[g.node_ips[n] for n in nn_],
                                         hoverinfo="text", showlegend=False))
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False)
                st.plotly_chart(fig_layout(fig, 330, legend=False), width="stretch")
                st.caption(f"Red: the explained flow. Blue: neighbouring flows, thicker = more important. "
                           f"{stc['important_flows']} of {stc['neighbour_flows']} neighbours are important; "
                           f"{stc['share_same_destination']:.0%} of the weight is on flows to the same destination.")

        report = (f"# XGNN-IDS alert report\n\nModel: {det.display_name}\n\n{exp['narrative']}\n\n## Network evidence\n"
                  + "\n".join(f"- {e['text']}" for e in exp.get("evidence", []))
                  + "\n\n## Feature attribution\n"
                  + "\n".join(f"- {name}: " + ", ".join(f"{x['label']} ({x['attribution']:+.3f})" for x in feats_[:5])
                              for name, feats_ in exp["features"].items()))
        st.download_button("Download alert report (Markdown)", report, f"alert_{row}.md", "text/markdown")

# ------------------------------------------------------------------ scorecard
with tab_sim:
    if not has_truth:
        st.info("The scorecard needs ground-truth labels - use the attack simulator or a labelled dataset.")
    else:
        y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        yp = df["pred_label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        m = classification_metrics(y, yp)
        c = st.columns(4)
        kpi(c[0], f"{m['macro_f1']:.3f}", "macro-F1 on this capture")
        kpi(c[1], f"{m['detection_rate']:.1%}", "attack flows detected")
        kpi(c[2], f"{m['fpr']:.2%}", "benign flows falsely flagged", "ok" if m["fpr"] < 0.01 else "warn")
        kpi(c[3], f"{m['accuracy']:.3f}", "accuracy")
        rows = []
        for cname in CLASSES:
            pc = m["per_class"][cname]
            if pc["support"] == 0:
                continue
            truth = df[df["label"] == cname]
            hit = truth[truth["pred_label"] == cname]
            delay = (hit["ts"].min() - truth["ts"].min()) if len(hit) else np.nan
            rows.append({"class": cname, "flows": pc["support"], "precision": pc["precision"], "recall": pc["recall"],
                         "F1": pc["f1"], "time to first detection (s)": None if cname == "BENIGN" else delay})
        st.markdown("### Per attack type")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ("precision", "recall", "F1")})
        cm = pd.crosstab(df["label"], df["pred_label"]).reindex(index=[c for c in CLASSES if c in set(df["label"])],
                                                                columns=CLASSES, fill_value=0)
        cmn = cm.div(cm.sum(1), axis=0)
        fig = go.Figure(go.Heatmap(z=cmn.values, x=cmn.columns, y=cmn.index, colorscale="Blues", zmin=0, zmax=1,
                                   text=cm.values, texttemplate="%{text}", hovertemplate="%{y} -> %{x}: %{z:.2f}"))
        fig.update_layout(title="Confusion matrix (counts; colour = share of the true class)",
                          xaxis_title="predicted", yaxis_title="true")
        st.plotly_chart(fig_layout(fig, 420, legend=False), width="stretch")

# ------------------------------------------------------------------ evaluation
with tab_eval:
    if not RESULTS.exists():
        st.info("Run `python -m xgnn_ids train` to produce `results/metrics.json`.")
    else:
        R = json.loads(RESULTS.read_text())
        names = {**{k: v["display"] for k, v in art["models"].items()},
                 "rf": "Random Forest (flow features)", "rf_graphstats": "Random Forest + graph statistics",
                 "hgb": "Gradient Boosting (flow features)"}
        st.markdown(f"Experiment data: **{R['data']['source']}**, {R['data']['flows']:,} flows, chronological "
                    "train / validation / test split (the model is always tested on traffic from *after* its "
                    "training period).")
        tab = pd.DataFrame([{"model": names.get(k, k), "macro-F1": v["macro_f1"], "DDoS F1": v["ddos_f1"],
                             "accuracy": v["accuracy"], "false-positive rate": v["fpr"],
                             "detection rate": v["detection_rate"], "ms / window": v["latency_ms_per_window"],
                             "flows / s": v["throughput_flows_per_s"]} for k, v in R["results"].items()])
        st.markdown("### Detection quality on unseen future traffic")
        st.dataframe(tab.sort_values("macro-F1", ascending=False), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in
                                    ("macro-F1", "DDoS F1", "accuracy", "false-positive rate", "detection rate")}
                     | {"ms / window": st.column_config.NumberColumn(format="%.1f"),
                        "flows / s": st.column_config.NumberColumn(format="%,.0f")})
        fig = go.Figure()
        pal = ["#A3AEB9", "#7B8794", "#C9A227", "#1F9E8F", "#2F6FDE", "#1B2733", "#8A63D2", "#E8833A"]
        for i, (k, v) in enumerate(R["results"].items()):
            fig.add_trace(go.Bar(x=CLASSES, y=[v["per_class"][c]["f1"] for c in CLASSES], name=names.get(k, k),
                                 marker_color=pal[i % len(pal)]))
        fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.02], title="F1"), title="Per-class F1")
        st.plotly_chart(fig_layout(fig, 380), width="stretch")

        e1, e2 = st.columns(2)
        if "cross_domain" in R:
            with e1:
                st.markdown("### Generalisation to an unseen network")
                cd = R["cross_domain"]
                st.caption(f"Trained on the enterprise LAN, tested on an IoT network without retraining. "
                           f"Benign-traffic drift: PSI {cd['drift_psi_iot']:.2f} (in-domain {cd['drift_psi_in_domain']:.3f}).")
                st.dataframe(pd.DataFrame([{"model": names.get(k, k), "in-domain macro-F1": R["results"][k]["macro_f1"],
                                            "IoT macro-F1": v["macro_f1"], "IoT false-positive rate": v["fpr"]}
                                           for k, v in cd["results"].items()]).sort_values("IoT macro-F1", ascending=False),
                             hide_index=True, width="stretch")
                if "few_shot" in cd:
                    fs = cd["few_shot"]
                    st.markdown("**After adapting with a few minutes of local traffic**")
                    st.caption(f"Fine-tuned on the first {fs['adapt_flows']:,} labelled IoT flows, evaluated on the "
                               f"last {fs['eval_flows']:,}.")
                    st.dataframe(pd.DataFrame([{
                        "model": names.get(k, k), "IoT macro-F1 before": v["zero_shot"]["macro_f1"],
                        "after": v["few_shot"]["macro_f1"], "IoT FPR before": v["zero_shot"]["fpr"],
                        "FPR after": v["few_shot"]["fpr"], "enterprise macro-F1 after": v["enterprise_after"]["macro_f1"]}
                        for k, v in fs["results"].items()]), hide_index=True, width="stretch")
        if "adversarial" in R:
            with e2:
                st.markdown("### Robustness to evasion")
                fig = go.Figure()
                for i, (k, rows) in enumerate(R["adversarial"].items()):
                    fig.add_trace(go.Scatter(x=[r["eps"] for r in rows], y=[r["detection_rate"] for r in rows],
                                             mode="lines+markers", name=names.get(k, k), line=dict(color=pal[(i * 2 + 1) % len(pal)])))
                fig.update_layout(xaxis_title="attacker perturbation budget ε", yaxis=dict(range=[0, 1.02], title="attacks still detected"))
                st.plotly_chart(fig_layout(fig, 300), width="stretch")
        if "zero_day" in R:
            st.markdown("### Zero-day test: attack families never labelled in training")
            st.dataframe(pd.DataFrame(R["zero_day"]["results"]).rename(columns={
                "unseen_class": "unseen family", "flagged_as_attack": "flagged as attack", "flagged_uncertain": "flagged uncertain",
                "detected": "detected (either)", "benign_false_alarm": "benign false alarms",
                "most_common_prediction": "most common label"}).drop(columns=["threshold"]),
                hide_index=True, width="stretch")
        if "xai" in R:
            st.markdown("### Explanation quality")
            st.caption("Fidelity+ high = highlighted features are necessary. Fidelity- low = they are sufficient. "
                       "Stability = top-5 overlap under small input noise. Random is the sanity-check baseline.")
            x1, x2 = st.columns(2)
            cols = ["method", "fidelity_plus", "fidelity_minus", "sparsity", "stability", "runtime_ms"]
            x1.markdown("Feature attribution")
            x1.dataframe(pd.DataFrame(R["xai"]["feature_explainers"])[cols], hide_index=True, width="stretch")
            x2.markdown("Structural (neighbouring flows)")
            x2.dataframe(pd.DataFrame(R["xai"]["structural_explainers"])[[c for c in cols]], hide_index=True,
                         width="stretch")

# ------------------------------------------------------------------ how it works
with tab_how:
    st.markdown("### Pipeline")
    st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#D9DFE5", fontname="IBM Plex Sans", fontsize=11];
  edge [color="#5F6B78"];
  cap [label="Packets\\n(live NIC / pcap)"]; csv [label="Flow CSVs\\nCICFlowMeter, NetFlow"];
  fm [label="Flow meter\\n5-tuple flows"]; norm [label="Canonical schema\\n27 flow features"];
  gb [label="Temporal graph builder\\nhosts = nodes, flows = edges\\n20 s windows"];
  gnn [label="Edge-level GNN\\nE-GraphSAGE / GAT / GCN", fillcolor="#E8F0FC", color="#2F6FDE"];
  xai [label="Explainers\\nIG, SHAP, GNNExplainer,\\ncounterfactuals, evidence", fillcolor="#E8F0FC", color="#2F6FDE"];
  inc [label="Incident correlation\\nMITRE ATT&CK, firewall rules"]; drift [label="Drift monitor\\n(PSI)"];
  ui [label="Analyst console / REST API"];
  cap -> fm -> norm; csv -> norm; norm -> gb -> gnn -> inc -> ui; gnn -> xai -> ui; gb -> drift -> ui;
}""")
    st.markdown("""
**Why a graph?** A single DDoS request looks like any page visit, and a single scan probe looks like a refused
connection. What gives them away is the relationship between flows: hundreds of sources converging on one host,
one host touching hundreds of ports, several internal machines calling the same rare address every few seconds.
The GNN reads each flow together with its neighbourhood, so it sees those patterns directly.

**What the model outputs.** One of eight classes for every flow: benign, DDoS, DoS, port scan, brute force,
botnet, web attack, infiltration. Alerts are grouped into incidents and mapped to MITRE ATT&CK with a suggested
response and firewall rules (for review, never applied automatically).

**How an alert is explained.**
- *Network evidence* - plain-language facts about the window, such as fan-in to the destination or beacon regularity.
- *Feature attribution* - Integrated Gradients (fast, exact gradients) and SHAP (game-theoretic, slower).
- *GNNExplainer* - which neighbouring flows the decision relied on.
- *Counterfactuals* - the smallest change that would have made the flow benign.

Explanation quality is measured (fidelity, sparsity, stability, runtime) rather than assumed.
""")
    st.markdown("### Literature gaps addressed")
    st.dataframe(pd.DataFrame([
        ("Single-dataset evaluation", "Cross-domain test (enterprise to IoT) + loaders for CIC-IDS2017/2018, CICDDoS2019, NF-* NetFlow"),
        ("Explanation quality not measured", "Fidelity+/-, sparsity, stability and runtime for every explainer, against a random baseline"),
        ("Real-time cost unreported", "Latency per window and throughput reported for every model; live capture mode"),
        ("Adversarial robustness", "White-box PGD evasion on attacker-controllable features"),
        ("Unknown / zero-day attacks", "Leave-one-attack-out test with uncertainty flagging"),
        ("Temporal leakage", "Chronological train / validation / test split"),
        ("Concept drift", "PSI drift monitor on benign traffic in the console"),
        ("Alerts are hard to act on", "Incident correlation, MITRE ATT&CK mapping, firewall rule suggestions"),
    ], columns=["Gap in the reviewed studies", "What this project does"]), hide_index=True, width="stretch")
