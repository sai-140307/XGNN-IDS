"""Model, explainer and end-to-end tests (need PyTorch)."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xgnn_ids.config import CLASSES, GraphConfig, TrainConfig  # noqa: E402
from xgnn_ids.data.features import EDGE_FEATURE_NAMES  # noqa: E402
from xgnn_ids.data.synthetic import generate_traffic  # noqa: E402
from xgnn_ids.explain.core import batched_target_proba, build_context, target_proba  # noqa: E402
from xgnn_ids.explain.explainers import (edge_integrated_gradients, feature_counterfactual,  # noqa: E402
                                         gnn_explainer, integrated_gradients, permutation_shap,
                                         structural_counterfactual)
from xgnn_ids.explain.quality import feature_fidelity, sparsity, structural_fidelity, topk_jaccard  # noqa: E402
from xgnn_ids.graph.builder import build_graphs, chronological_split  # noqa: E402
from xgnn_ids.models.gnn import build_model, num_hops  # noqa: E402
from xgnn_ids.training.data import Preprocessor  # noqa: E402
from xgnn_ids.training.trainer import train_gnn  # noqa: E402

MODELS = ["mlp", "gcn", "gat", "egraphsage"]


@pytest.fixture(scope="module")
def graphs():
    df = generate_traffic("enterprise", duration_s=240, seed=2)
    return df, build_graphs(df, GraphConfig(window_s=20, max_edges=3000))


@pytest.fixture(scope="module")
def prep(graphs):
    return Preprocessor.fit(graphs[1])


@pytest.mark.parametrize("name", MODELS)
def test_forward_shapes_and_masking(name, graphs, prep):
    torch.manual_seed(0)
    g = max(graphs[1], key=lambda g: g.num_edges)
    model = build_model(name, prep.n_node_features, prep.n_edge_features, len(CLASSES)).eval()
    x, ei, ea, _ = prep.tensors(g)
    out = model(x, ei, ea)
    assert out.shape == (g.num_edges, len(CLASSES))
    assert torch.isfinite(out).all()
    masked = model(x, ei, ea, torch.zeros(g.num_edges))
    if name == "mlp":
        assert torch.allclose(out, masked)
    else:
        assert not torch.allclose(out, masked)


@pytest.mark.parametrize("name", MODELS)
def test_computation_subgraph_is_exact(name, graphs, prep):
    """Explaining on the (L+1)-hop subgraph must give exactly the full-graph prediction."""
    torch.manual_seed(1)
    g = max(graphs[1], key=lambda g: g.num_edges)
    model = build_model(name, prep.n_node_features, prep.n_edge_features, len(CLASSES), layers=2).eval()
    x, ei, ea, _ = prep.tensors(g)
    with torch.no_grad():
        full = torch.softmax(model(x, ei, ea), 1).numpy()
    rng = np.random.default_rng(0)
    for e in rng.choice(g.num_edges, 5, replace=False):
        ctx = build_context(model, prep, g, int(e), num_hops(name, 2))
        assert np.allclose(ctx.probs, full[e], atol=1e-5)
        assert ctx.m <= g.num_edges


def test_batched_proba_matches_single(graphs, prep):
    torch.manual_seed(2)
    g = graphs[1][3]
    model = build_model("egraphsage", prep.n_node_features, prep.n_edge_features, len(CLASSES)).eval()
    ctx = build_context(model, prep, g, 0, 3)
    X = np.random.default_rng(0).standard_normal((5, prep.n_edge_features)).astype(np.float32)
    batched = batched_target_proba(ctx, X, max_edges=ctx.m * 2)   # forces chunking
    for i in range(5):
        ea = ctx.edge_attr.clone()
        ea[ctx.target] = torch.from_numpy(X[i])
        assert np.allclose(batched[i], target_proba(ctx, ea), atol=1e-5)


def test_training_learns(graphs, prep):
    tr, va, _ = chronological_split(graphs[1])
    cfg = TrainConfig(epochs=6, patience=10, batch_edges=4000)
    model, info = train_gnn("egraphsage", tr, va, prep, cfg, log=lambda *_: None)
    h = info["history"]
    assert h[-1]["loss"] < h[0]["loss"]
    assert info["best_val_macro_f1"] > 0.5


def test_explainers(graphs, prep):
    torch.manual_seed(3)
    tr, va, te = chronological_split(graphs[1])
    model, _ = train_gnn("egraphsage", tr, va, prep, TrainConfig(epochs=5, batch_edges=4000), log=lambda *_: None)
    g = max(te, key=lambda g: int((g.y > 0).sum()))
    e = int(np.flatnonzero(g.y > 0)[0])
    ctx = build_context(model, prep, g, e, 3)

    ig = integrated_gradients(ctx, steps=128)
    assert ig.shape == (len(ctx.z0),) and len(ctx.z0) > len(EDGE_FEATURE_NAMES)   # flow + host context
    p0 = batched_target_proba(ctx, np.zeros((1, ig.size), np.float32))[0, ctx.cls]
    assert abs(ig.sum() - (ctx.probs[ctx.cls] - p0)) < 0.15    # IG completeness (Riemann approximation)

    eig = edge_integrated_gradients(ctx, steps=16)
    assert eig.shape == (ctx.m,) and np.isfinite(eig).all()
    assert eig[ctx.target] == np.abs(eig).max()

    em, fm = gnn_explainer(ctx, epochs=30)
    assert em.shape == (ctx.m,) and fm.shape == (ig.size,)
    assert em[ctx.target] == 1.0 and (em >= 0).all() and (em <= 1).all()

    fid = feature_fidelity(ctx, ig, 5)
    assert np.isfinite([fid["fidelity_plus"], fid["fidelity_minus"]]).all()
    assert -1 <= fid["fidelity_plus_prob"] <= 1
    sfid = structural_fidelity(ctx, em, 0.2)
    assert np.isfinite(sfid["fidelity_plus"]) and sfid["k"] >= 1
    assert 0 <= sparsity(ig) < 1
    assert topk_jaccard(ig, ig) == 1.0

    sc = prep.edge_scaler
    cf = feature_counterfactual(ctx, EDGE_FEATURE_NAMES, 0, (sc.lo, sc.hi), steps=150)
    if cf["found"]:
        assert int(np.argmax(cf["new_probs"])) == 0
        assert len(cf["changes"]) <= 4
        for i, _, new in cf["changes"]:                      # stays inside the realistic range
            assert sc.lo[i] - 1e-4 <= new <= sc.hi[i] + 1e-4 or new == pytest.approx(ctx.edge_attr[ctx.target, i].item())
    scf = structural_counterfactual(ctx, em)
    assert "found" in scf


def test_permutation_shap_efficiency():
    w = np.array([1.0, -2.0, 0.5])
    f = lambda X: X @ w  # noqa: E731
    bg = np.zeros((4, 3))
    phi = permutation_shap(f, np.array([1.0, 1.0, 2.0]), bg, n_perm=6)
    assert np.allclose(phi, [1.0, -2.0, 1.0])


# ------------------------------------------------------------------ end to end


def test_experiment_outputs(trained):
    rep = trained["report"]
    assert (trained["models"] / "artifacts.json").exists()
    assert (trained["results"] / "RESULTS.md").exists()
    art = json.loads((trained["models"] / "artifacts.json").read_text())
    assert art["default_model"] in {"egraphsage", "gat"}
    assert rep["results"]["egraphsage"]["macro_f1"] > 0.6
    assert "cross_domain" in rep and "adversarial" in rep and "xai" in rep
    methods = {r["method"] for r in rep["xai"]["feature_explainers"]}
    assert {"Integrated Gradients", "SHAP", "Random"} <= methods


def test_detector_roundtrip(trained):
    from xgnn_ids.inference import Detector

    det = Detector.load(trained["models"])
    df = generate_traffic("enterprise", duration_s=180, seed=11)
    res = det.detect(df)
    assert len(res.df) == len(df) and set(res.df["pred_label"]) <= set(CLASSES)
    assert (res.window_of >= 0).all()
    assert not res.windows.empty
    row = int(res.alerts["confidence"].idxmax())
    exp = det.explain(res, row, ("ig", "edge_ig", "gnnexplainer", "counterfactual", "shap"))
    assert exp["prediction"]["label"] != "BENIGN"
    if exp["subgraph"]["flows"] > 1:
        assert exp["structure"]["method"] == "Edge Integrated Gradients"
        assert exp["structure_gnnexplainer"]["method"] == "GNNExplainer"
    assert "Integrated Gradients" in exp["features"] and "SHAP" in exp["features"]
    assert exp["narrative"].startswith("**")
    inc = det.incidents(res)
    assert inc and all("firewall" in i for i in inc)

    empty = det.detect(df.iloc[0:0])                 # live capture before the first flow completes
    assert empty.graphs == [] and empty.alerts.empty
