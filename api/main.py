"""REST API for integrating XGNN-IDS with other tools (SIEM, SOAR, scripts).

    uvicorn api.main:app --port 8000         # then open http://localhost:8000/docs

Endpoints
  GET  /health                  service + model status
  GET  /models                  available models and their test scores
  POST /detect                  upload a flow CSV (CICFlowMeter / NetFlow / canonical) or pcap
  POST /detect/json             send flows as JSON records
  POST /explain                 explain one flow of the last detection
"""
from __future__ import annotations

import io
import os
import tempfile
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from xgnn_ids import __version__
from xgnn_ids.data.loaders import normalise
from xgnn_ids.inference import Detection, Detector

MODELS_DIR = os.environ.get("XGNN_MODELS_DIR", "models")
app = FastAPI(title="XGNN-IDS", version=__version__,
              description="Explainable Graph Neural Network based Network Intrusion Detection")
_last: dict[str, Detection] = {}


@lru_cache(maxsize=8)
def detector(name: str | None = None) -> Detector:
    try:
        return Detector.load(MODELS_DIR, name)
    except FileNotFoundError as exc:
        raise HTTPException(503, "no trained model found - run `python -m xgnn_ids train` first") from exc


class FlowBatch(BaseModel):
    flows: list[dict]
    model: str | None = None
    min_confidence: float = 0.5


class ExplainRequest(BaseModel):
    row: int
    model: str | None = None
    shap: bool = False


def _summary(det: Detector, res: Detection, min_conf: float, max_alerts: int = 500) -> dict:
    alerts = res.alerts[res.alerts["confidence"] >= min_conf]
    cols = ["ts", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "pred_label", "confidence", "uncertain"]
    return {
        "model": det.name,
        "flows": int(len(res.df)),
        "windows": int(len(res.graphs)),
        "alerts": int(len(alerts)),
        "by_class": alerts["pred_label"].value_counts().to_dict(),
        "incidents": det.incidents(res, min_conf),
        "alert_rows": alerts[cols].head(max_alerts).reset_index().rename(columns={"index": "row"}).to_dict("records"),
        "drift_psi": res.windows["drift_psi"].dropna().tolist() if "drift_psi" in res.windows else [],
    }


@app.get("/health")
def health():
    models = Detector.available_models(MODELS_DIR)
    return {"status": "ok" if models else "no-model", "version": __version__, "models": models}


@app.get("/models")
def models():
    try:
        return Detector.read_artifacts(MODELS_DIR)["models"]
    except FileNotFoundError as exc:
        raise HTTPException(503, "no trained model found") from exc


@app.post("/detect")
async def detect(file: UploadFile = File(...), model: str | None = None, min_confidence: float = 0.5):
    data = await file.read()
    name = (file.filename or "").lower()
    if name.endswith((".pcap", ".pcapng", ".cap")):
        from xgnn_ids.capture.flowmeter import pcap_to_flows

        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as fh:
            fh.write(data)
        try:
            df = pcap_to_flows(fh.name)
        finally:
            os.unlink(fh.name)
    else:
        try:
            df = normalise(pd.read_csv(io.BytesIO(data), low_memory=False), segment=name or "upload")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    det = detector(model)
    res = det.detect(df)
    _last[det.name] = res
    return _summary(det, res, min_confidence)


@app.post("/detect/json")
def detect_json(batch: FlowBatch):
    df = pd.DataFrame(batch.flows)
    try:
        df = normalise(df, segment="api")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    det = detector(batch.model)
    res = det.detect(df)
    _last[det.name] = res
    return _summary(det, res, batch.min_confidence)


@app.post("/explain")
def explain(req: ExplainRequest):
    det = detector(req.model)
    res = _last.get(det.name)
    if res is None:
        raise HTTPException(409, "call /detect first")
    if not 0 <= req.row < len(res.df):
        raise HTTPException(404, "row out of range")
    methods = ("ig", "gnnexplainer", "counterfactual") + (("shap",) if req.shap else ())
    exp = det.explain(res, req.row, methods)
    if "structure" in exp:
        exp["structure"] = {k: v for k, v in exp["structure"].items() if k not in {"edge_mask", "sub_edges"}}
    return exp
