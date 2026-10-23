# Deployment

## Streamlit Community Cloud (free, public link)

1. Sign in at <https://share.streamlit.io> with the GitHub account that owns this repository.
2. *Create app* → *Deploy a public app from GitHub* → repository `sai-140307/XGNN-IDS`, branch `main`,
   main file `app/streamlit_app.py`. Under *Advanced settings* choose Python 3.11.
3. Deploy. The first build installs CPU PyTorch from `requirements.txt` (a few minutes). The trained models in
   `models/` are committed by the *Train & publish results* workflow, so the app starts ready to use.

## Hugging Face Spaces (alternative)

Create a Space with the *Docker* SDK and push this repository to it; the `Dockerfile` starts the dashboard on
port 8501 (set `app_port: 8501` in the Space README header).

## Docker (local or a lab server)

```bash
docker compose up --build              # dashboard on :8501, REST API on :8000
docker compose --profile sensor up sensor   # live sensor on the host network (Linux, needs NET_RAW)
```

## REST API

```bash
uvicorn api.main:app --port 8000       # interactive docs at http://localhost:8000/docs
curl -F file=@flows.csv http://localhost:8000/detect
curl -X POST localhost:8000/explain -H 'content-type: application/json' -d '{"row": 123}'
```

## Retraining

`Actions` → *Train & publish results* → *Run workflow*. It trains every model, runs all evaluations, and
commits `models/` and `results/` back to `main`. Locally: `python -m xgnn_ids train` (about 15-25 minutes on a
laptop CPU, `--quick` for ~3 minutes).
