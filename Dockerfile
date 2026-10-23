# XGNN-IDS - dashboard + REST API in one CPU image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libpcap0.8 graphviz && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# train a model at build time if none was committed
RUN test -f models/artifacts.json || python -m xgnn_ids train --quick --no-zero-day --no-adversarial
EXPOSE 8501 8000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
