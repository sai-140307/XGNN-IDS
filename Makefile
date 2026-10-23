.PHONY: install train quick test app api docker
install:
	pip install -r requirements.txt pytest httpx ruff
train:
	python -m xgnn_ids train
quick:
	python -m xgnn_ids train --quick
test:
	pytest -q
app:
	streamlit run app/streamlit_app.py
api:
	uvicorn api.main:app --reload --port 8000
docker:
	docker compose up --build
