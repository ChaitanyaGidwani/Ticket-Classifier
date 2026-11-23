VENV ?= .venv
PYTHON ?= $(shell if [ -f $(VENV)/bin/python ]; then echo $(VENV)/bin/python; elif [ -f ticket-classifier/.venv/bin/python ]; then echo ticket-classifier/.venv/bin/python; else echo python; fi)
UVICORN ?= $(shell if [ -f $(VENV)/bin/uvicorn ]; then echo $(VENV)/bin/uvicorn; elif [ -f ticket-classifier/.venv/bin/uvicorn ]; then echo ticket-classifier/.venv/bin/uvicorn; else echo uvicorn; fi)
STREAMLIT ?= $(shell if [ -f $(VENV)/bin/streamlit ]; then echo $(VENV)/bin/streamlit; elif [ -f ticket-classifier/.venv/bin/streamlit ]; then echo ticket-classifier/.venv/bin/streamlit; else echo streamlit; fi)
PYTEST ?= $(shell if [ -f $(VENV)/bin/pytest ]; then echo $(VENV)/bin/pytest; elif [ -f ticket-classifier/.venv/bin/pytest ]; then echo ticket-classifier/.venv/bin/pytest; else echo pytest; fi)

.PHONY: help run-api run-ui test eval eval-heldout tune export-onnx docker-build docker-run

help:
	@echo "Support Ticket Classifier - Developer Commands"
	@echo "  make run-api      Run FastAPI REST server (port 8000)"
	@echo "  make run-ui       Run Streamlit interactive UI (port 8501)"
	@echo "  make test         Run pytest test suite"
	@echo "  make eval         Run original 28-ticket benchmark"
	@echo "  make eval-heldout Run held-out test benchmark (94 tickets)"
	@echo "  make tune         Run hyperparameter tuning on validation set"
	@echo "  make export-onnx  Export model to ONNX & quantize to INT8"
	@echo "  make docker-build Build Docker container image"
	@echo "  make docker-run   Run Docker container"

run-api:
	$(UVICORN) api:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	$(STREAMLIT) run app.py

test:
	$(PYTEST) tests/ -v

eval:
	$(PYTHON) evaluate.py

eval-heldout:
	$(PYTHON) evaluate_heldout.py --split test

tune:
	$(PYTHON) scripts/tune_thresholds.py

export-onnx:
	$(PYTHON) scripts/export_onnx.py

docker-build:
	docker build -t ticket-classifier:latest .

docker-run:
	docker run -p 8000:8000 -p 8501:8501 ticket-classifier:latest

