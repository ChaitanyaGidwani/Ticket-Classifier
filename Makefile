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
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run app.py

test:
	pytest tests/ -v

eval:
	python evaluate.py

eval-heldout:
	python evaluate_heldout.py --split test

tune:
	python scripts/tune_thresholds.py

export-onnx:
	python scripts/export_onnx.py

docker-build:
	docker build -t ticket-classifier:latest .

docker-run:
	docker run -p 8000:8000 -p 8501:8501 ticket-classifier:latest

