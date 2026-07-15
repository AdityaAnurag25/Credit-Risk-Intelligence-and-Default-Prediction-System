.PHONY: install lint format test train evaluate explain mlflow-ui serve monitor docker-build docker-run compose-up compose-down clean

MODEL ?= xgboost
MODEL_PATH ?=
IMAGE ?= credit-risk-api:latest

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

train:
	uv run python scripts/train.py --model $(MODEL)

evaluate:
	uv run python scripts/evaluate.py --model-path $(MODEL_PATH)

explain:
	uv run python scripts/explain.py --model-path $(MODEL_PATH)

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve:
	uv run uvicorn credit_risk.api.main:app --host 0.0.0.0 --port 8000

monitor:
	uv run python scripts/monitor.py

docker-build:
	docker build --build-arg APP_UID=$(shell id -u) --build-arg APP_GID=$(shell id -g) -t $(IMAGE) .

# mlruns is mounted read-write at the same absolute path on both sides: mlflow's
# local artifact store records that path as absolute in mlflow.db, and loading a
# registered model writes a small provenance file back into it.
docker-run:
	docker run --rm -p 8000:8000 \
		-v "$(CURDIR)/mlflow.db:/app/mlflow.db:ro" \
		-v "$(CURDIR)/mlruns:$(CURDIR)/mlruns" \
		-v "$(CURDIR)/outputs/models:/app/outputs/models:ro" \
		$(IMAGE)

compose-up:
	UID=$(shell id -u) GID=$(shell id -g) PROJECT_ROOT="$(CURDIR)" docker compose up --build -d

compose-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -not -path "./venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
