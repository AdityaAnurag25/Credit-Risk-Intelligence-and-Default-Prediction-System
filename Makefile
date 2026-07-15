.PHONY: install lint format test train evaluate explain mlflow-ui clean

MODEL ?= xgboost
MODEL_PATH ?=

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

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -not -path "./venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
