.PHONY: install lint test train

MODEL ?= xgboost

install:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest

train:
	uv run python scripts/train.py --model $(MODEL)
