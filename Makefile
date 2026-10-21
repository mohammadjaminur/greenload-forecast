.PHONY: install data train evaluate experiment test lint smoke
PYTHON ?= python
RUN_DIR ?= results/manual

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m greenload.data.generate_synthetic_data --out data/raw_series.npz

train:
	$(PYTHON) -m greenload.train --run-dir "$(RUN_DIR)"

evaluate:
	$(PYTHON) -m greenload.evaluate --run-dir "$(RUN_DIR)"

experiment:
	$(PYTHON) -m greenload.experiments --budget-hours 4

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m ruff check src/ tests/

smoke:
	$(PYTHON) -m greenload.train --smoke-test --run-dir results/smoke
	$(PYTHON) -m greenload.evaluate --run-dir results/smoke
