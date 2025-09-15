# ===== Makefile =====
PY=python
VENVDIR=.venv
PIP=$(VENVDIR)/bin/pip
PYBIN=$(VENVDIR)/bin/python
MLFLOW=$(VENVDIR)/bin/mlflow
DVC=$(VENVDIR)/bin/dvc
.PHONY: help setup clean data features backtest repro mlflow ui test fmt
help:
@echo "Targets: setup | data | features | backtest | repro | mlflow | ui | test | fmt | clean"
setup:
python -m venv $(VENVDIR)
$(PIP) install --upgrade pip
$(PIP) install -r requirements.txt
@echo "✅ setup done. Activate: source $(VENVDIR)/bin/activate"
data:
$(PYBIN) -m src.data.make_prices --config configs/config.yaml
2
features:
$(PYBIN) -m src.data.prep_features --config configs/config.yaml
backtest:
$(PYBIN) -m src.backtest.run_backtest --config configs/config.yaml
repro:
$(DVC) repro
mlflow:
$(MLFLOW) ui --port 5000 --host 0.0.0.0
ui: mlflow
test:
$(PYBIN) -m pytest -q
fmt:
$(VENVDIR)/bin/isort src tests
$(VENVDIR)/bin/black src tests
clean:
rm -rf $(VENVDIR) .pytest_cache mlruns
find . -name "*.pyc" -delete