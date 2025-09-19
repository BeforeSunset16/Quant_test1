# 📘 README.md

## Project Overview

This is a **quantitative trading learning scaffold**, including data
fetching, feature engineering, backtesting, and result visualization.\
The goal is to quickly run through a minimal viable quantitative
research workflow and gradually expand it.

Core pipeline: 1. **Fetch market data** (yfinance / stooq)\
2. **Generate features** (returns, rolling mean/std, z-score)\
3. **Train a baseline model** (logistic regression for direction
prediction)\
4. **Backtest** (with trading cost & slippage)\
5. **Output reports** (equity curve plot, metrics JSON)

------------------------------------------------------------------------

## Environment Setup

### 1. Clone or download the repo

``` bash
git clone <repo_url> quant-6mo
cd quant-6mo
```

### 2. Create virtual environment

Windows PowerShell:

``` powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Run Steps

1.  **Fetch raw data**

``` bash
python -m src.data.make_prices --config configs/config.yaml
```

Output: `data/raw/prices.parquet`

2.  **Generate features**

``` bash
python -m src.data.prep_features --config configs/config.yaml
```

Output: `data/clean/features.parquet`

3.  **Run backtest**

``` bash
python -m src.backtest.run_backtest --config configs/config.yaml
```

Outputs: - `reports/figures/equity_curve.png`\
- `reports/tables/metrics.json`

4.  **(Optional) View experiments via MLflow**

``` bash
mlflow ui --port 5000
```

Open http://localhost:5000

------------------------------------------------------------------------

## Config File (`configs/config.yaml`)

Example:

``` yaml
market:
  tickers: ["SPY", "QQQ"]
  start: "2015-01-01"
  end: "2025-01-01"
  interval: "1d"

features:
  windows: [5, 10, 20]
  target: "next_ret"

cv:
  n_splits: 5
  embargo_days: 5

trading:
  cost_bps: 5
  slip_bps: 5
  max_weight: 1.0

seed: 42

paths:
  raw_prices: "data/raw/prices.parquet"
  features: "data/clean/features.parquet"
  equity_png: "reports/figures/equity_curve.png"
  metrics_json: "reports/tables/metrics.json"
```

------------------------------------------------------------------------

## Project Structure

``` text
quant-6mo/
├─ configs/           # Configuration files
│  └─ config.yaml
├─ data/              # Data
│  ├─ raw/            # Raw market data
│  └─ clean/          # Feature data
├─ reports/           # Outputs
│  ├─ figures/        # Equity curve plots
│  └─ tables/         # Metrics JSON
├─ src/               # Source code
│  ├─ data/           # Data fetch & features
│  ├─ lib/            # Metrics & CV tools
│  ├─ backtest/       # Backtesting logic
│  └─ utils/          # IO helpers
└─ tests/             # Unit tests
```

------------------------------------------------------------------------

## Common Issues (especially on Windows)

1.  **`ModuleNotFoundError: No module named 'yaml'`**\
    → Install missing package: `pip install pyyaml`

2.  **`UnicodeDecodeError: 'gbk'`**\
    → Save `config.yaml` in **UTF-8** encoding and read with
    `encoding="utf-8"`

3.  **`curl: (77) error setting certificate verify locations`**

    -   Method 1: Set environment variables

        ``` powershell
        $env:SSL_CERT_FILE="C:\...\certifi\cacert.pem"
        $env:REQUESTS_CA_BUNDLE=$env:SSL_CERT_FILE
        ```

    -   Method 2: Reinstall certifi/yfinance/requests

        ``` bash
        pip install --upgrade --force-reinstall certifi yfinance requests
        ```

    -   Method 3: Move the project to a pure English path (e.g.,
        `C:\quant\Quant_test1`)

4.  **Empty data / `KeyError: 'open' not in index`**\
    → Data fetch failed (certificate/network issue). Fix certificates or
    use stooq fallback.

------------------------------------------------------------------------

## Next Steps

-   Add more features (technical indicators, industry dummies)\
-   Try other models (tree-based, SVM, XGBoost)\
-   Backtest with advanced position sizing\
-   Use DVC for data versioning and MLflow for experiment tracking
