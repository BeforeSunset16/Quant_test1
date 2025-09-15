# ✅ 前 2 周待办清单（可勾选）

> 节奏建议：每周 5 小时左右（可拆成 3×90min）。完成度以“脚手架可跑通 + 有最小产出”为准。

## Week 1（数据清洗 & 指标基线）

* [ ] 建环境：创建虚拟环境并 `pip install -r requirements.txt`
* [ ] 配置 `configs/config.yaml`（市场、日期、参数）
* [ ] `src/data/make_prices.py`：拉取行情（默认 yfinance），保存 `data/raw/prices.parquet`
* [ ] `src/lib/metrics.py`：实现年化收益、波动率、Sharpe/Sortino、最大回撤
* [ ] `src/data/prep_features.py`：对数收益、滚动均值/方差、z-score（先最小特征）
* [ ] 运行 `make data features`，产出 `data/clean/features.parquet`
* [ ] 写 `README.md` 的“数据口径&缺失处理”小节

## Week 2（时序CV & 最小回测可复现）

* [ ] `src/lib/ts_cv.py`：实现 **Purged KFold + Embargo** 的时序交叉验证
* [ ] `src/backtest/run_backtest.py`：LR 基线（方向/超额收益），含交易成本/滑点
* [ ] 打通 **MLflow**（记录参数/指标/工件）与 **DVC**（数据快照）
* [ ] 跑 `dvc repro` 完成 `fetch_data → features → backtest` 全链条
* [ ] 生成首份报告：`reports/tables/metrics.json` 与 `reports/figures/equity_curve.png`
* [ ] 写问题清单 & 下一步需求（如：行业中性、更多特征、稳健性评估）

---

# 📁 项目目录骨架

```text
quant-6mo/
├─ Makefile
├─ MLproject
├─ dvc.yaml
├─ .dvcignore
├─ requirements.txt
├─ .gitignore
├─ README.md
├─ configs/
│  └─ config.yaml
├─ data/
│  ├─ raw/.gitkeep
│  ├─ clean/.gitkeep
│  └─ interim/.gitkeep
├─ notebooks/
│  ├─ 01_cleaning.ipynb   # 可空白占位，用于探索
│  └─ 02_ts_cv_demo.ipynb
├─ reports/
│  ├─ figures/.gitkeep
│  └─ tables/.gitkeep
├─ src/
│  ├─ __init__.py
│  ├─ utils/
│  │  ├─ __init__.py
│  │  └─ io.py
│  ├─ lib/
│  │  ├─ __init__.py
│  │  ├─ metrics.py
│  │  └─ ts_cv.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ make_prices.py
│  │  └─ prep_features.py
│  ├─ backtest/
│  │  ├─ __init__.py
│  │  ├─ run_backtest.py
│  │  └─ reporting.py
│  └─ service/
│     ├─ __init__.py
│     └─ predictor.py
└─ tests/
   ├─ test_metrics.py
   └─ test_ts_cv.py
```

---

# 🧰 Makefile（可直接用）

```makefile
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
```

---

# 📦 requirements.txt（精简可跑）

```txt
pandas>=2.0
numpy>=1.26
scikit-learn>=1.3
statsmodels>=0.14
yfinance>=0.2
pyarrow>=14
mlflow>=2.12
dvc>=3.0
pyyaml>=6.0
matplotlib>=3.8
plotly>=5.20
numba>=0.59
pytest>=8.0
python-dotenv>=1.0
```

---

# 🔧 MLproject（最小可复现入口）

```yaml
name: quant-6mo

entry_points:
  backtest:
    parameters:
      config: {type: string, default: configs/config.yaml}
    command: "python -m src.backtest.run_backtest --config {config}"
  fetch:
    parameters:
      config: {type: string, default: configs/config.yaml}
    command: "python -m src.data.make_prices --config {config}"
```

> 注：MLflow 不强制需要该文件，但保留可直接 `mlflow run . -e backtest`。

---

# 🗃 dvc.yaml（数据→特征→回测 三阶段）

```yaml
stages:
  fetch_data:
    cmd: python -m src.data.make_prices --config configs/config.yaml
    deps:
      - src/data/make_prices.py
      - configs/config.yaml
    outs:
      - data/raw/prices.parquet

  features:
    cmd: python -m src.data.prep_features --config configs/config.yaml
    deps:
      - src/data/prep_features.py
      - data/raw/prices.parquet
      - configs/config.yaml
    outs:
      - data/clean/features.parquet

  backtest:
    cmd: python -m src.backtest.run_backtest --config configs/config.yaml
    deps:
      - src/backtest/run_backtest.py
      - src/lib/metrics.py
      - src/lib/ts_cv.py
      - data/clean/features.parquet
      - configs/config.yaml
    outs:
      - reports/tables/metrics.json
      - reports/figures/equity_curve.png
```

`.dvcignore`

```txt
mlruns/
.venv/
notebooks/
reports/figures/*.png
```

`.gitignore`

```txt
.venv/
__pycache__/
*.pyc
mlruns/
.DS_Store
.env
```

---

# ⚙️ configs/config.yaml（默认可跑）

```yaml
market:
  tickers: ["SPY", "QQQ"]
  start: "2015-01-01"
  end: "2025-01-01"
  interval: "1d"

features:
  windows: [5, 10, 20]
  target: "next_ret"   # 也可 "direction"

cv:
  n_splits: 5
  embargo_days: 5

trading:
  cost_bps: 5          # 单边 0.05%
  slip_bps: 5
  max_weight: 1.0

seed: 42
paths:
  raw_prices: "data/raw/prices.parquet"
  features: "data/clean/features.parquet"
  equity_png: "reports/figures/equity_curve.png"
  metrics_json: "reports/tables/metrics.json"
```

---

# 🧩 src/utils/io.py

```python
from __future__ import annotations
import pandas as pd
from pathlib import Path


def ensure_parent(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    ensure_parent(path)
    df.to_parquet(path, index=True)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
```

---

# 📈 src/lib/metrics.py（基础指标）

```python
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    # 日度简单收益序列 → 年化收益
    ret = (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1
    return float(ret)


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    ex = returns - rf / TRADING_DAYS
    vol = returns.std(ddof=0)
    return float(np.sqrt(TRADING_DAYS) * (ex.mean() / (vol + 1e-12)))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    ex = returns - rf / TRADING_DAYS
    downside = ex[ex < 0].std(ddof=0)
    return float(np.sqrt(TRADING_DAYS) * (ex.mean() / (downside + 1e-12)))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())
```

---

# ⏱ src/lib/ts\_cv.py（Purged KFold + Embargo）

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterator, Tuple

class PurgedKFold:
    """Time-series split with purge and embargo.

    Splits by *unique dates* in chronological order. For each fold, removes
    an embargo window around the test block from the training set.
    """

    def __init__(self, n_splits: int = 5, embargo_days: int = 0):
        assert n_splits >= 2
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        # Expect DatetimeIndex; if MultiIndex, use level 0.
        if isinstance(X.index, pd.MultiIndex):
            dates = X.index.get_level_values(0).normalize()
        else:
            dates = pd.to_datetime(X.index).normalize()
        uniq = dates.unique().sort_values()
        folds = np.array_split(uniq, self.n_splits)

        # Build map date -> row indices
        by_date = {}
        for d in uniq:
            by_date[d] = np.where(dates == d)[0]

        for i in range(self.n_splits):
            test_dates = pd.DatetimeIndex(folds[i])
            test_idx = np.concatenate([by_date[d] for d in test_dates])

            # Embargo range
            if self.embargo_days > 0:
                left = test_dates.min() - pd.Timedelta(days=self.embargo_days)
                right = test_dates.max() + pd.Timedelta(days=self.embargo_days)
                mask = (dates < left) | (dates > right)
            else:
                mask = ~dates.isin(test_dates)

            train_idx = np.where(mask)[0]
            yield train_idx, test_idx
```

---

# 🧪 tests/test\_metrics.py（最小单测）

```python
import pandas as pd
from src.lib.metrics import annualized_return, max_drawdown

def test_basic_metrics():
    # 构造一个稳步上升的 100 天收益序列
    ret = pd.Series([0.001] * 100)
    assert annualized_return(ret) > 0
    assert max_drawdown(ret) == 0.0
```

# 🧪 tests/test\_ts\_cv.py

```python
import pandas as pd
from src.lib.ts_cv import PurgedKFold

def test_purged_kfold_basic():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    X = pd.DataFrame(index=idx, data={"x": range(100)})
    cv = PurgedKFold(n_splits=5, embargo_days=3)
    splits = list(cv.split(X))
    assert len(splits) == 5
    # 确认 train/test 无重叠
    for tr, te in splits:
        assert set(tr).isdisjoint(set(te))
```

---

# 🧮 src/data/make\_prices.py（拉取价格并存 Parquet）

```python
from __future__ import annotations
import argparse
import yaml
import yfinance as yf
import pandas as pd
from src.utils.io import save_parquet


def fetch_prices(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    # yfinance 多标的列是 MultiIndex: (field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=1).rename_axis(index=["date", "ticker"]).reset_index()
    else:
        df["ticker"] = tickers[0]
        df = df.reset_index().rename(columns={"index": "date"})
    df = df.rename(columns={"Adj Close": "adj_close", "Close": "close"})
    df = df.set_index(["date", "ticker"]).sort_index()
    return df[["open", "high", "low", "close", "adj_close", "volume"]]


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    m = cfg["market"]
    out = cfg["paths"]["raw_prices"]
    df = fetch_prices(m["tickers"], m["start"], m["end"], m.get("interval", "1d"))
    save_parquet(df, out)
    print(f"✅ saved raw prices → {out} rows={len(df):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
```

---

# 🧪 src/data/prep\_features.py（最小特征 & 目标）

```python
from __future__ import annotations
import argparse
import yaml
import numpy as np
import pandas as pd
from src.utils.io import load_parquet, save_parquet


def add_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df.groupby(level=1)["adj_close"].pct_change().fillna(0.0)
    for w in windows:
        df[f"ret_mean_{w}"] = df.groupby(level=1)["ret"].rolling(w).mean().reset_index(level=0, drop=True)
        df[f"ret_std_{w}"] = df.groupby(level=1)["ret"].rolling(w).std().reset_index(level=0, drop=True)
        df[f"z_{w}"] = df[f"ret_mean_{w}"] / (df[f"ret_std_{w}"] + 1e-12)
    # 目标：下一期收益 & 方向
    df["next_ret"] = df.groupby(level=1)["ret"].shift(-1)
    df["direction"] = (df["next_ret"] > 0).astype(int)
    return df.dropna()


def main(cfg_path: str):
    import warnings
    warnings.filterwarnings("ignore")
    cfg = yaml.safe_load(open(cfg_path))
    df = load_parquet(cfg["paths"]["raw_prices"])  # MultiIndex(date, ticker)
    feat = add_features(df, cfg["features"]["windows"])
    save_parquet(feat, cfg["paths"]["features"])
    print(f"✅ saved features → {cfg['paths']['features']} rows={len(feat):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
```

---

# 🔁 src/backtest/run\_backtest.py（最小回测 + MLflow）

```python
from __future__ import annotations
import argparse, json, yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
from src.utils.io import load_parquet
from src.lib.ts_cv import PurgedKFold
from src.lib.metrics import annualized_return, annualized_vol, sharpe, sortino, max_drawdown


def equity_from_positions(returns: pd.Series, pos: pd.Series, cost_bps: float, slip_bps: float) -> pd.Series:
    # 成本按换手计
    pos = pos.reindex(returns.index).fillna(0.0)
    churn = pos.diff().abs().fillna(0.0)
    tc = (cost_bps + slip_bps) / 1e4
    strat_ret = pos.shift(1).fillna(0.0) * returns - churn * tc
    return strat_ret


def run(cfg: dict):
    feat_path = cfg["paths"]["features"]
    feat = load_parquet(feat_path).reset_index()
    # 只用价格派生特征
    feature_cols = [c for c in feat.columns if c.startswith("ret_mean_") or c.startswith("ret_std_") or c.startswith("z_")]
    target = cfg["features"]["target"]

    # 按日期聚合到日频（多 ticker 简化为等权）
    feat["date"] = pd.to_datetime(feat["date"])
    daily = feat.groupby("date").agg({**{c: "mean" for c in feature_cols}, **{target: "mean"}}).sort_index()

    # 交叉验证
    X = daily[feature_cols]
    y = (daily[target] > 0).astype(int)  # 方向分类

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, n_jobs=None))
    ])

    cv = PurgedKFold(n_splits=cfg["cv"]["n_splits"], embargo_days=cfg["cv"]["embargo_days"])

    preds = pd.Series(index=daily.index, dtype=float)
    for tr_idx, te_idx in cv.split(X):
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = pipe.predict_proba(X.iloc[te_idx])[:, 1]
        preds.iloc[te_idx] = p

    # 生成仓位：概率>0.5 做多，否则空仓（最小策略）
    pos = (preds > 0.5).astype(float).clip(0, cfg["trading"]["max_weight"])  # [0,1]

    # 策略收益（用日均 next_ret 作为近似；严格应逐票聚合）
    rets = daily[cfg["features"]["target"]].rename("ret")
    strat_ret = equity_from_positions(rets, pos, cfg["trading"]["cost_bps"], cfg["trading"]["slip_bps"]).dropna()

    metrics = {
        "ann_ret": annualized_return(strat_ret),
        "ann_vol": annualized_vol(strat_ret),
        "sharpe": sharpe(strat_ret),
        "sortino": sortino(strat_ret),
        "max_dd": max_drawdown(strat_ret),
        "n_days": int(len(strat_ret)),
    }

    # 保存图表
    eq = (1 + strat_ret).cumprod()
    out_png = Path(cfg["paths"]["equity_png"])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    eq.plot()
    plt.title("Equity Curve (baseline)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    # 保存指标
    out_json = Path(cfg["paths"]["metrics_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))

    return metrics


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    mlflow.set_experiment("quant-6mo-baseline")
    with mlflow.start_run():
        mlflow.log_params({
            "tickers": ",".join(cfg["market"]["tickers"]),
            "windows": ",".join(map(str, cfg["features"]["windows"])),
            "n_splits": cfg["cv"]["n_splits"],
            "embargo_days": cfg["cv"]["embargo_days"],
            "cost_bps": cfg["trading"]["cost_bps"],
            "slip_bps": cfg["trading"]["slip_bps"],
        })
        metrics = run(cfg)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cfg["paths"]["equity_png"])
        mlflow.log_artifact(cfg["paths"]["metrics_json"])
    print("✅ backtest done:", metrics)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
```

---

# 📊 src/backtest/reporting.py（可选：扩展图表）

```python
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

# 预留：叠加回撤、分 Regime 表现等
```

---

# 🧪 src/service/predictor.py（占位，Week10 用）

```python
class Predictor:
    def __init__(self, model=None):
        self.model = model
    def predict(self, X):
        return self.model.predict_proba(X)[:,1]
```

---

# 🗒 README.md（核心使用说明）

````markdown
# quant-6mo

## Quickstart
```bash
make setup
make data
make features
make backtest
````

## DVC Pipeline

```bash
dvc init
dvc repro
```

## MLflow UI

```bash
make mlflow  # http://localhost:5000
```

## 结构说明

* data/: 原始与清洗后数据（Parquet）
* configs/config.yaml: 市场与回测参数
* src/: 数据处理、特征、回测、评估
* reports/: 图表与指标输出

```
```
