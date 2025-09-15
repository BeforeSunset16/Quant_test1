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
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
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