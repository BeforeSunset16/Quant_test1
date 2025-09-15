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