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