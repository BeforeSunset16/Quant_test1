from __future__ import annotations
import argparse
import yaml
import pandas as pd
from pathlib import Path
from src.utils.io import save_parquet

# yfinance & 备用数据源
import yfinance as yf

def fetch_yf_multi(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
    # 注意：这里用 auto_adjust=False，保留原始 Close 和 Adj Close 两列（有的源才有）
    df = yf.download(
        tickers, start=start, end=end, interval=interval,
        auto_adjust=False, progress=False
    )

    # 兼容多票/单票的列结构
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=1, future_stack=True).rename_axis(index=["date", "ticker"]).reset_index()
    else:
        df = df.reset_index().rename(columns={"index": "date"})
        df["ticker"] = tickers[0]

    # 统一列名
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
    })

    # 如果没有 adj_close（比如 auto_adjust=True 或源不提供），就用 close 代替
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # 只保留需要的列（存在哪列就留哪列）
    keep = ["open", "high", "low", "close", "adj_close", "volume"]
    have = [c for c in keep if c in df.columns]
    df = df[["date", "ticker"] + have].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "ticker"]).sort_index()
    return df



def fetch_yf_one_by_one(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
    frames = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(start=start, end=end, interval=interval, auto_adjust=True)
            if hist.empty:
                continue
            hist = hist.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close",
                "Adj Close": "adj_close", "Volume": "volume"
            })
            keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in hist.columns]
            if not keep:
                continue
            hist["ticker"] = t
            hist = hist.reset_index().rename(columns={"Date": "date"})
            hist["date"] = pd.to_datetime(hist["date"])
            frames.append(hist[["date", "ticker"] + keep].set_index(["date", "ticker"]))
        except Exception as e:
            print(f"[warn] yfinance 单票失败 {t}: {e}")
    if frames:
        return pd.concat(frames).sort_index()
    return pd.DataFrame()


def fetch_stooq(tickers: list[str]) -> pd.DataFrame:
    """备用：pandas-datareader 的 stooq 源（只支持日频，历史较全）"""
    try:
        from pandas_datareader import data as pdr
    except Exception:
        print("[warn] 未安装 pandas-datareader，跳过 stooq 备用源。 pip install pandas-datareader")
        return pd.DataFrame()

    frames = []
    for t in tickers:
        try:
            d = pdr.DataReader(t, "stooq")
            if d.empty:
                continue
            d = d.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
            })
            # stooq 没有 adj_close（这里用 close 代替）
            d["adj_close"] = d.get("adj_close", d["close"])
            d["ticker"] = t
            d = d.reset_index().rename(columns={"Date": "date"})
            d["date"] = pd.to_datetime(d["date"])
            frames.append(d[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
                          .set_index(["date", "ticker"]))
        except Exception as e:
            print(f"[warn] stooq 单票失败 {t}: {e}")
    if frames:
        return pd.concat(frames).sort_index()
    return pd.DataFrame()


def fetch_prices(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
    # 1) 先尝试 yfinance 多票一次抓
    try:
        df = fetch_yf_multi(tickers, start, end, interval)
        if not df.empty:
            return df
        print("[warn] yfinance 多票结果为空，尝试逐票抓取 …")
    except Exception as e:
        print(f"[warn] yfinance 多票失败：{e}，尝试逐票抓取 …")

    # 2) 逐票抓
    df = fetch_yf_one_by_one(tickers, start, end, interval)
    if not df.empty:
        return df

    # 3) 备用源 stooq（日频）
    print("[warn] 使用 stooq 备用数据源（日频），不保证与 yfinance 完全一致 …")
    df = fetch_stooq(tickers)
    if not df.empty:
        # 裁剪到起止日期
        idx = df.index.get_level_values(0)
        mask = (idx >= pd.to_datetime(start)) & (idx <= pd.to_datetime(end))
        return df[mask]

    # 4) 都失败，给出明确提示
    raise RuntimeError(
        "下载行情失败：yfinance 与 stooq 都未取到数据。\n"
        "请检查网络/代理/证书（设置 SSL_CERT_FILE / REQUESTS_CA_BUNDLE 或将项目迁移到英文路径），"
        "并尝试：pip install --upgrade certifi yfinance requests pandas-datareader"
    )


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件解析为空或格式不对：{cfg_path}")

    m = cfg["market"]; p = cfg["paths"]
    df = fetch_prices(m["tickers"], m["start"], m["end"], m.get("interval", "1d"))
    # 最终只保留我们需要的列（如果有缺就按存在的来）
    keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep].sort_index()
    save_parquet(df, p["raw_prices"])
    print(f"✅ saved raw prices → {p['raw_prices']} rows={len(df):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)