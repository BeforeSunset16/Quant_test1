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