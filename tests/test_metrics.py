import pandas as pd
from src.lib.metrics import annualized_return, max_drawdown

def test_basic_metrics():
    # 构造一个稳步上升的 100 天收益序列
    ret = pd.Series([0.001] * 100)
    assert annualized_return(ret) > 0
    assert max_drawdown(ret) == 0.0