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