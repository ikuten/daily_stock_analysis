import importlib.util
from pathlib import Path
from unittest.mock import patch

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "etf_premium_monitor.py"
spec = importlib.util.spec_from_file_location("etf_premium_monitor", MODULE_PATH)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_premium_threshold_boundaries():
    assert monitor._classify(5.0001) == "不买：溢价高于5%"
    assert monitor._classify(5.0) == "等待：溢价2%至5%"
    assert monitor._classify(2.0001) == "等待：溢价2%至5%"
    assert monitor._classify(2.0) == "候选区：溢价不高于2%"
    assert monitor._classify(0.99) == "重点机会：低于1%或折价"
    assert monitor._classify(-0.2) == "重点机会：低于1%或折价"
    assert monitor._classify(None) == "数据不足，不判断"


def test_num_rejects_missing_and_non_positive_values():
    assert monitor._num(None) is None
    assert monitor._num("—") is None
    assert monitor._num("0") is None
    assert monitor._num("2.345") == 2.345


def test_non_trading_session_never_emits_buy_opportunity(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "OUT", tmp_path / "reports")
    monkeypatch.setattr(monitor, "HISTORY", tmp_path / "data" / "history.json")
    monkeypatch.setattr(monitor, "_is_cn_session", lambda now: False)
    frame = pd.DataFrame([{
        "代码": "513500", "名称": "博时标普500ETF", "最新价": 2.0,
        "IOPV实时估值": 2.01, "成交额": 1000000, "涨跌幅": -1.0,
    }])
    with patch.object(monitor.ak, "fund_etf_spot_em", return_value=frame):
        result = monitor.collect()
    row = next(item for item in result["rows"] if item["code"] == "513500")
    assert row["premium_pct"] is not None
    assert "不触发入场提醒" in row["action"]
    assert not (tmp_path / "data" / "history.json").exists()
