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
    assert monitor._num_signed("-2.35%") == -2.35
    assert monitor._num_signed("0.00%") == 0.0


def test_trading_time_guard_excludes_lunch_and_after_close():
    from datetime import datetime

    assert monitor._is_cn_trading_time(datetime.fromisoformat("2026-11-17T09:30:00"))
    assert monitor._is_cn_trading_time(datetime.fromisoformat("2026-11-17T11:30:00"))
    assert monitor._is_cn_trading_time(datetime.fromisoformat("2026-11-17T13:00:00"))
    assert monitor._is_cn_trading_time(datetime.fromisoformat("2026-11-17T15:00:00"))
    assert not monitor._is_cn_trading_time(datetime.fromisoformat("2026-11-17T09:29:00"))
    assert not monitor._is_cn_trading_time(datetime.fromisoformat("2026-11-17T12:00:00"))
    assert not monitor._is_cn_trading_time(datetime.fromisoformat("2026-11-17T15:01:00"))


def test_non_trading_session_never_emits_buy_opportunity(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "OUT", tmp_path / "reports")
    monkeypatch.setattr(monitor, "HISTORY", tmp_path / "data" / "history.json")
    monkeypatch.setattr(monitor, "_is_cn_trading_day", lambda now: False)
    frame = pd.DataFrame([{
        "代码": "513500", "名称": "博时标普500ETF", "最新价": 2.0,
        "IOPV实时估值": 2.01, "成交额": 1000000, "涨跌幅": -1.0,
    }])
    with patch.object(monitor.ak, "fund_etf_spot_em", return_value=frame):
        result = monitor.collect()
    row = next(item for item in result["rows"] if item["code"] == "513500")
    assert row["premium_pct"] is not None
    assert row["change_pct"] == -1.0
    assert "不触发入场提醒" in row["action"]
    assert not (tmp_path / "data" / "history.json").exists()


def test_report_displays_fund_name_and_code_together(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "OUT", tmp_path / "reports")
    monkeypatch.setattr(monitor, "HISTORY", tmp_path / "data" / "history.json")
    monkeypatch.setattr(monitor, "_is_cn_trading_day", lambda now: False)
    frame = pd.DataFrame([{
        "代码": "513500", "名称": "简称标普500", "最新价": 2.0,
        "IOPV实时估值": 2.01, "成交额": 1000000, "涨跌幅": -1.0,
    }])
    with patch.object(monitor.ak, "fund_etf_spot_em", return_value=frame):
        result = monitor.collect()
    row = next(item for item in result["rows"] if item["code"] == "513500")
    assert row["name"] == "博时标普500ETF"
    assert row["display_label"] == "博时标普500ETF（513500）"
    report = (tmp_path / "reports" / "etf_premium_latest.md").read_text(encoding="utf-8")
    assert "基金名称（代码）" in report
    assert "博时标普500ETF（513500）" in report
    assert "简称标普500" not in report
