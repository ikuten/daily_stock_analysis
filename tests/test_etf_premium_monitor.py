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
    monkeypatch.setattr(monitor, "_quote_timestamp_meta", lambda row, now: {
        "timestamp": None, "age_seconds": None, "fresh": True, "reason": "mock fresh quote"
    })
    with patch.object(monitor.ak, "fund_etf_spot_em", return_value=frame):
        with patch.object(monitor.ak, "fund_lof_spot_em", return_value=pd.DataFrame()):
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
        with patch.object(monitor.ak, "fund_lof_spot_em", return_value=pd.DataFrame()):
            result = monitor.collect()
    row = next(item for item in result["rows"] if item["code"] == "513500")
    assert row["name"] == "博时标普500ETF"
    assert row["display_label"] == "博时标普500ETF（513500）"
    report = (tmp_path / "reports" / "etf_premium_latest.md").read_text(encoding="utf-8")
    assert "基金名称（代码）" in report
    assert "博时标普500ETF（513500）" in report
    assert "简称标普500" not in report


def test_quote_timestamp_must_be_current_and_fresh():
    now = pd.Timestamp("2026-11-18T09:31:00+08:00").to_pydatetime()
    fresh = monitor._quote_timestamp_meta({
        "数据日期": "2026-11-18",
        "更新时间": "2026-11-18 09:30:00+08:00",
    }, now)
    assert fresh["fresh"] is True
    assert fresh["age_seconds"] == 60.0

    stale = monitor._quote_timestamp_meta({
        "数据日期": "2026-11-17",
        "更新时间": "2026-11-17 15:00:00+08:00",
    }, now)
    assert stale["fresh"] is False
    assert "不是当前交易日" in stale["reason"]

    missing = monitor._quote_timestamp_meta({"数据日期": "2026-11-18"}, now)
    assert missing["fresh"] is False
    assert "缺少行情源更新时间" in missing["reason"]


def test_lof_is_not_treated_as_etf_iopv(monkeypatch, tmp_path):
    from datetime import datetime

    fixed_now = datetime.fromisoformat("2026-11-18T10:00:00+08:00")

    class FixedDatetime:
        @staticmethod
        def now(tz=None):
            return fixed_now

    monkeypatch.setattr(monitor, "datetime", FixedDatetime)
    monkeypatch.setattr(monitor, "OUT", tmp_path / "reports")
    monkeypatch.setattr(monitor, "HISTORY", tmp_path / "data" / "history.json")
    monkeypatch.setattr(monitor, "_is_cn_trading_day", lambda now: True)
    etf_frame = pd.DataFrame([{
        "代码": "513500", "名称": "博时标普500ETF", "最新价": 2.0,
        "IOPV实时估值": 2.01, "成交额": 1000000, "涨跌幅": 0.3,
        "数据日期": "2026-11-18", "更新时间": "2026-11-18 09:59:30+08:00",
    }])
    lof_frame = pd.DataFrame([{
        "代码": "161125", "名称": "易方达标普500LOF", "最新价": 1.8,
        "成交额": 500000,
    }])
    with patch.object(monitor.ak, "fund_etf_spot_em", return_value=etf_frame):
        with patch.object(monitor.ak, "fund_lof_spot_em", return_value=lof_frame):
            result = monitor.collect()

    lof = next(row for row in result["rows"] if row["code"] == "161125")
    assert lof["instrument_type"] == "LOF"
    assert lof["iopv"] is None
    assert lof["premium_pct"] is None
    assert "不套用ETF实时IOPV" in lof["action"]
    etf = next(row for row in result["rows"] if row["code"] == "513500")
    assert etf["quote_fresh"] is True
    assert etf["premium_pct"] is not None


def test_historical_replay_is_explicitly_official_nav_not_iopv(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor, "OUT", tmp_path / "reports")
    monkeypatch.setattr(monitor, "_is_cn_trading_day", lambda now: True)
    monkeypatch.setattr(monitor, "WATCH", {
        "513500": ("博时标普500ETF", "标普500", "重点"),
        "161125": ("易方达标普500指数证券投资基金（LOF）", "标普500", "覆盖"),
    })
    monkeypatch.setattr(monitor, "_fetch_daily_close", lambda code, day, kind: 2.0 if code == "513500" else 1.8)
    monkeypatch.setattr(monitor, "_fetch_same_day_official_nav", lambda code, day: 1.9 if code == "513500" else 1.75)
    result = monitor.replay_historical_nav("2026-11-17")
    etf = next(row for row in result["rows"] if row["code"] == "513500")
    lof = next(row for row in result["rows"] if row["code"] == "161125")
    assert etf["nav_premium_pct"] == round((2.0 / 1.9 - 1) * 100, 4)
    assert lof["instrument_type"] == "LOF"
    assert "非盘中IOPV" in etf["premium_basis"]
    report = (tmp_path / "reports" / "etf_premium_history_replay_20261117.md").read_text(encoding="utf-8")
    assert "博时标普500ETF（513500）" in report
    assert "历史复盘" in report


def test_quote_timestamp_rejects_future_quote_relative_to_observation_time():
    observed_at = pd.Timestamp("2026-11-18T09:31:00+08:00").to_pydatetime()
    future = monitor._quote_timestamp_meta({
        "数据日期": "2026-11-18",
        "更新时间": "2026-11-18 09:31:01+08:00",
    }, observed_at)
    assert future["fresh"] is False
    assert future["age_seconds"] == -1.0
    assert "晚于本次采集观察时间" in future["reason"]
