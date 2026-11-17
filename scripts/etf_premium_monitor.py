#!/usr/bin/env python3
"""Collect mainland-listed Nasdaq-100 and S&P 500 ETF/LOF IOPV premiums.

The quote endpoint is Eastmoney via AkShare. IOPV is an intraday indicative
value, not the fund's official NAV. Missing or invalid IOPV is never replaced
with NAV or an estimate.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

TZ = ZoneInfo("Asia/Shanghai")
WATCH = {
    "159632": ("华安纳斯达克100ETF", "纳指100", "重点"),
    "159501": ("嘉实纳斯达克100ETF", "纳指100", "重点"),
    "513100": ("国泰纳斯达克100ETF", "纳指100", "重点"),
    "513390": ("博时纳斯达克100ETF", "纳指100", "重点"),
    "159696": ("纳指100 ETF", "纳指100", "覆盖"),
    "159941": ("纳指100 ETF", "纳指100", "覆盖"),
    "159659": ("纳指100 ETF", "纳指100", "覆盖"),
    "513870": ("纳指100 ETF", "纳指100", "覆盖"),
    "513300": ("纳指100 ETF", "纳指100", "覆盖"),
    "159660": ("纳指100 ETF", "纳指100", "覆盖"),
    "159513": ("纳指100 ETF", "纳指100", "覆盖"),
    "513110": ("纳指100 ETF", "纳指100", "覆盖"),
    "161130": ("纳指相关LOF", "纳指100", "覆盖"),
    "161125": ("标普500相关LOF", "标普500", "覆盖"),
    "513650": ("南方标普500ETF", "标普500", "重点"),
    "513500": ("博时标普500ETF", "标普500", "重点"),
    "159612": ("国泰标普500ETF", "标普500", "重点"),
    "159655": ("华夏标普500ETF", "标普500", "重点"),
}
OUT = Path("reports")
HISTORY = Path("data/etf_premium_history.json")


def _num_signed(value):
    """Parse signed numeric fields such as daily percentage changes."""
    try:
        if pd.isna(value):
            return None
        result = float(str(value).replace(",", "").replace("%", "").strip())
        return result if pd.notna(result) and abs(result) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _num(value):
    result = _num_signed(value)
    return result if result is not None and result > 0 else None


def _classify(premium):
    if premium is None:
        return "数据不足，不判断"
    if premium < 1:
        return "重点机会：低于1%或折价"
    if premium <= 2:
        return "候选区：溢价不高于2%"
    if premium <= 5:
        return "等待：溢价2%至5%"
    return "不买：溢价高于5%"


def _is_cn_trading_day(now):
    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar("XSHG")
        return bool(cal.is_session(pd.Timestamp(now.date())))
    except Exception:
        # Fail closed: calendar uncertainty must not create a buy alert.
        return False


def _is_cn_trading_time(now):
    # China A-share continuous auction: 09:30–11:30 and 13:00–15:00.
    # In particular, do not treat the 12:00 lunch-break run as live data.
    current = now.time()
    from datetime import time
    return (
        time(9, 30) <= current <= time(11, 30)
        or time(13, 0) <= current <= time(15, 0)
    )


def _is_cn_session(now):
    return _is_cn_trading_day(now) and _is_cn_trading_time(now)


def collect():
    now = datetime.now(TZ)
    OUT.mkdir(parents=True, exist_ok=True)
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    is_trading_day = _is_cn_trading_day(now)
    is_trading_session = is_trading_day and _is_cn_trading_time(now)
    market_state = (
        "交易日盘中/收盘快照" if is_trading_session
        else "非交易日或交易日历不可验证" if not is_trading_day
        else "非交易时段（午休或收盘后），不触发入场提醒"
    )
    result = {
        "generated_at": now.isoformat(),
        "timezone": "Asia/Shanghai",
        "source": "AkShare fund_etf_spot_em (Eastmoney)",
        "source_timestamp_note": "generated_at is the collection time, not an exchange-provided quote timestamp",
        "is_trading_session": is_trading_session,
        "market_state": market_state,
        "premium_definition": "(最新价 / IOPV实时估值 - 1) * 100",
        "rows": [],
        "errors": [],
    }
    try:
        frame = ak.fund_etf_spot_em()
        if frame is None or frame.empty:
            raise RuntimeError("AkShare returned an empty ETF spot table")
        frame["代码"] = frame["代码"].astype(str).str.zfill(6)
        by_code = frame.drop_duplicates("代码").set_index("代码")
        discovered = {}
        for _, row in frame.iterrows():
            code = str(row.get("代码", "")).zfill(6)
            name = str(row.get("名称", ""))
            lowered = name.lower()
            if any(k in name for k in ("纳指", "纳斯达克")) or "nasdaq" in lowered:
                discovered.setdefault(code, (name, "纳指100", "市场发现"))
            elif "标普500" in name or "标普 500" in name or "s&p 500" in lowered:
                discovered.setdefault(code, (name, "标普500", "市场发现"))
        codes = dict(discovered)
        for code, (name, category, priority) in WATCH.items():
            codes[code] = (name, category, priority)

        for code, (fallback_name, category, priority) in sorted(codes.items()):
            if code not in by_code.index:
                result["rows"].append({
                    "code": code, "name": fallback_name, "category": category,
                    "priority": priority, "status": "本次行情源未返回该代码",
                    "price": None, "iopv": None, "premium_pct": None,
                    "amount": None, "change_pct": None,
                    "action": "不判断，不能当作折价机会",
                })
                continue
            row = by_code.loc[code]
            price = _num(row.get("最新价"))
            iopv = _num(row.get("IOPV实时估值"))
            premium = ((price / iopv) - 1) * 100 if price and iopv else None
            amount = _num(row.get("成交额"))
            status = "IOPV有效" if premium is not None else "IOPV缺失/无效"
            if not result["is_trading_session"]:
                action = "非交易日/非交易时段/日历未验证，不触发入场提醒"
            else:
                action = _classify(premium)
            result["rows"].append({
                "code": code,
                "name": str(row.get("名称") or fallback_name),
                "category": category,
                "priority": priority,
                "price": price,
                "iopv": iopv,
                "premium_pct": round(premium, 4) if premium is not None else None,
                "amount": amount,
                "change_pct": _num_signed(row.get("涨跌幅")),
                "status": status,
                "action": action,
            })
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")

    result["rows"].sort(key=lambda x: (
        x["category"], x["priority"] != "重点",
        x["premium_pct"] is None, x["premium_pct"] if x["premium_pct"] is not None else 999
    ))
    json_path = OUT / "etf_premium_latest.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 场内纳指/标普ETF溢价监控",
        "",
        f"- 采集时间：{result['generated_at']}（Asia/Shanghai）",
        f"- 行情状态：{result['market_state']}",
        f"- 数据源：{result['source']}",
        "- 计算口径：实时成交价 / IOPV实时估值 - 1；不是官方NAV溢价。",
        "- 注意：采集时间不等于交易所行情时间；缺少IOPV时不推算、不触发机会提醒。",
        "",
        "| 代码 | 名称 | 类别 | 最新价 | IOPV | 溢价率 | 成交额 | 判断 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for item in result["rows"]:
        def fmt(v, digits=4):
            return "缺失" if v is None else f"{v:.{digits}f}"
        lines.append(
            f"| {item['code']} | {item['name']} | {item['category']} | "
            f"{fmt(item['price'])} | {fmt(item['iopv'])} | "
            f"{fmt(item['premium_pct'], 2)}% | {fmt(item['amount'], 0)} | {item['action']} |"
        )
    if result["errors"]:
        lines += ["", "## 数据错误", *[f"- {e}" for e in result["errors"]]]
    (OUT / "etf_premium_latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Store only a verified China-market session snapshot, so non-trading days
    # and failed/empty responses cannot pollute the 1/5/20-session history.
    if result["is_trading_session"] and result["rows"] and not result["errors"]:
        previous = []
        if HISTORY.exists():
            try:
                previous = json.loads(HISTORY.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                previous = []
        day = now.date().isoformat()
        snapshot = {
            "date": day,
            "collected_at": result["generated_at"],
            "premiums": {
                item["code"]: item["premium_pct"]
                for item in result["rows"] if item["premium_pct"] is not None
            },
        }
        # Each intraday run refreshes today's snapshot. The 15:00 workflow
        # run persists the final session snapshot to the repository.
        if previous and previous[-1].get("date") == day:
            previous[-1] = snapshot
        else:
            previous.append(snapshot)
        previous = previous[-30:]
        HISTORY.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "generated_at": result["generated_at"],
        "is_trading_session": result["is_trading_session"],
        "row_count": len(result["rows"]),
        "errors": result["errors"],
        "candidate_count": sum(1 for r in result["rows"] if r["premium_pct"] is not None and r["premium_pct"] <= 2 and result["is_trading_session"]),
        "priority_alert_count": sum(1 for r in result["rows"] if r["premium_pct"] is not None and r["premium_pct"] < 1 and result["is_trading_session"]),
    }, ensure_ascii=False))
    return result


if __name__ == "__main__":
    collect()
