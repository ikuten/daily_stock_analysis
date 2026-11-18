#!/usr/bin/env python3
"""Collect mainland-listed Nasdaq-100 and S&P 500 ETF/LOF IOPV premiums.

The quote endpoint is Eastmoney via AkShare. IOPV is an intraday indicative
value, not the fund's official NAV. Missing or invalid IOPV is never replaced
with NAV or an estimate.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

TZ = ZoneInfo("Asia/Shanghai")
WATCH = {
    "159632": ("华安纳斯达克100ETF（QDII）", "纳指100", "重点"),
    "159501": ("嘉实纳斯达克100ETF（QDII）", "纳指100", "重点"),
    "513100": ("国泰纳斯达克100ETF（QDII）", "纳指100", "重点"),
    "513390": ("博时纳斯达克100ETF（QDII）", "纳指100", "重点"),
    "159696": ("易方达纳斯达克100ETF（QDII）", "纳指100", "覆盖"),
    "159941": ("广发纳斯达克100ETF（QDII）", "纳指100", "覆盖"),
    "159659": ("招商纳斯达克100ETF（QDII）", "纳指100", "覆盖"),
    "513870": ("富国纳斯达克100ETF（QDII）", "纳指100", "覆盖"),
    "513300": ("华夏纳斯达克100ETF（QDII）", "纳指100", "覆盖"),
    "159660": ("汇添富纳斯达克100ETF（QDII）", "纳指100", "覆盖"),
    "159513": ("大成纳斯达克100ETF（QDII）", "纳指100", "覆盖"),
    "513110": ("华泰柏瑞纳斯达克100ETF（QDII）", "纳指100", "覆盖"),
    "161130": ("易方达纳斯达克100ETF联接（QDII-LOF）", "纳指100", "覆盖"),
    "161125": ("易方达标普500指数证券投资基金（LOF）", "标普500", "覆盖"),
    "513650": ("南方标普500ETF（QDII）", "标普500", "重点"),
    "513500": ("博时标普500ETF", "标普500", "重点"),
    "159612": ("国泰标普500ETF（QDII）", "标普500", "重点"),
    "159655": ("华夏标普500ETF（QDII）", "标普500", "重点"),
}
OUT = Path("reports")
HISTORY = Path("data/etf_premium_history.json")
LOF_CODES = {"161130", "161125"}
MAX_QUOTE_AGE_SECONDS = 300


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


def _quote_timestamp_meta(row, now):
    """Validate the upstream quote timestamp; collection time alone is not enough."""
    raw_update = row.get("更新时间")
    raw_date = row.get("数据日期")
    if raw_update is None or pd.isna(raw_update) or not str(raw_update).strip():
        return {"timestamp": None, "age_seconds": None, "fresh": False, "reason": "缺少行情源更新时间"}
    try:
        update_text = str(raw_update).strip()
        if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", update_text):
            if raw_date is None or pd.isna(raw_date) or not str(raw_date).strip():
                return {"timestamp": None, "age_seconds": None, "fresh": False, "reason": "更新时间只有时分秒，缺少数据日期"}
            quote_ts = pd.Timestamp(f"{str(raw_date).strip()} {update_text}")
        else:
            quote_ts = pd.Timestamp(update_text)
        if pd.isna(quote_ts):
            raise ValueError("invalid timestamp")
        if quote_ts.tzinfo is None:
            quote_ts = quote_ts.tz_localize(TZ)
        else:
            quote_ts = quote_ts.tz_convert(TZ)
        quote_date = pd.Timestamp(raw_date).date() if raw_date is not None and not pd.isna(raw_date) else quote_ts.date()
        now_ts = pd.Timestamp(now).tz_convert(TZ)
        age_seconds = (now_ts - quote_ts).total_seconds()
        if quote_date != now.date():
            reason = f"行情数据日期为{quote_date.isoformat()}，不是当前交易日"
            fresh = False
        elif age_seconds < -30:
            reason = "行情源时间晚于当前时间，疑似时钟异常"
            fresh = False
        elif age_seconds > MAX_QUOTE_AGE_SECONDS:
            reason = f"行情已过期{int(age_seconds)}秒，超过{MAX_QUOTE_AGE_SECONDS}秒阈值"
            fresh = False
        else:
            reason = "行情时间戳有效"
            fresh = True
        return {"timestamp": quote_ts.isoformat(), "age_seconds": round(age_seconds, 1), "fresh": fresh, "reason": reason}
    except Exception as exc:
        return {"timestamp": None, "age_seconds": None, "fresh": False,
                "reason": f"无法解析行情源更新时间：{type(exc).__name__}: {exc}"}


def _fetch_daily_close(code, target_day, instrument_type):
    day = target_day.strftime("%Y%m%d")
    if instrument_type == "LOF":
        frame = ak.fund_lof_hist_em(symbol=code, period="daily", start_date=day, end_date=day, adjust="")
    else:
        frame = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=day, end_date=day, adjust="")
    if frame is None or frame.empty:
        return None
    date_col = "日期" if "日期" in frame.columns else "date" if "date" in frame.columns else None
    close_col = "收盘" if "收盘" in frame.columns else "close" if "close" in frame.columns else None
    if not date_col or not close_col:
        return None
    matched = frame[pd.to_datetime(frame[date_col], errors="coerce").dt.date == target_day]
    if matched.empty:
        return None
    return _num(matched.iloc[-1][close_col])


def _fetch_same_day_official_nav(code, target_day):
    day = target_day.strftime("%Y%m%d")
    frame = None
    try:
        frame = ak.fund_etf_fund_info_em(fund=code, start_date=day, end_date=day)
    except Exception:
        frame = None
    if frame is None or frame.empty:
        try:
            frame = ak.fund_em_open_fund_info(fund=code, indicator="单位净值走势")
        except Exception:
            return None
    if frame is None or frame.empty or "净值日期" not in frame.columns or "单位净值" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["净值日期"], errors="coerce").dt.date
    matched = frame[dates == target_day]
    if matched.empty:
        return None
    return _num(matched.iloc[-1]["单位净值"])


def replay_historical_nav(target_date):
    """Ex-post close-vs-official-NAV review; never labels NAV premium as intraday IOPV."""
    target_day = datetime.strptime(target_date, "%Y-%m-%d").date()
    calendar_probe = datetime.combine(target_day, time(15, 0), tzinfo=TZ)
    if not _is_cn_trading_day(calendar_probe):
        raise ValueError(f"{target_date}不是可验证的A股交易日")
    rows, errors = [], []
    for code, (name, category, priority) in WATCH.items():
        instrument_type = "LOF" if code in LOF_CODES else "ETF"
        close = nav = premium = None
        status = []
        try:
            close = _fetch_daily_close(code, target_day, instrument_type)
            if close is None:
                status.append("缺少该日收盘价")
        except Exception as exc:
            errors.append(f"{code}历史行情：{type(exc).__name__}: {exc}")
            status.append("历史行情获取失败")
        try:
            nav = _fetch_same_day_official_nav(code, target_day)
            if nav is None:
                status.append("缺少同日官方单位净值")
        except Exception as exc:
            errors.append(f"{code}官方净值：{type(exc).__name__}: {exc}")
            status.append("官方净值获取失败")
        if close is not None and nav is not None:
            premium = round((close / nav - 1) * 100, 4)
            status.append("同日官方NAV可核验")
        rows.append({
            "code": code, "name": name, "display_label": f"{name}（{code}）",
            "category": category, "instrument_type": instrument_type,
            "close": close, "official_nav": nav,
            "official_nav_date": target_date if nav is not None else None,
            "nav_premium_pct": premium,
            "premium_basis": "收盘价/同日官方单位净值（事后复盘口径，非盘中IOPV溢价）",
            "status": "；".join(status) if status else "数据完整",
            "action": "历史复盘，不生成实时入场提醒",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"etf_premium_history_replay_{target_day:%Y%m%d}.json"
    md_path = OUT / f"etf_premium_history_replay_{target_day:%Y%m%d}.md"
    payload = {
        "as_of_date": target_date, "mode": "historical_close_vs_official_nav",
        "source": "AkShare/Eastmoney daily fund quotes and official NAV history",
        "note": "Ex-post daily close vs same-day official NAV; not intraday IOPV and not a real-time trading alert.",
        "rows": rows, "errors": errors,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# 场内纳指/标普基金历史溢价复盘（{target_date}）", "",
        "**口径说明：收盘价与同日官方单位净值比较，是事后复盘口径，不是盘中实时IOPV溢价，不产生实时入场提醒。**", "",
        "| 基金名称（代码） | 产品类型 | 收盘价 | 同日官方NAV | NAV日期 | NAV溢价 | 数据状态 |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for item in rows:
        close_text = "缺失" if item["close"] is None else f"{item['close']:.4f}"
        nav_text = "缺失" if item["official_nav"] is None else f"{item['official_nav']:.4f}"
        premium_text = "缺失" if item["nav_premium_pct"] is None else f"{item['nav_premium_pct']:.2f}%"
        lines.append(f"| {item['display_label']} | {item['instrument_type']} | {close_text} | {nav_text} | {item['official_nav_date'] or '缺失'} | {premium_text} | {item['status']} |")
    if errors:
        lines += ["", "## 数据错误", *[f"- {error}" for error in errors]]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"mode": payload["mode"], "as_of_date": target_date, "rows": len(rows), "valid_nav_rows": sum(row["nav_premium_pct"] is not None for row in rows), "errors": errors}, ensure_ascii=False))
    return payload


def collect():
    now = datetime.now(TZ)
    OUT.mkdir(parents=True, exist_ok=True)
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    is_trading_day = _is_cn_trading_day(now)
    is_trading_session = _is_cn_session(now)
    market_state = (
        "交易日盘中/收盘快照" if is_trading_session
        else "非交易日或交易日历不可验证" if not is_trading_day
        else "非交易时段（午休或收盘后），不触发入场提醒"
    )
    result = {
        "generated_at": now.isoformat(),
        "timezone": "Asia/Shanghai",
        "source": "AkShare/Eastmoney fund_etf_spot_em (ETF) and fund_lof_spot_em (LOF)",
        "source_timestamp_note": "ETF opportunities require current upstream 数据日期/更新时间 and pass a maximum-age check; collection time is not a substitute.",
        "max_quote_age_seconds": MAX_QUOTE_AGE_SECONDS,
        "is_trading_session": is_trading_session,
        "market_state": market_state,
        "premium_definition": "(最新价 / IOPV实时估值 - 1) * 100",
        "rows": [],
        "errors": [],
        "warnings": [],
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
            if code not in LOF_CODES:
                codes[code] = (name, category, priority)

        for code, (fallback_name, category, priority) in sorted(codes.items()):
            if code not in by_code.index:
                result["rows"].append({
                    "code": code, "name": fallback_name, "category": category,
                    "priority": priority, "status": "本次行情源未返回该代码",
                    "price": None, "iopv": None, "premium_pct": None,
                    "amount": None, "change_pct": None,
                    "quote_timestamp": None, "quote_age_seconds": None,
                    "quote_fresh": False, "quote_timestamp_reason": "行情源未返回该代码",
                    "instrument_type": "ETF",
                    "action": "不判断，不能当作折价机会",
                })
                continue
            row = by_code.loc[code]
            quote_meta = _quote_timestamp_meta(row, now)
            price = _num(row.get("最新价"))
            iopv = _num(row.get("IOPV实时估值"))
            premium = ((price / iopv) - 1) * 100 if price and iopv and quote_meta["fresh"] else None
            amount = _num(row.get("成交额"))
            if not quote_meta["fresh"]:
                status = "行情时间戳无效/过期：" + quote_meta["reason"]
            elif premium is not None:
                status = "IOPV有效，行情时间戳通过新鲜度校验"
            else:
                status = "IOPV缺失/无效"
            if not result["is_trading_session"]:
                action = "非交易日/非交易时段/日历未验证，不触发入场提醒"
            elif not quote_meta["fresh"]:
                action = "行情源时间戳未通过校验，不触发入场提醒"
            else:
                action = _classify(premium)
            result["rows"].append({
                "code": code,
                "name": fallback_name if code in WATCH else str(row.get("名称") or fallback_name),
                "category": category,
                "priority": priority,
                "instrument_type": "ETF",
                "price": price,
                "iopv": iopv,
                "premium_pct": round(premium, 4) if premium is not None else None,
                "amount": amount,
                "change_pct": _num_signed(row.get("涨跌幅")),
                "quote_timestamp": quote_meta["timestamp"],
                "quote_age_seconds": quote_meta["age_seconds"],
                "quote_fresh": quote_meta["fresh"],
                "quote_timestamp_reason": quote_meta["reason"],
                "status": status,
                "action": action,
            })
    except Exception as exc:
        result["errors"].append(f"ETF行情源 {type(exc).__name__}: {exc}")

    # LOF has its own quote endpoint; never force it through ETF IOPV arithmetic.
    try:
        lof_frame = ak.fund_lof_spot_em()
        if lof_frame is not None and not lof_frame.empty:
            lof_frame["代码"] = lof_frame["代码"].astype(str).str.zfill(6)
            lof_by_code = lof_frame.drop_duplicates("代码").set_index("代码")
        else:
            lof_by_code = pd.DataFrame()
        for code in sorted(LOF_CODES):
            fallback_name, category, priority = WATCH[code]
            if code in lof_by_code.index:
                row = lof_by_code.loc[code]
                lof_price = _num(row.get("最新价"))
                lof_amount = _num(row.get("成交额"))
                lof_status = "LOF独立行情已获取；该接口不提供可验证ETF IOPV"
            else:
                lof_price = lof_amount = None
                lof_status = "LOF行情源未返回该代码"
            result["rows"].append({
                "code": code, "name": fallback_name, "category": category,
                "priority": priority, "instrument_type": "LOF",
                "price": lof_price, "iopv": None, "premium_pct": None,
                "amount": lof_amount, "change_pct": None,
                "quote_timestamp": None, "quote_age_seconds": None,
                "quote_fresh": False,
                "quote_timestamp_reason": "LOF现货接口未提供与ETF相同的可验证IOPV时间戳字段",
                "status": lof_status,
                "action": "独立观察：不套用ETF实时IOPV溢价阈值；需另核同日官方NAV",
            })
    except Exception as exc:
        result["warnings"].append(f"LOF行情源 {type(exc).__name__}: {exc}")
        for code in sorted(LOF_CODES):
            name, category, priority = WATCH[code]
            result["rows"].append({
                "code": code, "name": name, "category": category,
                "priority": priority, "instrument_type": "LOF",
                "price": None, "iopv": None, "premium_pct": None,
                "amount": None, "change_pct": None,
                "quote_timestamp": None, "quote_age_seconds": None,
                "quote_fresh": False, "quote_timestamp_reason": "LOF行情接口调用失败",
                "status": "LOF行情获取失败",
                "action": "不判断，不触发入场提醒",
            })

    # Every consumer gets one unambiguous human-readable identifier.
    for item in result["rows"]:
        item["display_label"] = f"{item['name']}（{item['code']}）"

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
        "| 基金名称（代码） | 产品类型/指数 | 最新价 | IOPV | 溢价率 | 成交额 | 行情时间 | 数据状态 | 判断 |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for item in result["rows"]:
        def fmt(v, digits=4):
            return "缺失" if v is None else f"{v:.{digits}f}"
        lines.append(
            f"| {item['display_label']} | {item.get('instrument_type', 'ETF')}/{item['category']} | "
            f"{fmt(item['price'])} | {fmt(item['iopv'])} | "
            f"{fmt(item['premium_pct'], 2)}% | {fmt(item['amount'], 0)} | "
            f"{item.get('quote_timestamp') or '未提供'} | {item['status']} | {item['action']} |"
        )
    if result["errors"]:
        lines += ["", "## 数据错误", *[f"- {e}" for e in result["errors"]]]
    if result["warnings"]:
        lines += ["", "## 数据警告", *[f"- {warning}" for warning in result["warnings"]]]
    (OUT / "etf_premium_latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Store only a verified China-market session snapshot, so non-trading days
    # and failed/empty responses cannot pollute the 1/5/20-session history.
    if result["is_trading_session"] and any(item["instrument_type"] == "ETF" and item["premium_pct"] is not None for item in result["rows"]) and not result["errors"]:
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
        "candidate_count": sum(1 for r in result["rows"] if r["instrument_type"] == "ETF" and r["premium_pct"] is not None and r["premium_pct"] <= 2 and result["is_trading_session"]),
        "priority_alert_count": sum(1 for r in result["rows"] if r["instrument_type"] == "ETF" and r["premium_pct"] is not None and r["premium_pct"] < 1 and result["is_trading_session"]),
    }, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="纳指/标普场内基金溢价监控")
    parser.add_argument("--as-of-date", help="历史复盘日期 YYYY-MM-DD；按收盘价/同日官方NAV回溯，不代表盘中IOPV")
    args = parser.parse_args()
    if args.as_of_date:
        replay_historical_nav(args.as_of_date)
    else:
        collect()
