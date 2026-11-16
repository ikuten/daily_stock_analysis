# -*- coding: utf-8 -*-
"""Deterministic output guardrails for US market index analysis."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

from data_provider.us_index_mapping import is_us_index_code


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _normalized_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _find_numeric_metric(payload: Any, keys: Iterable[str]) -> Optional[float]:
    """Find a numeric metric in source context, never in model-generated prose."""
    wanted = {_normalized_key(key) for key in keys}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if _normalized_key(key) in wanted and _is_number(value):
                return float(value)
        for value in payload.values():
            found = _find_numeric_metric(value, wanted)
            if found is not None:
                return found
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            found = _find_numeric_metric(value, wanted)
            if found is not None:
                return found
    return None


def us_index_system_prompt_rules(report_language: str = "zh") -> str:
    """Return high-priority instructions for US index symbols only."""
    if str(report_language or "").lower() in {"en", "ko"}:
        return """
## US INDEX REPORT SAFETY RULES (HIGHEST PRIORITY)
This symbol is a market index or volatility index, not an individual stock.
- Do not produce entry prices, secondary entries, stop-losses, take-profit targets, position sizes, allocation percentages, or short-term trading triggers.
- Do not tell the user to buy, sell, add, reduce, or liquidate an index. Use a neutral market-observation conclusion.
- Do not infer volume expansion/contraction, selling pressure, valuation, PE, or capital flows unless the corresponding numeric field is explicitly present in the supplied source data. Missing means unknown, never zero.
- If PE/valuation data is absent, state that valuation cannot be assessed. Analyst price targets are not PE evidence.
- Keep the report focused on index trend, volatility, macro context, data quality, and longer-horizon observations.
"""
    return """
## 美股指数报告强制规则（最高优先级）
本次分析对象是市场指数或波动率指数，不是可直接交易的个股。
- 禁止输出理想买点、次优买点、止损位、目标价、仓位比例、组合配置比例和短线交易触发点。
- 禁止建议用户买入、卖出、加仓、减仓或清仓指数；结论仅可描述市场状态并给出观察重点。
- 只有当输入数据明确提供对应数值时，才可判断成交量放大/缩小、抛压、估值、市盈率或资金流。缺失值代表未知，不代表零。
- PE/估值数据缺失时必须明确写“无法判断”；机构目标价不能替代 PE 证据。
- 报告聚焦指数趋势、波动率、宏观环境、数据质量和中长期观察，不输出短线交易指令。
"""


def apply_us_index_report_guardrail(result: Any, source_context: Optional[Dict[str, Any]] = None) -> Any:
    """Normalize a US index AnalysisResult before report consumers see it."""
    if result is None or not is_us_index_code(getattr(result, "code", None)):
        return result

    language = str(getattr(result, "report_language", "zh") or "zh").lower()
    english = language in {"en", "ko"}
    unknown_volume = (
        "US index volume/volume-ratio data is unavailable or not comparable; volume expansion, contraction, and selling pressure cannot be assessed."
        if english
        else "美股指数成交量/量比数据未提供或口径不可比，无法判断放量、缩量或抛压。"
    )
    unknown_pe = (
        "Index PE/valuation data is not available from the supplied source; valuation cannot be assessed."
        if english
        else "输入数据未提供可核验的指数 PE，无法判断估值水平。"
    )
    neutral_summary = (
        "Index trend, volatility, and macro context are presented for market observation only; they are not short-term ETF trading instructions."
        if english
        else "本报告仅用于观察指数趋势、波动率与宏观环境，不据此生成短线 ETF 买卖指令。"
    )
    neutral_risk = (
        "Equity indices can experience substantial volatility and drawdowns. Use a long-term asset-allocation plan and personal risk capacity."
        if english
        else "股票指数可能出现较大波动和回撤，应结合长期资产配置、投资期限与风险承受能力决策。"
    )
    neutral_advice = "Monitor" if english else ("관찰" if language == "ko" else "观察")

    dashboard = getattr(result, "dashboard", None)
    if not isinstance(dashboard, dict):
        dashboard = {}
        result.dashboard = dashboard

    # Never expose stock-style entry/exit levels or position sizing for an index.
    dashboard.pop("battle_plan", None)
    dashboard.pop("strategy_synthesis", None)

    core = dashboard.get("core_conclusion")
    if not isinstance(core, dict):
        core = {}
        dashboard["core_conclusion"] = core
    core["one_sentence"] = neutral_summary
    core["signal_type"] = "Monitor" if english else "🟡持有观望"
    core["time_sensitivity"] = "No urgency" if english else "不急"
    core["position_advice"] = {
        "no_position": (
            "Follow the established long-term allocation and staged-investment plan; do not trade on a single day's index move."
            if english
            else "结合既定长期资产配置与分批计划评估，不依据单日指数波动决定买卖。"
        ),
        "has_position": (
            "Follow the long-term plan and periodically review allocation and risk capacity; do not apply short-term index-level stops."
            if english
            else "按长期投资计划持有，定期复核资产配置与风险承受能力，不使用指数短线点位止损。"
        ),
    }

    data_perspective = dashboard.get("data_perspective")
    if not isinstance(data_perspective, dict):
        data_perspective = {}
        dashboard["data_perspective"] = data_perspective
    data_perspective["volume_analysis"] = {
        "volume_ratio": None,
        "volume_status": "unknown" if english else "未知",
        "turnover_rate": None,
        "volume_meaning": unknown_volume,
    }

    pe_value = _find_numeric_metric(
        source_context or {},
        ("pe", "pe_ttm", "trailing_pe", "forward_pe", "pe_ratio", "市盈率", "动态市盈率", "静态市盈率"),
    )
    checklist = dashboard.get("action_checklist")
    normalized_checklist = [
        item for item in (checklist or [])
        if isinstance(item, str)
        and not any(token in item.lower() for token in ("量能", "成交量", "volume", "pe估值", "市盈率", "估值"))
    ]
    normalized_checklist.append("⚠️ " + unknown_volume)
    if pe_value is None:
        normalized_checklist.append("⚠️ " + unknown_pe)
        result.fundamental_analysis = unknown_pe
    dashboard["action_checklist"] = normalized_checklist

    phase = dashboard.get("phase_decision")
    if isinstance(phase, dict):
        phase["watch_conditions"] = (
            ["关注指数中期趋势、VIX 波动率及利率环境的变化。", "仅在数据日期、数据源和指标口径明确时更新判断。"]
            if not english
            else ["Monitor the medium-term index trend, VIX, and interest-rate conditions.", "Update the assessment only when data dates, sources, and metric definitions are clear."]
        )
        phase["immediate_action"] = "观察" if not english else "Monitor"
        phase["next_check_time"] = "下一次完整交易日数据更新后" if not english else "After the next complete-session data update"

    result.operation_advice = neutral_advice
    result.decision_type = "hold"
    result.action = "watch"
    result.analysis_summary = neutral_summary
    result.short_term_outlook = ""
    result.volume_analysis = unknown_volume
    result.buy_reason = ""
    result.risk_warning = neutral_risk
    result.key_points = "指数趋势、波动率、利率环境、数据质量" if not english else "Index trend, volatility, rates, data quality"
    return result
