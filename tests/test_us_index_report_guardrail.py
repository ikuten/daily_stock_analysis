# -*- coding: utf-8 -*-
"""Regression tests for US index report guardrails."""

from types import SimpleNamespace

from src.us_index_report_guardrail import (
    apply_us_index_report_guardrail,
    is_us_index_report_code,
    us_index_system_prompt_rules,
)


def _result(code="NDX", language="zh"):
    return SimpleNamespace(
        code=code,
        report_language=language,
        operation_advice="买入",
        decision_type="buy",
        action="buy",
        analysis_summary="缩量回调，建议买入。",
        short_term_outlook="等待回踩买入。",
        volume_analysis="缩量回调，抛压减轻。",
        fundamental_analysis="PE估值合理。",
        buy_reason="突破买入。",
        risk_warning="跌破支撑止损。",
        key_points="缩量、估值合理",
        dashboard={
            "core_conclusion": {
                "one_sentence": "等待MA10附近买入",
                "position_advice": {
                    "no_position": "空仓者买入",
                    "has_position": "持仓者加仓",
                },
            },
            "data_perspective": {
                "volume_analysis": {
                    "volume_ratio": None,
                    "volume_status": "缩量",
                    "turnover_rate": None,
                    "volume_meaning": "抛压减轻",
                }
            },
            "battle_plan": {
                "sniper_points": {
                    "ideal_buy": "27843",
                    "stop_loss": "27600",
                    "take_profit": "28500",
                },
                "position_strategy": {"suggested_position": "3成"},
                "action_checklist": ["⚠️ 量能配合", "✅ 趋势正常", "✅ PE估值合理"],
            },
            "phase_decision": {
                "immediate_action": "等待买入",
                "watch_conditions": ["跌破27600止损"],
                "next_check_time": "今日",
            },
        },
    )


def test_us_index_guardrail_removes_stock_style_trade_plan_and_unknown_volume_claims():
    result = _result()

    guarded = apply_us_index_report_guardrail(
        result,
        source_context={"quote": {"volume_ratio": None}, "fundamentals": {"pe_ttm": None}},
    )

    assert guarded is result
    assert result.decision_type == "hold"
    assert result.action == "watch"
    assert result.operation_advice == "观察"
    assert "battle_plan" not in result.dashboard
    assert "缩量回调" not in result.volume_analysis
    assert result.dashboard["data_perspective"]["volume_analysis"]["volume_status"] == "未知"
    assert "无法判断" in result.dashboard["data_perspective"]["volume_analysis"]["volume_meaning"]
    assert "无法判断" in result.fundamental_analysis
    assert "止损" not in result.dashboard["phase_decision"]["watch_conditions"][0]
    assert "PE估值合理" not in " ".join(result.dashboard["action_checklist"])


def test_us_index_guardrail_preserves_non_index_results():
    result = _result(code="AAPL")
    original_dashboard = result.dashboard

    assert apply_us_index_report_guardrail(result, source_context={}) is result
    assert result.dashboard is original_dashboard
    assert result.operation_advice == "买入"
    assert result.decision_type == "buy"


def test_us_index_guardrail_does_not_mark_pe_missing_when_source_has_numeric_pe():
    result = _result()

    apply_us_index_report_guardrail(
        result,
        source_context={"fundamentals": {"forward_pe": 24.5}},
    )

    assert result.fundamental_analysis == "PE估值合理。"
    assert not any("无法判断" in item and "PE" in item for item in result.dashboard["action_checklist"])


def test_us_index_prompt_rules_are_explicit_and_language_aware():
    zh = us_index_system_prompt_rules("zh")
    en = us_index_system_prompt_rules("en")

    assert "禁止输出" in zh
    assert "止损位" in zh
    assert "Missing means unknown" in en
    assert "price targets" in en



def test_us_index_code_detection_covers_canonical_aliases_only():
    for code in ("SPX", "^GSPC", "NDX", "^NDX", "IXIC", "VIX", "^VIX", "RUT"):
        assert is_us_index_report_code(code)
    assert not is_us_index_report_code("AAPL")
    assert not is_us_index_report_code("600519")
