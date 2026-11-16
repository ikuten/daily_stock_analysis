# -*- coding: utf-8 -*-
"""Regression tests for the US market recap output boundary."""

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.market_analyzer import MarketAnalyzer, MarketIndex, MarketOverview
from src.core.market_strategy import get_market_strategy_blueprint


def test_us_market_recap_replaces_trading_plan_and_unattributed_news():
    analyzer = object.__new__(MarketAnalyzer)
    analyzer.region = "us"
    analyzer._get_review_language = lambda: "zh"

    report = """## 2026-11-16 大盘复盘

### 一、盘面总览
10年期美债收益率升至5.1%，导致科技股下跌。
VIX 20.12，波动率处于中位。

### 三、消息催化
AI 股因某机构报告下跌。

### 六、策略判断
建议仓位三成，跌破 27000 止损，目标价 29000。
"""
    news = [
        {
            "title": "Federal Reserve releases meeting minutes",
            "snippet": "The minutes discuss the policy outlook.",
            "source": "Reuters",
            "published_date": "2026-11-15",
            "url": "https://example.com/fed-minutes",
        }
    ]

    output = analyzer._apply_us_market_review_guardrail(
        report,
        MarketOverview(
            date="2026-11-16",
            indices=[MarketIndex(code="VIX", name="VIX恐慌指数", current=20.12)],
        ),
        news,
    )

    assert "行情时点" in output
    assert "Reuters" in output
    assert "2026-11-15" in output
    assert "https://example.com/fed-minutes" in output
    assert "不代表这些事件已被证实为指数涨跌原因" in output
    assert "建议仓位三成" not in output
    assert "27000 止损" not in output
    assert "29000" not in output
    assert "5.1%" not in output
    assert "VIX 20.12" in output
    assert "具体数字已从正文中省略" in output
    assert "不提供组合仓位比例" in output
    assert "数据边界" in output


def test_us_strategy_blueprint_has_no_stock_style_execution_rules():
    prompt = get_market_strategy_blueprint("us").to_prompt_block().lower()

    assert "stop-losses" in prompt
    assert "price targets" in prompt
    assert "volume confirm the move" not in prompt
    assert "clear invalidation points" not in prompt
