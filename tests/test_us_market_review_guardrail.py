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

### 二、指数结构
模型生成的指数点评包含未经核实的因果判断。

### 三、消息催化
AI 股因某机构报告下跌。

### 四、后市观察
如果收益率突破5.1%，指数可能继续下跌。

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
    overview = MarketOverview(
        date="2026-11-16",
        indices=[
            MarketIndex(code="SPX", name="标普500指数", current=7561.67, prev_close=7592.03, change_pct=-0.40),
            MarketIndex(code="NDX", name="纳斯达克100指数", current=27943.74, prev_close=28288.82, change_pct=-1.22),
            MarketIndex(code="IXIC", name="纳斯达克综合指数", current=25136.27, prev_close=25366.94, change_pct=-0.91),
            MarketIndex(code="DJI", name="道琼斯工业指数", current=52347.55, prev_close=52295.27, change_pct=0.10),
            MarketIndex(code="VIX", name="VIX恐慌指数", current=20.12, prev_close=19.43, change_pct=3.58),
        ],
    )

    output = analyzer._apply_us_market_review_guardrail(report, overview, news)

    assert "行情时点" in output
    assert "主要指数涨跌分化" in output
    assert "最新可用值 20.12" in output
    assert "Reuters" in output
    assert "2026-11-15" in output
    assert "https://example.com/fed-minutes" in output
    assert "不代表这些事件已被证实为指数涨跌原因" in output
    assert "模型生成的指数点评" not in output
    assert "建议仓位三成" not in output
    assert "27000 止损" not in output
    assert "29000" not in output
    assert "5.1%" not in output
    assert "具体数字已从正文中省略" in output
    assert "不提供组合仓位比例" in output
    assert "数据边界" in output


def test_us_strategy_blueprint_has_no_stock_style_execution_rules():
    prompt = get_market_strategy_blueprint("us").to_prompt_block().lower()

    assert "stop-losses" in prompt
    assert "price targets" in prompt
    assert "volume confirm the move" not in prompt
    assert "clear invalidation points" not in prompt
