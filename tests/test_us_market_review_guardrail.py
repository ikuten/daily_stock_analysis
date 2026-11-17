# -*- coding: utf-8 -*-
"""Regression tests for the US market recap output boundary."""

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.market_analyzer import (
    MarketAnalyzer,
    MarketIndex,
    MarketOverview,
    _classify_us_market_news_source,
    _select_us_market_news,
)
from src.core.market_strategy import get_market_strategy_blueprint
from src.core.market_profile import get_profile


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

| Index | Last | Change % |
|---|---:|---:|
| SPX | 7561.67 | -0.40% |
| Unsupported macro | 123.45 | 9.99% |

### 三、消息催化
AI 股因某机构报告下跌。

### 四、后市观察
如果收益率突破5.1%，指数可能继续下跌。

### 五、风险提示
某项未经核实的指标达到6.7%，风险上升。

### 六、策略判断
建议仓位三成，跌破 27000 止损，目标价 29000。
"""
    news = [
        {
            "title": "Federal Reserve releases meeting minutes",
            "snippet": "The minutes discuss the policy outlook.",
            "source": "Reuters",
            "published_date": "2026-11-15",
            "url": "https://www.reuters.com/world/us/fed-minutes-2026-11-15/",
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
    assert "https://www.reuters.com/world/us/fed-minutes-2026-11-15/" in output
    assert "不代表这些事件已被证实为指数涨跌原因" in output
    assert "模型生成的指数点评" not in output
    assert "| Index | Last | Change % |" in output
    assert "Unsupported macro" not in output
    assert "建议仓位三成" not in output
    assert "27000 止损" not in output
    assert "29000" not in output
    assert "5.1%" not in output
    assert "6.7%" not in output
    assert "已从正文中省略" in output
    assert "不提供组合仓位比例" in output
    assert "数据边界" in output


def test_us_strategy_blueprint_has_no_stock_style_execution_rules():
    prompt = get_market_strategy_blueprint("us").to_prompt_block().lower()

    assert "stop-losses" in prompt
    assert "price targets" in prompt
    assert "volume confirm the move" not in prompt
    assert "clear invalidation points" not in prompt



def test_us_market_recap_guardrail_handles_template_fallback_section_five():
    analyzer = object.__new__(MarketAnalyzer)
    analyzer.region = "us"
    analyzer._get_review_language = lambda: "zh"
    report = """## US market recap

### 三、板块主线
模型声称AI板块领涨，芯片股资金流入。

### 四、资金与情绪
资金净流入达到100亿美元，市场广度明显改善。

### 五、消息催化
模型编造的消息催化内容。

### 六、策略框架
建议仓位三成并设置止损。
"""
    news = [
        {
            "title": "FOMC minutes published",
            "snippet": "The committee discussed its policy outlook.",
            "source": "Federal Reserve",
            "published_date": "2026-11-15",
            "url": "https://www.federalreserve.gov/monetary20261115a.htm",
        }
    ]

    output = analyzer._apply_us_market_review_guardrail(
        report,
        MarketOverview(date="2026-11-16"),
        news,
    )

    assert "### 五、消息催化" in output
    assert "模型编造的消息催化内容" not in output
    assert "Federal Reserve" in output
    assert "建议仓位三成" not in output
    assert "设置止损" not in output
    assert "模型声称AI板块领涨" not in output
    assert "无法判断行业强弱" in output
    assert "无法判断市场宽度及资金净流入/流出" in output
    assert "100亿美元" not in output



def test_us_market_recap_guardrail_handles_emoji_headings_and_quote_dates():
    analyzer = object.__new__(MarketAnalyzer)
    analyzer.region = "us"
    analyzer._get_review_language = lambda: "zh"
    report = """## 2026-11-16 大盘复盘

> 今日美股科技领跌，某新闻导致纳指下跌，后续关注支撑买盘。

模型又补充了一段没有来源的导语，声称科技股资金正在流出。

### 一、盘面总览 🌐
模型生成的市场判断仍包含未经核实的因果关系。
- **操作建议**：信号分化，控制仓位并等待量价确认。

### 二、指数结构 📈
模型生成的指数点评声称资金流向防守板块。
| 指数 | 最新 | 涨跌幅 |
|---|---:|---:|
| SPX | 7561.67 | -0.40% |

### 三、消息催化 📰
模型编造的新闻内容。

### 四、后市观察 👁️
纳指关键支撑买盘即将出现，跌破支撑应减仓。

### 五、风险提示 ⚠️
美债收益率上升导致科技股下跌。

### 六、策略判断 🎯
建议控制仓位，设置止损和目标价。
"""
    news = [{
        "title": "Dated market news lead",
        "snippet": "A retrieved source discusses the policy outlook.",
        "source": "Reuters",
        "published_date": "2026-11-15",
        "url": "https://www.reuters.com/markets/us/dated-news-2026-11-15/",
    }]
    overview = MarketOverview(
        date="2026-11-16",
        indices=[
            MarketIndex(code="SPX", name="标普500指数", current=7561.67, prev_close=7592.03, change_pct=-0.40, quote_date="2026-11-13"),
            MarketIndex(code="NDX", name="纳斯达克100指数", current=27943.74, prev_close=28288.82, change_pct=-1.22, quote_date="2026-11-13"),
            MarketIndex(code="IXIC", name="纳斯达克综合指数", current=25136.27, prev_close=25366.94, change_pct=-0.91, quote_date="2026-11-13"),
            MarketIndex(code="DJI", name="道琼斯工业指数", current=52347.55, prev_close=52295.27, change_pct=0.10, quote_date="2026-11-13"),
            MarketIndex(code="VIX", name="VIX恐慌指数", current=19.94, prev_close=19.30, change_pct=3.32, quote_date="2026-11-13"),
        ],
    )

    output = analyzer._apply_us_market_review_guardrail(report, overview, news)

    assert "行情日期：2026-11-13" in output
    assert "报告生成日期：2026-11-16" in output
    assert "今日美股科技领跌" not in output
    assert "模型又补充了一段没有来源的导语" not in output
    assert "某新闻导致纳指下跌" not in output
    assert "控制仓位并等待量价确认" not in output
    assert "支撑买盘" not in output
    assert "模型生成的市场判断" not in output
    assert "模型生成的指数点评" not in output
    assert "Reuters" in output
    assert "2026-11-15" in output
    assert "https://www.reuters.com/markets/us/dated-news-2026-11-15/" in output
    assert "不提供组合仓位比例" in output
    assert "美债收益率上升导致科技股下跌" not in output
    assert "新闻只作为背景线索" in output


def test_us_news_source_classifier_uses_publisher_domain_and_tiers():
    assert _classify_us_market_news_source({
        "source": "Reuters",
        "url": "https://www.reuters.com/markets/us/story/",
    })["tier"] == "A"
    assert _classify_us_market_news_source({
        "source": "CNBC",
        "url": "https://www.cnbc.com/2026/11/16/market-story.html",
    })["tier"] == "B"
    # A trusted publisher's quote page is not an editorial news article.
    assert _classify_us_market_news_source({
        "source": "Yahoo Finance",
        "url": "https://finance.yahoo.com/quote/%5EGSPC/",
    })["tier"] == "C"
    assert _classify_us_market_news_source({
        "source": "Moomoo",
        "url": "https://www.moomoo.com/community/feed/example",
    })["tier"] == "C"
    # A source label cannot override an unrecognized publisher URL.
    assert _classify_us_market_news_source({
        "source": "Reuters",
        "url": "https://unknown-publisher.example/story/",
    })["tier"] == "C"


def test_us_news_selection_prioritizes_a_and_b_and_drops_low_trust_duplicates():
    items = [
        {
            "title": "Community rumor",
            "source": "Moomoo",
            "url": "https://www.moomoo.com/community/feed/rumor",
        },
        {
            "title": "CNBC report",
            "source": "CNBC",
            "url": "https://www.cnbc.com/2026/11/16/report.html",
        },
        {
            "title": "Reuters report",
            "source": "Reuters",
            "url": "https://www.reuters.com/markets/us/report.html",
        },
        {
            "title": "Duplicate Reuters report",
            "source": "Reuters",
            "url": "https://www.reuters.com/markets/us/report.html",
        },
        {
            "title": "Unclassified blog",
            "source": "Unknown",
            "url": "https://unknown.example/blog",
        },
    ]

    selected = _select_us_market_news(items, max_results=6)

    assert [item["title"] for item in selected] == ["Reuters report", "CNBC report"]


def test_us_market_guardrail_excludes_low_trust_news_from_final_report():
    analyzer = object.__new__(MarketAnalyzer)
    analyzer.region = "us"
    analyzer._get_review_language = lambda: "zh"
    news = [
        {
            "title": "Reuters verified headline",
            "snippet": "A dated report describes the policy announcement.",
            "source": "Reuters",
            "published_date": "2026-11-15",
            "url": "https://www.reuters.com/world/us/policy-story/",
        },
        {
            "title": "Moomoo community rumor",
            "snippet": "Unverified social discussion.",
            "source": "Moomoo",
            "published_date": "2026-11-15",
            "url": "https://www.moomoo.com/community/feed/rumor",
        },
        {
            "title": "Unknown site speculation",
            "snippet": "An unverified blog post.",
            "source": "Unknown",
            "published_date": "2026-11-15",
            "url": "https://unknown.example/speculation",
        },
    ]
    overview = MarketOverview(date="2026-11-16")

    output = analyzer._apply_us_market_review_guardrail(
        "## 2026-11-16 US Market Recap\n\n### 一、盘面总览\nModel text.\n\n### 五、消息催化\nUntrusted draft headlines.\n\n### 六、策略判断\nDraft strategy.",
        overview,
        news,
    )

    assert "Reuters verified headline" in output
    assert "A级（官方/一线通讯社）" in output
    assert "Moomoo community rumor" not in output
    assert "Unknown site speculation" not in output
    assert "未发现达到 A/B 级标准" not in output


def test_us_market_guardrail_discloses_when_only_low_trust_news_is_available():
    analyzer = object.__new__(MarketAnalyzer)
    analyzer.region = "us"
    analyzer._get_review_language = lambda: "zh"
    news = [{
        "title": "Community-only claim",
        "snippet": "No editorial source.",
        "source": "Moomoo",
        "published_date": "2026-11-15",
        "url": "https://www.moomoo.com/community/feed/claim",
    }]

    output = analyzer._apply_us_market_review_guardrail(
        "## 2026-11-16 US Market Recap\n\n### 一、盘面总览\nModel text.",
        MarketOverview(date="2026-11-16"),
        news,
    )

    assert "未发现达到 A/B 级标准的新闻来源" in output
    assert "Community-only claim" not in output



def test_us_market_news_queries_cover_reputable_publishers_and_official_macro_sources():
    queries = get_profile("us").news_queries

    assert len(queries) == 3
    assert any("Reuters" in query and "Bloomberg" in query for query in queries)
    assert any("Federal Reserve" in query and "CPI" in query for query in queries)
    assert any("Nasdaq" in query and "Financial Times" in query for query in queries)
