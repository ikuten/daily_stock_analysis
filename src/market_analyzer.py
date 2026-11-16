# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块
===================================

职责：
1. 获取大盘指数数据（上证、深证、创业板）
2. 搜索市场新闻形成复盘情报
3. 使用大模型生成每日大盘复盘报告
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from inspect import getattr_static
from typing import Optional, Dict, Any, List

import pandas as pd

from src.agent.provider_trace import resolved_model_provider_identity
from src.config import get_config
from src.report_language import normalize_report_language
from src.search_service import SearchService
from src.core.market_profile import get_profile, MarketProfile
from src.core.market_strategy import get_market_strategy_blueprint
from src.llm.backend_registry import (
    LOCAL_CLI_GENERATION_BACKEND_IDS,
    LITELLM_BACKEND_ID,
    resolve_generation_backend_id,
    resolve_generation_fallback_backend_id,
)
from src.llm.generation_backend import GenerationError, GenerationResult
from src.schemas.market_light import MARKET_LIGHT_REGIONS, MarketLightSnapshot
from src.services.run_diagnostics import record_llm_run, record_llm_run_started
from src.services.intelligence_service import IntelligenceService
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)

_LEGACY_ANALYZER_BACKEND_ID = "legacy_analyzer"

_ENGLISH_SECTION_PATTERNS = {
    "market_summary": r"###\s*(?:1\.\s*)?Market Summary",
    "index_commentary": r"###\s*(?:2\.\s*)?(?:Index Commentary|Major Indices)",
    "sector_highlights": r"###\s*(?:4\.\s*)?(?:Sector Highlights|Sector/Theme Highlights)",
}

_CHINESE_SECTION_PATTERNS = {
    "market_summary": r"###\s*一、(?:盘面总览|市场总结)",
    "index_commentary": r"###\s*二、(?:指数结构|指数点评|主要指数)",
    "sector_highlights": r"###\s*三、(?:板块主线|热点解读|板块表现)",
    "funds_sentiment": r"###\s*四、(?:资金与情绪|资金动向)",
    "news_catalysts": r"###\s*五、(?:消息催化|后市展望)",
}


@dataclass
class MarketIndex:
    """大盘指数数据"""
    code: str                    # 指数代码
    name: str                    # 指数名称
    current: float = 0.0         # 当前点位
    change: float = 0.0          # 涨跌点数
    change_pct: float = 0.0      # 涨跌幅(%)
    open: float = 0.0            # 开盘点位
    high: float = 0.0            # 最高点位
    low: float = 0.0             # 最低点位
    prev_close: float = 0.0      # 昨收点位
    volume: float = 0.0          # 成交量（手）
    amount: float = 0.0          # 成交额（元）
    amplitude: float = 0.0       # 振幅(%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """市场概览数据"""
    date: str                           # 日期
    indices: List[MarketIndex] = field(default_factory=list)  # 主要指数
    up_count: int = 0                   # 上涨家数
    down_count: int = 0                 # 下跌家数
    flat_count: int = 0                 # 平盘家数
    limit_up_count: int = 0             # 涨停家数
    limit_down_count: int = 0           # 跌停家数
    total_amount: float = 0.0           # 两市成交额（亿元）
    # north_flow: float = 0.0           # 北向资金净流入（亿元）- 已废弃，接口不可用
    
    # 板块涨幅榜
    top_sectors: List[Dict] = field(default_factory=list)     # 涨幅前5板块
    bottom_sectors: List[Dict] = field(default_factory=list)  # 跌幅前5板块
    top_concepts: List[Dict] = field(default_factory=list)    # 涨幅前5概念
    bottom_concepts: List[Dict] = field(default_factory=list) # 跌幅前5概念


@dataclass
class MarketLightReviewResult:
    """Internal market-review parts built from one overview fetch."""

    overview: MarketOverview
    report: str
    market_light_snapshot: Optional[Dict[str, Any]]
    structured_payload: Dict[str, Any] = field(default_factory=dict)


class MarketAnalyzer:
    """
    大盘复盘分析器
    
    功能：
    1. 获取大盘指数实时行情
    2. 获取市场涨跌统计
    3. 获取板块涨跌榜
    4. 搜索市场新闻
    5. 生成大盘复盘报告
    """
    
    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        analyzer=None,
        region: str = "cn",
        config: Optional[Any] = None,
    ):
        """
        初始化大盘分析器

        Args:
            search_service: 搜索服务实例
            analyzer: AI分析器实例（用于调用LLM）
            region: 市场区域 cn=A股 hk=港股 us=美股 jp=日本 kr=韩国
            config: 本次复盘使用的配置；未传时读取全局配置
        """
        self.config = config or get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.data_manager = DataFetcherManager()
        self.region = region if region in ("cn", "us", "hk", "jp", "kr") else "cn"
        self.profile: MarketProfile = get_profile(self.region)
        self.strategy = get_market_strategy_blueprint(self.region)

    def _log_context(self) -> str:
        return f"component=market_review region={self.region}"

    @staticmethod
    def _resolve_configured_response_provider(
        configured_model: str,
        response_model: str,
        model_list: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Match the actual response model against all deployments of one alias."""
        normalized_configured_model = str(configured_model or "").strip()
        normalized_response_model = str(response_model or "").strip().lower()
        if not normalized_configured_model or not normalized_response_model or not model_list:
            return ""

        for entry in model_list:
            params = entry.get("litellm_params", {}) or {}
            model_name = str(entry.get("model_name") or "").strip()
            if not model_name:
                model_name = str(params.get("model") or "").strip()
            if model_name != normalized_configured_model:
                continue

            deployment_model = str(params.get("model") or "").strip()
            if deployment_model.lower() != normalized_response_model:
                continue

            normalized_deployment_model = deployment_model.lower()
            if normalized_deployment_model.startswith("openai/~") or "openrouter" in normalized_deployment_model:
                return "openrouter"

            _resolved_model, resolved_provider = resolved_model_provider_identity(
                deployment_model,
            )
            if resolved_provider:
                return resolved_provider
        return ""

    def _resolve_recorded_provider(
        self,
        *,
        provider: str,
        model: str,
        backend: str,
        usage_provider: str = "",
        response_model: str = "",
    ) -> str:
        """Resolve LiteLLM router aliases before persisting diagnostics."""
        normalized_backend = str(backend or "").strip().lower()
        normalized_model = str(model or "").strip()
        normalized_usage_provider = str(usage_provider or "").strip()
        normalized_response_model = str(response_model or "").strip()
        normalized_provider = str(provider or "").strip()
        resolved_route_provider = ""
        if normalized_backend == "litellm" and normalized_model:
            resolved_route_model, resolved_route_provider = resolved_model_provider_identity(
                normalized_model,
                getattr(self.config, "llm_model_list", None) or [],
            )
            normalized_route = str(resolved_route_model or normalized_model).strip().lower()
            if normalized_route.startswith("openai/~") or "openrouter" in normalized_route:
                resolved_route_provider = "openrouter"
        resolved_response_provider = ""
        if normalized_response_model:
            resolved_response_provider = self._resolve_configured_response_provider(
                normalized_model,
                normalized_response_model,
                getattr(self.config, "llm_model_list", None) or [],
            )
            if resolved_response_provider == "openrouter":
                return resolved_response_provider
        if normalized_usage_provider:
            return normalized_usage_provider
        if normalized_response_model:
            if resolved_response_provider:
                return resolved_response_provider
            if resolved_route_provider == "openrouter":
                return resolved_route_provider
            _wire_model, resolved_provider = resolved_model_provider_identity(
                normalized_response_model,
            )
            if resolved_provider:
                return resolved_provider
        if normalized_backend != "litellm" or not normalized_model:
            return normalized_provider
        return resolved_route_provider or normalized_provider or "openai"

    def _resolve_recorded_error_model(
        self,
        *,
        error: Any,
        fallback_model: str = "",
    ) -> str:
        """Preserve route/model diagnostics for LiteLLM configuration failures."""
        details = getattr(error, "details", None)
        if isinstance(details, dict):
            visited: set[int] = set()

            def _find_error_model(payload: Dict[str, Any]) -> str:
                payload_id = id(payload)
                if payload_id in visited:
                    return ""
                visited.add(payload_id)
                for key in ("last_model", "route_name"):
                    candidate = str(payload.get(key) or "").strip()
                    if candidate:
                        return candidate
                fallback_error = payload.get("fallback_error")
                if isinstance(fallback_error, dict):
                    nested_details = fallback_error.get("details")
                    if isinstance(nested_details, dict):
                        candidate = _find_error_model(nested_details)
                        if candidate:
                            return candidate
                    return _find_error_model(fallback_error)
                return ""

            candidate = _find_error_model(details)
            if candidate:
                return candidate
        backend = str(getattr(error, "backend", "") or "").strip()
        configured_model = str(getattr(self.config, "litellm_model", "") or "").strip()
        normalized_fallback_model = str(fallback_model or "").strip()
        if backend == "litellm":
            return normalized_fallback_model or configured_model or backend
        return normalized_fallback_model or backend

    def _get_output_language(self) -> str:
        """Return the truthful report language (zh/en/ko) for payload and directives."""
        return normalize_report_language(
            getattr(getattr(self, "config", None), "report_language", "zh")
        )

    def _get_review_language(self) -> str:
        # Structural/template language. Korean reuses the English scaffolding;
        # the Korean output directive is applied in the prompt builder.
        language = self._get_output_language()
        return "en" if language == "ko" else language

    def _get_template_review_language(self) -> str:
        return self._get_review_language()

    def _get_market_scope_name(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "us":
            return "US market" if review_language == "en" else "美股市场"
        if self.region == "hk":
            return "Hong Kong market" if review_language == "en" else "港股市场"
        if self.region == "jp":
            return "Japan market" if review_language == "en" else "日本市场"
        if self.region == "kr":
            return "Korea market" if review_language == "en" else "韩国市场"
        if review_language == "en":
            return "A-share market"
        return "A股市场"

    def _get_turnover_unit_label(self) -> str:
        """Return the turnover unit label for the current market/language."""
        if self.region == "us":
            return "USD bn" if self._get_review_language() == "en" else "十亿美元"
        if self.region == "hk":
            return "HKD bn" if self._get_review_language() == "en" else "十亿港元"
        if self.region == "jp":
            return "JPY bn" if self._get_review_language() == "en" else "十亿日元"
        if self.region == "kr":
            return "KRW bn" if self._get_review_language() == "en" else "十亿韩元"
        return "CNY 100m" if self._get_review_language() == "en" else "亿"

    def _format_turnover_value(self, amount_raw: float) -> str:
        """Format raw turnover according to market-specific units."""
        if amount_raw == 0.0:
            return "N/A"
        if self.region in ("us", "hk", "jp", "kr"):
            return f"{amount_raw / 1e9:.2f}"
        if amount_raw > 1e6:
            return f"{amount_raw / 1e8:.0f}"
        return f"{amount_raw:.0f}"

    def _get_index_change_arrow(self, change_pct: float) -> str:
        if change_pct == 0:
            return "⚪"
        color_scheme = getattr(getattr(self, "config", None), "market_review_color_scheme", "green_up")
        if color_scheme == "red_up":
            return "🔴" if change_pct > 0 else "🟢"
        return "🟢" if change_pct > 0 else "🔴"

    def _get_review_title(self, date: str) -> str:
        if self._get_review_language() == "en":
            market_names = {
                "us": "US Market Recap",
                "hk": "HK Market Recap",
                "jp": "Japan Market Recap",
                "kr": "Korea Market Recap",
            }
            market_name = market_names.get(self.region, "A-share Market Recap")
            return f"## {date} {market_name}"
        return f"## {date} 大盘复盘"

    def _get_index_hint(self) -> str:
        if self._get_review_language() == "en":
            if self.region == "us":
                return (
                    "Analyze the S&P 500 (SPX), Nasdaq-100 (NDX), and Dow separately. "
                    "Do not confuse Nasdaq-100 (NDX) with Nasdaq Composite (IXIC), "
                    "and never substitute one index's price, return, constituents, or news for another."
                )
            if self.region == "hk":
                return "Analyze the key moves in the HSI, Hang Seng Tech, HSCEI, and other major indices."
            if self.region == "jp":
                return "Analyze the key moves in the Nikkei 225, TOPIX, and other major Japanese indices."
            if self.region == "kr":
                return "Analyze the key moves in the KOSPI, KOSDAQ, and other major Korean indices."
            return "Analyze the price action in the SSE, SZSE, ChiNext, and other major indices."
        return self.profile.prompt_index_hint

    def _get_strategy_prompt_block(self) -> str:
        if self.region == "hk" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Hong Kong Market Regime Strategy
Focus on HSI trend, southbound flow dynamics, and sector rotation to define next-session risk posture.

### Strategy Principles
- Read market regime from HSI, HSTECH, and HSCEI alignment first.
- Track southbound capital flow as a key sentiment driver.
- Translate recap into actionable risk-on/risk-off stance with clear invalidation points.

### Analysis Dimensions
- Trend Regime: Classify the market as momentum, range, or risk-off.
  - Are HSI/HSTECH/HSCEI directionally aligned
  - Did volume confirm the move
  - Are key index levels reclaimed or lost
- Capital Flows: Map southbound flow and macro narrative into equity risk appetite.
  - Southbound net flow direction and magnitude
  - USD/HKD and China policy implications
  - Breadth and leadership concentration
- Sector Themes: Identify persistent leaders and vulnerable laggards.
  - Tech/internet platform trend persistence
  - Financials/property sensitivity to policy shifts
  - Defensive vs growth factor rotation

### Action Framework
- Risk-on: broad index breakout with expanding southbound participation.
- Neutral: mixed index signals; focus on selective relative strength.
- Risk-off: failed breakouts and rising volatility; prioritize capital preservation."""
        if self.region == "jp" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Japan Market Regime Strategy
Focus on Nikkei 225, TOPIX, currency dynamics, and global risk appetite to define the next-session trading plan.

### Strategy Principles
- Read Nikkei 225 and TOPIX alignment first, then assess yen moves, semiconductor/export chains, and financials.
- Translate index conclusions into position sizing, trading pace, and risk-control actions.
- Base judgments only on available index data, news, and price action without inventing breadth or sector statistics.

### Analysis Dimensions
- Trend Regime: Classify Japan equities as advancing, range-bound, or defensive.
  - Are Nikkei 225 and TOPIX directionally aligned
  - Have key index ranges been reclaimed or lost
  - Are large-cap weights and growth chains moving together
- Macro & FX: Map yen, rates, and global risk appetite into equity impact.
  - Yen direction and implications for exporters
  - Bank of Japan and US Treasury yield narratives
  - Overseas technology and semiconductor read-through
- Theme Signals: Identify durable leadership and crowded areas to avoid.
  - Semiconductor, automation, and auto-chain persistence
  - Rotation between financials and domestic-demand stocks
  - Whether news catalysts confirm price action

### Action Framework
- Risk-on: major indices rise together with improving external risk appetite and stronger leadership.
- Neutral: index divergence or FX disruption; avoid chasing and wait for confirmation.
- Risk-off: major indices weaken or external risk rises; prioritize position control."""
        if self.region == "kr" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Korea Market Regime Strategy
Focus on KOSPI, KOSDAQ, semiconductor heavyweights, and global technology risk appetite to define the next-session trading plan.

### Strategy Principles
- Read KOSPI and KOSDAQ alignment first, then assess heavyweight signals from Samsung Electronics, SK Hynix, and related technology leaders.
- Separate broad index beta, semiconductor cycle exposure, and growth-stock risk appetite.
- Base judgments only on available index data, news, and price action without inventing breadth or sector statistics.

### Analysis Dimensions
- Trend Regime: Classify Korea equities as advancing, range-bound, or defensive.
  - Are KOSPI and KOSDAQ directionally aligned
  - Are heavyweight technology names supporting the indices
  - Have key support or resistance levels been reclaimed or lost
- Technology Cycle: Map semiconductor, AI hardware, and global technology moves into Korea equity risk.
  - Memory and semiconductor-chain catalysts
  - US technology-market read-through
  - Foreign investor risk appetite signals
- Theme Signals: Identify durable leadership and crowded areas to avoid.
  - Rotation across batteries, autos, and internet platforms
  - KOSDAQ growth-stock risk appetite
  - Whether news catalysts confirm price action

### Action Framework
- Risk-on: KOSPI and KOSDAQ rise together with confirmed technology leadership and improving external risk appetite.
- Neutral: index or heavyweight divergence; keep sizing controlled and wait for confirmation.
- Risk-off: technology heavyweights weaken or external risk rises; prioritize drawdown control."""
        if self.region == "us" and self._get_review_language() == "zh":
            return """## 美股市场观察框架
聚焦指数趋势、波动率与有来源的宏观信息，形成定性市场观察，不生成交易计划。

### 策略原则
- 分别观察标普500（SPX）、纳斯达克100（NDX）、纳斯达克综合指数（IXIC）和道琼斯，不得混用。
- 区分行情数据、新闻事实与分析推断；重要新闻必须带来源和日期。
- 未提供成交量、PE、市场宽度或资金流数据时，明确标记未知，不作推断。
- 不输出组合仓位、买卖点、止损位、目标价或短线交易触发条件。

### 分析维度
- 趋势结构：仅根据同一完整交易日的指数数据判断方向与分化。
- 宏观环境：只总结有来源的利率、通胀、美元及美联储信息。
- 波动率：结合已提供的 VIX 数值描述风险环境，不将相关性直接写成因果关系。

### 市场状态
- 积极：主要指数方向一致，且输入证据支持风险偏好改善。
- 中性：指数信号分化或关键数据缺失，明确说明未知项。
- 防守：指数走弱且输入的波动率/宏观数据支持风险上升。

仅作市场观察，不提供投资组合仓位或短线交易指令。"""
        if not (self.region == "cn" and self._get_review_language() == "en"):
            return self.strategy.to_prompt_block()
        return """## Strategy Blueprint: A-share Three-Phase Recap Strategy
Focus on index trend, liquidity, and sector rotation to shape the next-session trading plan.

### Strategy Principles
- Read index direction first, then confirm liquidity structure, and finally test sector persistence.
- Every conclusion must map to position sizing, trading pace, and risk-control actions.
- Base judgments on today's data and the latest 3-day news flow without inventing unverified information.

### Analysis Dimensions
- Trend Structure: Determine whether the market is in an uptrend, range, or defensive phase.
  - Are the SSE, SZSE, and ChiNext moving in the same direction
  - Is the market advancing on expanding volume or slipping on contracting volume
  - Have key support or resistance levels been reclaimed or broken
- Liquidity & Sentiment: Identify near-term risk appetite and market temperature.
  - Advance/decline breadth and limit-up/limit-down structure
  - Whether turnover is expanding or fading
  - Whether high-beta leaders are showing divergence
- Leading Themes: Distill tradable leadership and areas to avoid.
  - Whether leading sectors have clear event catalysts
  - Whether sector leaders are pulling the group higher
  - Whether weakness is broadening across lagging sectors

### Action Framework
- Offensive: indices rise in sync, turnover expands, and core themes strengthen.
- Balanced: index divergence or low-volume consolidation; keep sizing controlled and wait for confirmation.
- Defensive: indices weaken and laggards broaden; prioritize risk control and de-risking."""

    def _get_strategy_markdown_block(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "hk" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify the market as momentum, range, or risk-off based on HSI/HSTECH/HSCEI alignment.
- **Capital Flows**: Track southbound flow direction and macro narrative for risk appetite signals.
- **Sector Themes**: Focus on tech/internet platform persistence and financials/property policy sensitivity.
"""
        if self.region == "jp" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify Japan equities as advancing, range-bound, or defensive based on Nikkei 225/TOPIX alignment.
- **Macro & FX**: Track yen, rates, and global risk appetite for exporter and financial-sector implications.
- **Theme Signals**: Focus on semiconductor, automation, auto-chain, financial, and domestic-demand rotation.
"""
        if self.region == "kr" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify Korea equities as advancing, range-bound, or defensive based on KOSPI/KOSDAQ alignment.
- **Technology Cycle**: Track semiconductor, AI hardware, and global technology read-through for market risk appetite.
- **Theme Signals**: Focus on battery, auto, internet-platform, and KOSDAQ growth-stock rotation.
"""
        if self.region == "us" and review_language == "zh":
            return """### 六、策略框架
- **趋势结构**：判断市场在进攻、震荡与防守中的状态是否一致。
- **资金与情绪**：结合波动率、宽度和主题轮动评估风险偏好。
- **主题主线**：识别可延续和可放大的行业主线与防守线索。
"""
        if not (self.region == "cn" and review_language == "en"):
            return self.strategy.to_markdown_block()
        return """### 6. Strategy Framework
- **Trend Structure**: Determine whether the market is in an uptrend, range, or defensive phase.
- **Liquidity & Sentiment**: Track breadth, turnover expansion, and whether leaders are diverging.
- **Leading Themes**: Focus on sectors with catalysts and sustained leadership while avoiding broadening weakness.
"""

    def _get_market_mood_text(self, mood_key: str, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if review_language == "en":
            mapping = {
                "strong_up": "strong gains",
                "mild_up": "moderate gains",
                "mild_down": "mild losses",
                "strong_down": "clear weakness",
                "range": "range-bound trading",
            }
        else:
            mapping = {
                "strong_up": "强势上涨",
                "mild_up": "小幅上涨",
                "mild_down": "小幅下跌",
                "strong_down": "明显下跌",
                "range": "震荡整理",
            }
        return mapping[mood_key]

    def get_market_overview(self) -> MarketOverview:
        """
        获取市场概览数据
        
        Returns:
            MarketOverview: 市场概览数据对象
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)
        
        # 1. 获取主要指数行情（按 region 切换 A 股/美股）
        overview.indices = self._get_main_indices()

        # 2. 获取涨跌统计（A 股有，美股无等效数据）
        if self.profile.has_market_stats:
            self._get_market_statistics(overview)

        # 3. 获取板块涨跌榜（A 股有，美股暂无）
        if self.profile.has_sector_rankings:
            self._get_sector_rankings(overview)
            self._get_concept_rankings(overview)
        
        # 4. 获取北向资金（可选）
        # self._get_north_flow(overview)
        
        return overview

    
    def _get_main_indices(self) -> List[MarketIndex]:
        """获取主要指数实时行情"""
        indices = []

        try:
            logger.info("[大盘] %s action=get_main_indices status=start", self._log_context())

            # 使用 DataFetcherManager 获取指数行情（按 region 切换）
            data_list = self.data_manager.get_main_indices(region=self.region)

            if data_list:
                for item in data_list:
                    index = MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=item['current'],
                        change=item['change'],
                        change_pct=item['change_pct'],
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        prev_close=item['prev_close'],
                        volume=item['volume'],
                        amount=item['amount'],
                        amplitude=item['amplitude']
                    )
                    indices.append(index)

            if not indices:
                logger.warning("[大盘] %s action=get_main_indices status=empty", self._log_context())
            else:
                logger.info(
                    "[大盘] %s action=get_main_indices status=success count=%d",
                    self._log_context(),
                    len(indices),
                )

        except Exception as e:
            logger.error("[大盘] %s action=get_main_indices status=failed error=%s", self._log_context(), e)

        return indices

    def _get_market_statistics(self, overview: MarketOverview):
        """获取市场涨跌统计"""
        try:
            logger.info("[大盘] %s action=get_market_stats status=start", self._log_context())

            stats = self.data_manager.get_market_stats(purpose=f"market_review:{self.region}")

            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)

                logger.info(
                    "[大盘] %s action=get_market_stats status=success up=%s down=%s flat=%s "
                    "limit_up=%s limit_down=%s amount=%.0f亿",
                    self._log_context(),
                    overview.up_count,
                    overview.down_count,
                    overview.flat_count,
                    overview.limit_up_count,
                    overview.limit_down_count,
                    overview.total_amount,
                )
            else:
                logger.warning("[大盘] %s action=get_market_stats status=empty", self._log_context())

        except Exception as e:
            logger.error("[大盘] %s action=get_market_stats status=failed error=%s", self._log_context(), e)

    def _get_sector_rankings(self, overview: MarketOverview):
        """获取板块涨跌榜"""
        try:
            logger.info("[大盘] %s action=get_sector_rankings status=start", self._log_context())

            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)

            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors

                logger.info(
                    "[大盘] %s action=get_sector_rankings status=success top=%s bottom=%s",
                    self._log_context(),
                    [s['name'] for s in overview.top_sectors],
                    [s['name'] for s in overview.bottom_sectors],
                )
            else:
                logger.warning("[大盘] %s action=get_sector_rankings status=empty", self._log_context())

        except Exception as e:
            logger.error("[大盘] %s action=get_sector_rankings status=failed error=%s", self._log_context(), e)

    def _get_concept_rankings(self, overview: MarketOverview):
        """获取概念/题材涨跌榜（fail-open）。"""
        try:
            logger.info("[大盘] %s action=get_concept_rankings status=start", self._log_context())

            top_concepts, bottom_concepts = self.data_manager.get_concept_rankings(5)

            if top_concepts or bottom_concepts:
                overview.top_concepts = top_concepts
                overview.bottom_concepts = bottom_concepts

                logger.info(
                    "[大盘] %s action=get_concept_rankings status=success top=%s bottom=%s",
                    self._log_context(),
                    [s.get('name') for s in overview.top_concepts],
                    [s.get('name') for s in overview.bottom_concepts],
                )
            else:
                logger.warning("[大盘] %s action=get_concept_rankings status=empty", self._log_context())

        except Exception as e:
            logger.warning("[大盘] %s action=get_concept_rankings status=failed error=%s", self._log_context(), e)
    
    # def _get_north_flow(self, overview: MarketOverview):
    #     """获取北向资金流入"""
    #     try:
    #         logger.info("[大盘] 获取北向资金...")
    #         
    #         # 获取北向资金数据
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
    #         
    #         if df is not None and not df.empty:
    #             # 取最新一条数据
    #             latest = df.iloc[-1]
    #             if '当日净流入' in df.columns:
    #                 overview.north_flow = float(latest['当日净流入']) / 1e8  # 转为亿元
    #             elif '净流入' in df.columns:
    #                 overview.north_flow = float(latest['净流入']) / 1e8
    #                 
    #             logger.info(f"[大盘] 北向资金净流入: {overview.north_flow:.2f}亿")
    #             
    #     except Exception as e:
    #         logger.warning(f"[大盘] 获取北向资金失败: {e}")
    
    def search_market_news(self) -> List[Dict]:
        """
        搜索市场新闻
        
        Returns:
            新闻列表
        """
        if not self.search_service:
            logger.warning(
                "[大盘] %s action=search_market_news status=skipped reason=no_search_service",
                self._log_context(),
            )
            return []
        
        all_news = []

        # 按 region 使用不同的新闻搜索词
        search_queries = self.profile.news_queries
        review_language = self._get_review_language()
        market_names = {
            "cn": "大盘" if review_language == "zh" else "A-share market",
            "us": "美股市场" if review_language == "zh" else "US market",
            "hk": "港股市场" if review_language == "zh" else "HK market",
            "jp": "日本股市" if review_language == "zh" else "Japan stock market",
            "kr": "韩国股市" if review_language == "zh" else "Korea stock market",
        }
        
        try:
            logger.info("[大盘] %s action=search_market_news status=start", self._log_context())
            
            # 根据 region 设置搜索上下文名称，避免美股搜索被解读为 A 股语境
            market_name = market_names.get(self.region, "大盘")
            for query in search_queries:
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name=market_name,
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(
                        "[大盘] %s action=search_market_news status=query_success count=%d",
                        self._log_context(),
                        len(response.results),
                    )
            
            logger.info(
                "[大盘] %s action=search_market_news status=success count=%d",
                self._log_context(),
                len(all_news),
            )
            
        except Exception as e:
            logger.error("[大盘] %s action=search_market_news status=failed error=%s", self._log_context(), e)
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大模型生成大盘复盘报告
        
        Args:
            overview: 市场概览数据
            news: 市场新闻列表 (SearchResult 对象列表)
            
        Returns:
            大盘复盘报告文本
        """
        backend_error = self._get_analyzer_generation_backend_config_error()
        if backend_error is not None:
            logger.error(
                "[大盘] %s action=generate_review status=failed error_type=%s error=%s",
                self._log_context(),
                type(backend_error).__name__,
                backend_error,
            )
            record_llm_run(
                success=False,
                provider=backend_error.provider or backend_error.backend,
                model=self._resolve_recorded_error_model(error=backend_error),
                call_type="market_review",
                error_type=type(backend_error).__name__,
                error_message=backend_error,
            )
            raise backend_error

        if not self.analyzer or not self.analyzer.is_available():
            logger.warning(
                "[大盘] %s action=generate_review status=fallback_template reason=no_analyzer",
                self._log_context(),
            )
            fallback_review = self._generate_template_review(overview, news)
            return (
                self._apply_us_market_review_guardrail(fallback_review, overview, news)
                if self.region == "us"
                else fallback_review
            )

        # 构建 Prompt
        prompt = self._build_review_prompt(overview, news)

        logger.info("[大盘] %s action=generate_review status=start", self._log_context())
        # Use public analyzer APIs so diagnostics reflect the actual execution backend.
        llm_started_at = time.perf_counter()
        provider, model = self._get_analyzer_generation_backend_identity()
        try:
            record_llm_run_started(
                provider=provider,
                model=model,
                call_type="market_review",
            )
            generation_result = self._generate_market_review_with_metadata(
                prompt,
                provider=provider,
                model=model,
            )
            review = generation_result.text if generation_result else None
            generation_diagnostics = (
                getattr(generation_result, "diagnostics", None) if generation_result is not None else None
            )
            if generation_result is not None:
                model = generation_result.model or model
                usage_payload = getattr(generation_result, "usage", None)
                provider = self._resolve_recorded_provider(
                    provider=generation_result.provider or generation_result.backend or provider,
                    model=model,
                    backend=generation_result.backend or "",
                    usage_provider=(
                        usage_payload.get("provider") if isinstance(usage_payload, dict) else ""
                    ),
                    response_model=(
                        usage_payload.get("response_model") if isinstance(usage_payload, dict) else ""
                    ),
                )
        except Exception as exc:
            error_provider = getattr(exc, "provider", None) or getattr(exc, "backend", None) or provider
            error_model = self._resolve_recorded_error_model(
                error=exc,
                fallback_model=model,
            )
            record_llm_run(
                success=False,
                provider=error_provider,
                model=error_model,
                call_type="market_review",
                duration_ms=int((time.perf_counter() - llm_started_at) * 1000),
                error_type=type(exc).__name__,
                error_message=exc,
            )
            raise

        failed_all_models = (
            isinstance(generation_diagnostics, dict)
            and generation_diagnostics.get("reason") == "all_models_failed"
        )
        record_llm_run(
            success=bool(review),
            provider=provider,
            model=model,
            call_type="market_review",
            duration_ms=int((time.perf_counter() - llm_started_at) * 1000),
            error_type=None if review else ("AllModelsFailed" if failed_all_models else "EmptyResponse"),
            error_message=(
                None
                if review
                else (
                    generation_diagnostics
                    if failed_all_models
                    else "empty market review response"
                )
            ),
        )

        if review:
            logger.info(
                "[大盘] %s action=generate_review status=success length=%d",
                self._log_context(),
                len(review),
            )
            # Inject structured data tables, then apply deterministic US-market
            # output constraints so the LLM cannot reintroduce trading instructions.
            review = self._inject_data_into_review(review, overview, news)
            if self.region == "us":
                review = self._apply_us_market_review_guardrail(review, overview, news)
            return review

        logger.warning(
            "[大盘] %s action=generate_review status=fallback_template reason=empty_llm_response",
            self._log_context(),
        )
        fallback_review = self._generate_template_review(overview, news)
        return (
            self._apply_us_market_review_guardrail(fallback_review, overview, news)
            if self.region == "us"
            else fallback_review
        )

    def _apply_us_market_review_guardrail(
        self,
        review: str,
        overview: MarketOverview,
        news: Optional[List] = None,
    ) -> str:
        """Replace trade-oriented US output sections with source-grounded observation text."""
        text = str(review or "")
        english = self._get_review_language() == "en"

        # The report date is not necessarily the quote's trading date, especially
        # before the US regular session or on a market holiday.
        timing_note = (
            "> Quote timing: figures below are the latest available source values; the report date is not necessarily the market trading date. During pre-market or holidays, use the latest completed session."
            if english
            else "> 行情时点：以下为本次抓取到的最新可用行情；报告日期不等于行情交易日期。美股盘前或休市时段，应以最近一个完整交易日为准。"
        )
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("#"):
            lines.insert(1, "")
            lines.insert(2, timing_note)
            text = "\n".join(lines)

        # Rebuild the market summary and index commentary from the structured
        # snapshot. This removes model-invented causal narratives and signal scores.
        main_indices = []
        vix_index = None
        for index_item in (getattr(overview, "indices", None) or []):
            code = str(getattr(index_item, "code", "") or "").strip().upper()
            current = getattr(index_item, "current", None)
            previous = getattr(index_item, "prev_close", None)
            change_pct = getattr(index_item, "change_pct", None)
            try:
                current_value = float(current)
                previous_value = float(previous)
                change_value = float(change_pct)
            except (TypeError, ValueError):
                continue
            if current_value <= 0 or previous_value <= 0:
                continue
            if code == "VIX":
                vix_index = (index_item, current_value, change_value)
            elif code in {"SPX", "NDX", "IXIC", "DJI"}:
                main_indices.append((index_item, current_value, change_value))

        def _index_label(item: Any) -> str:
            code = str(getattr(item, "code", "") or "").strip().upper()
            name = str(getattr(item, "name", "") or code).strip()
            return f"{name}（{code}）" if not english else f"{name} ({code})"

        if main_indices:
            up_count = sum(1 for _, _, change in main_indices if change > 0)
            down_count = sum(1 for _, _, change in main_indices if change < 0)
            if len(main_indices) < 3:
                regime = "部分指数快照" if not english else "a partial index snapshot"
            elif up_count == len(main_indices):
                regime = "主要指数普涨" if not english else "broadly higher"
            elif down_count == len(main_indices):
                regime = "主要指数普跌" if not english else "broadly lower"
            else:
                regime = "主要指数涨跌分化" if not english else "mixed across major indices"
            summary_body = (
                f"本次最新可用行情快照显示：{regime}。"
                if not english
                else f"The latest available snapshot shows {regime}."
            )
            index_lines = [
                f"- **{_index_label(item)}**：{current_value:,.2f}，涨跌幅 {change_value:+.2f}%"
                if not english
                else f"- **{_index_label(item)}**: {current_value:,.2f}, change {change_value:+.2f}%"
                for item, current_value, change_value in main_indices
            ]
        else:
            summary_body = (
                "本次未获取到足够的主要指数行情快照，暂不判断整体市场方向。"
                if not english
                else "A sufficient major-index snapshot was not available, so no overall market-direction call is made."
            )
            index_lines = [
                "主要指数数据不完整，暂不提供指数方向判断。"
                if not english
                else "Major-index data is incomplete; no index-direction assessment is made."
            ]

        if vix_index:
            _, current_value, change_value = vix_index
            index_lines.append(
                f"- **VIX**：最新可用值 {current_value:.2f}，涨跌幅 {change_value:+.2f}%"
                if not english
                else f"- **VIX**: latest available value {current_value:.2f}, change {change_value:+.2f}%"
            )
        index_body = "\n".join(index_lines)

        def _replace_section(pattern: str, heading: str, body: str, preserve_table: bool = False) -> None:
            nonlocal text
            match = re.search(pattern, text)
            if not match:
                return
            table_block = ""
            if preserve_table:
                table_match = re.search(r"(?m)^\|[^\n]*(?:\n\|[^\n]*)*", match.group(0))
                if table_match:
                    table_block = table_match.group(0)
            replacement = heading + "\n\n" + body
            if table_block:
                replacement += "\n\n" + table_block
            replacement = replacement.rstrip() + "\n\n"
            text = text[:match.start()] + replacement + text[match.end():]

        if english:
            _replace_section(
                r"(?ms)^###\s*1\.\s*Market Summary\s*\n.*?(?=^###\s|\Z)",
                "### 1. Market Summary",
                summary_body + "\n\nThe summary is descriptive only; it does not attribute index moves to news or provide ETF trading instructions.",
            )
            _replace_section(
                r"(?ms)^###\s*2\.\s*(?:Index Commentary|Major Indices)\s*\n.*?(?=^###\s|\Z)",
                "### 2. Index Commentary",
                index_body + "\n\nAll figures refer to the latest available source snapshot.",
                preserve_table=True,
            )
            _replace_section(
                r"(?ms)^###\s*4\.\s*(?:Outlook|Market Outlook)\s*\n.*?(?=^###\s|\Z)",
                "### 4. Outlook",
                "- Compare SPX, NDX, IXIC, and DJI using the same completed trading session.\n- Monitor the subsequent VIX direction; a single reading is not a trend.\n- Discuss macro/news implications only when supported by dated sources; do not infer causation.",
            )
        else:
            _replace_section(
                r"(?ms)^###\s*一、(?:盘面总览|市场总结)\s*\n.*?(?=^###\s|\Z)",
                "### 一、盘面总览",
                summary_body + "\n\n以上仅描述行情快照，不对涨跌原因作因果归因，也不构成 ETF 买卖指令。",
            )
            _replace_section(
                r"(?ms)^###\s*二、(?:指数结构|指数点评|主要指数)\s*\n.*?(?=^###\s|\Z)",
                "### 二、指数结构",
                index_body + "\n\n以下为同一报告数据源提供的最新可用指数快照。",
                preserve_table=True,
            )
            _replace_section(
                r"(?ms)^###\s*四、(?:后市观察|展望)\s*\n.*?(?=^###\s|\Z)",
                "### 四、后市观察",
                "- 对照 SPX、NDX、IXIC 和 DJI 在同一完整交易日的方向，观察指数间分化是否持续。\n- 观察 VIX 后续变化；单一时点读数不能单独代表风险趋势。\n- 宏观与新闻只引用带来源和日期的信息，不将其直接归因于指数涨跌。",
            )

        # Fallback templates may contain breadth/liquidity and sector sections.
        # Replace them with what the US snapshot actually supports.
        top_sector_names = [
            str(item.get("name", "")).strip()
            for item in (getattr(overview, "top_sectors", None) or [])[:5]
            if isinstance(item, dict) and item.get("name")
        ]
        bottom_sector_names = [
            str(item.get("name", "")).strip()
            for item in (getattr(overview, "bottom_sectors", None) or [])[:5]
            if isinstance(item, dict) and item.get("name")
        ]
        top_concept_names = [
            str(item.get("name", "")).strip()
            for item in (getattr(overview, "top_concepts", None) or [])[:5]
            if isinstance(item, dict) and item.get("name")
        ]
        bottom_concept_names = [
            str(item.get("name", "")).strip()
            for item in (getattr(overview, "bottom_concepts", None) or [])[:5]
            if isinstance(item, dict) and item.get("name")
        ]
        if english:
            if top_sector_names or bottom_sector_names or top_concept_names or bottom_concept_names:
                sector_body = (
                    f"Industry leaders: {', '.join(top_sector_names) or 'not supplied'}. "
                    f"Industry laggards: {', '.join(bottom_sector_names) or 'not supplied'}. "
                    f"Concept leaders: {', '.join(top_concept_names) or 'not supplied'}. "
                    f"Concept laggards: {', '.join(bottom_concept_names) or 'not supplied'}."
                )
            else:
                sector_body = "No verifiable US sector or industry ranking data was supplied; sector leadership and laggards cannot be assessed."
            funds_body = (
                "The supplied US index snapshot does not include comparable market breadth, consolidated turnover, "
                "or fund-flow data. Net inflows/outflows and market participation cannot be assessed. VIX is a volatility measure, not a fund-flow measure."
            )
            _replace_section(
                r"(?ms)^###\s*3\.\s*Breadth & Liquidity\s*\n.*?(?=^###\s|\Z)",
                "### 3. Breadth & Liquidity",
                "Market breadth, consolidated turnover, and fund flows are unavailable in the supplied data; no conclusion is made.",
            )
            _replace_section(
                r"(?ms)^###\s*4\.\s*Sector / Theme Highlights\s*\n.*?(?=^###\s|\Z)",
                "### 4. Sector / Theme Highlights",
                sector_body,
            )
            _replace_section(
                r"(?ms)^###\s*4\.\s*Funds & Sentiment\s*\n.*?(?=^###\s|\Z)",
                "### 4. Funds & Sentiment",
                funds_body,
            )
        else:
            if top_sector_names or bottom_sector_names or top_concept_names or bottom_concept_names:
                sector_body = (
                    f"行业领涨线索：{'、'.join(top_sector_names) or '未提供'}；"
                    f"行业落后线索：{'、'.join(bottom_sector_names) or '未提供'}；"
                    f"概念领涨线索：{'、'.join(top_concept_names) or '未提供'}；"
                    f"概念落后线索：{'、'.join(bottom_concept_names) or '未提供'}。"
                )
            else:
                sector_body = "本次未提供可核验的美股行业/板块涨跌榜，无法判断行业强弱或轮动方向。"
            funds_body = (
                "本次美股指数输入未提供可比的市场涨跌家数、成交额汇总或资金流向数据，"
                "无法判断市场宽度及资金净流入/流出。VIX 反映波动率，不等同于资金流向。"
            )
            _replace_section(
                r"(?ms)^###\s*三、(?:板块主线|热点解读|板块表现)\s*\n.*?(?=^###\s|\Z)",
                "### 三、板块主线",
                sector_body,
            )
            _replace_section(
                r"(?ms)^###\s*四、(?:资金与情绪|资金动向)\s*\n.*?(?=^###\s|\Z)",
                "### 四、资金与情绪",
                funds_body,
            )

        # Do not let generated prose invent or causally attribute headlines.
        items = []
        for item in (news or [])[:8]:
            normalized = self._normalize_news_item(item)
            title = normalized.get("title", "").strip()
            if not title:
                continue
            meta = " / ".join(
                value for value in (normalized.get("source", "").strip(), normalized.get("published_date", "").strip())
                if value
            ) or ("source/date not supplied" if english else "来源/日期未提供")
            snippet = normalized.get("snippet", "").strip()
            url = normalized.get("url", "").strip()
            line = f"- **{title}** ({meta})"
            if snippet:
                line += f"\n  {snippet}"
            if url:
                line += f"\n  [Source]({url})" if english else f"\n  [来源]({url})"
            items.append(line)

        if english:
            news_body = (
                "The following are retrieved news leads only. They are not proof of causation for index moves.\n\n"
                + "\n\n".join(items)
                if items
                else "No attributable news items with usable headlines were supplied; no event attribution is made."
            )
            strategy_body = (
                "**Market stance:** Describe risk-on, neutral, or risk-off conditions qualitatively from the supplied index and volatility data.\n\n"
                "**Watchpoints:** Monitor completed-session index direction, VIX, sourced macro releases, and data freshness.\n\n"
                "No portfolio percentages, entries, exits, stop-losses, price targets, or short-term trading triggers."
            )
            data_note = (
                "### Data Boundaries\n"
                "The supplied US index dataset does not establish volume, turnover, PE/valuation, market breadth, or capital flows unless those fields are explicitly present. Missing values remain unknown. News claims must be traceable to the dated source items above."
            )
            news_pattern = r"(?ms)^###\s*(?:3\.\s*News Catalysts|5\.\s*News Catalysts|V\.\s*News Catalysts)\s*\n.*?(?=^###\s|\Z)"
            strategy_pattern = r"(?ms)^###\s*(?:6\.\s*Strategy Plan|VI\.\s*Strategy Framework)\s*\n.*?(?=^###\s|\Z)"
            news_heading = "### 3. News Catalysts"
            strategy_heading = "### 6. Strategy Plan"
        else:
            news_body = (
                "以下仅列出实际检索到的新闻线索，不代表这些事件已被证实为指数涨跌原因。\n\n"
                + "\n\n".join(items)
                if items
                else "未提供具备可用标题的新闻来源，本报告不作事件归因。"
            )
            strategy_body = (
                "**市场状态：** 仅根据已提供的指数方向与波动率，定性描述积极、中性或防守状态。\n\n"
                "**观察重点：** 关注完整交易日的指数方向、VIX、具备来源的宏观数据及行情时效性。\n\n"
                "不提供组合仓位比例、精确买卖点、止损位、目标价或短线交易触发点。"
            )
            data_note = (
                "### 数据边界\n"
                "除非输入明确提供对应字段，否则不能判断成交量、成交额、PE/估值、市场宽度或资金流。缺失数据代表未知。新闻结论必须能追溯到上方带日期和来源的检索结果。"
            )
            news_pattern = r"(?ms)^###\s*(?:三、|五、)(?:消息催化|后市展望)\s*\n.*?(?=^###\s|\Z)"
            strategy_pattern = r"(?ms)^###\s*六、(?:策略判断|策略框架)\s*\n.*?(?=^###\s|\Z)"
            news_heading = "### 三、消息催化"
            strategy_heading = "### 六、策略判断"

        news_match = re.search(news_pattern, text)
        if news_match:
            news_heading = news_match.group(0).splitlines()[0].strip()
            text = re.sub(news_pattern, news_heading + "\n\n" + news_body + "\n\n", text, count=1)
        else:
            news_heading = "### Retrieved News" if english else "### 检索到的新闻"
            strategy_match = re.search(strategy_pattern, text)
            insert_at = strategy_match.start() if strategy_match else len(text)
            news_section = news_heading + "\n\n" + news_body + "\n\n"
            text = text[:insert_at] + news_section + text[insert_at:]
        strategy_match = re.search(strategy_pattern, text)
        if strategy_match:
            # Preserve the report's existing localized heading for compatibility.
            strategy_heading = strategy_match.group(0).splitlines()[0].strip()
            text = re.sub(
                strategy_pattern,
                strategy_heading + "\n\n" + strategy_body + "\n\n",
                text,
                count=1,
            )
        # Remove numeric claims that cannot be traced to the supplied index
        # snapshot or retrieved news. Keep markdown tables/headings/URLs intact.
        source_material = [str(getattr(overview, "date", "") or "")]
        for index_item in (getattr(overview, "indices", None) or []):
            source_material.append(str(getattr(index_item, "name", "") or ""))
            for field_name in (
                "current", "change", "change_pct", "open", "high", "low",
                "prev_close", "amplitude",
            ):
                value = getattr(index_item, field_name, None)
                if value is not None:
                    try:
                        numeric_value = float(value)
                        if numeric_value != 0.0:
                            source_material.append(str(value))
                            if field_name in {"change", "change_pct"}:
                                source_material.append(str(abs(numeric_value)))
                    except (TypeError, ValueError):
                        continue
        for item in (news or [])[:8]:
            normalized = self._normalize_news_item(item)
            source_material.extend(
                normalized.get(field_name, "")
                for field_name in ("title", "snippet", "source", "published_date")
            )

        allowed_numbers = set()
        number_pattern = r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?%?"
        for source_text in source_material:
            for token in re.findall(number_pattern, source_text or ""):
                try:
                    number = float(token.replace(",", "").replace("%", "").replace("+", ""))
                except ValueError:
                    continue
                for precision in (1, 2, 3):
                    allowed_numbers.add(round(number, precision))
                if number.is_integer():
                    allowed_numbers.add(number)

        redacted_numeric_lines = 0
        safe_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.startswith("|")
                or "http://" in stripped
                or "https://" in stripped
            ):
                safe_lines.append(line)
                continue
            content_line = re.sub(r"^\s*(?:[-*]\s*)?\d+[.)]\s+", "", line)
            unsupported_number = False
            for token in re.findall(number_pattern, content_line):
                try:
                    number = float(token.replace(",", "").replace("%", "").replace("+", ""))
                except ValueError:
                    continue
                supported = any(
                    abs(number - round(number, precision)) < 1e-9
                    and round(number, precision) in allowed_numbers
                    for precision in (1, 2, 3)
                )
                if number.is_integer() and number in allowed_numbers:
                    supported = True
                if not supported:
                    unsupported_number = True
                    break
            if unsupported_number:
                redacted_numeric_lines += 1
                continue
            safe_lines.append(line)
        text = "\n".join(safe_lines)

        if redacted_numeric_lines:
            redaction_note = (
                "Unsupported numeric claims not found in the supplied market data or retrieved news were omitted."
                if english
                else "未能在行情输入或检索新闻中找到依据的具体数字，已从正文中省略。"
            )
            data_note += "\n" + redaction_note
        if "Data Boundaries" not in text and "数据边界" not in text:
            text = text.rstrip() + "\n\n" + data_note + "\n"
        elif redacted_numeric_lines:
            text = text.rstrip() + "\n\n" + redaction_note + "\n"
        return text

    def _get_analyzer_generation_backend_config_error(self) -> Optional[GenerationError]:
        """Return analyzer backend config errors without relying on dynamic mock attributes."""
        if self.analyzer is None:
            try:
                resolve_generation_backend_id(self.config)
                resolve_generation_fallback_backend_id(self.config)
            except GenerationError as exc:
                return exc
            return None
        missing = object()
        if getattr_static(self.analyzer, "get_generation_backend_config_error", missing) is missing:
            return None
        method = getattr(self.analyzer, "get_generation_backend_config_error", None)
        if not callable(method):
            return None
        error = method()
        return error if isinstance(error, GenerationError) else None

    def _get_configured_generation_backend_identity(self) -> tuple[str, str]:
        """Best-effort backend identity for legacy analyzers without metadata APIs."""
        backend_id = str(getattr(self.config, "generation_backend", "") or "").strip().lower()
        if not backend_id:
            backend_id = LITELLM_BACKEND_ID
        if backend_id in LOCAL_CLI_GENERATION_BACKEND_IDS:
            return backend_id, backend_id
        return backend_id, str(getattr(self.config, "litellm_model", "") or "")

    def _get_legacy_analyzer_generation_backend_identity(self) -> tuple[str, str]:
        """Return a neutral identity for injected analyzers that expose no metadata APIs."""
        return _LEGACY_ANALYZER_BACKEND_ID, _LEGACY_ANALYZER_BACKEND_ID

    def _get_analyzer_generation_backend_identity(self) -> tuple[str, str]:
        """Use analyzer metadata API when available, otherwise fall back to configured identity."""
        if self.analyzer is None:
            return self._get_configured_generation_backend_identity()
        missing = object()
        if getattr_static(self.analyzer, "get_generation_backend_identity", missing) is not missing:
            method = getattr(self.analyzer, "get_generation_backend_identity", None)
            if callable(method):
                return method()
        return self._get_legacy_analyzer_generation_backend_identity()

    def _generate_market_review_with_metadata(
        self,
        prompt: str,
        *,
        provider: str,
        model: str,
    ) -> Optional[GenerationResult]:
        """Support legacy analyzers that only implement the public generate_text() contract."""
        if self.analyzer is None:
            return None
        missing = object()
        if getattr_static(self.analyzer, "generate_text_with_metadata", missing) is not missing:
            method = getattr(self.analyzer, "generate_text_with_metadata", None)
            if callable(method):
                return method(
                    prompt,
                    max_tokens=8192,
                    temperature=0.7,
                )

        if getattr_static(self.analyzer, "generate_text", missing) is missing:
            raise AttributeError(
                "analyzer must implement generate_text_with_metadata() or generate_text()"
            )
        legacy_method = getattr(self.analyzer, "generate_text", None)
        if not callable(legacy_method):
            raise AttributeError(
                "analyzer must implement generate_text_with_metadata() or generate_text()"
            )
        review = legacy_method(
            prompt,
            max_tokens=8192,
            temperature=0.7,
        )
        if review is None:
            return None
        return GenerationResult(
            text=review,
            provider=provider,
            model=model,
            backend=provider,
            usage={},
        )

    def build_market_review_payload(
        self,
        overview: MarketOverview,
        news: List,
        report: str,
        market_light_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the structured market-review contract consumed by API, Web, and notifications."""
        language = self._get_output_language()
        sections = self._split_report_sections(report)
        title = self._extract_report_title(report) or self._get_review_title(overview.date).lstrip("# ").strip()
        light = (
            market_light_snapshot or self.build_market_light_snapshot(overview)
            if self._supports_market_light()
            else None
        )
        breadth_dimensions = None
        if isinstance(light, dict):
            dimensions = light.get("dimensions")
            if isinstance(dimensions, dict):
                breadth_dimensions = dimensions.get("breadth")

        breadth_supported = bool(self.profile.has_market_stats)
        if breadth_supported and isinstance(breadth_dimensions, dict) and "available" in breadth_dimensions:
            breadth_supported = bool(breadth_dimensions.get("available"))

        has_breadth_data = False
        if breadth_supported:
            if isinstance(breadth_dimensions, dict) and "available" in breadth_dimensions:
                has_breadth_data = bool(breadth_dimensions.get("available"))
            else:
                breadth_available = overview.up_count + overview.down_count + overview.flat_count > 0
                limit_available = overview.limit_up_count + overview.limit_down_count > 0
                has_breadth_data = bool(breadth_available or limit_available)

        payload = {
            "version": 1,
            "kind": "market_review",
            "region": self.region,
            "language": language,
            "color_scheme": getattr(
                getattr(self, "config", None),
                "market_review_color_scheme",
                "green_up",
            ),
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "date": overview.date,
            "market_scope": self._get_market_scope_name(language),
            "indices": [idx.to_dict() for idx in overview.indices],
            "sectors": {
                "top": list(overview.top_sectors or []),
                "bottom": list(overview.bottom_sectors or []),
            },
            "concepts": {
                "top": list(overview.top_concepts or []),
                "bottom": list(overview.bottom_concepts or []),
            },
            "news": [self._normalize_news_item(item) for item in (news or [])[:8]],
            "sections": sections,
            "markdown_report": report,
        }

        if light is not None:
            payload["market_light"] = light

        if has_breadth_data:
            payload["breadth"] = {
                "up_count": overview.up_count,
                "down_count": overview.down_count,
                "flat_count": overview.flat_count,
                "limit_up_count": overview.limit_up_count,
                "limit_down_count": overview.limit_down_count,
                "total_amount": overview.total_amount,
                "turnover_unit": self._get_turnover_unit_label(),
            }

        return payload

    def _supports_market_light(self) -> bool:
        return self.region in MARKET_LIGHT_REGIONS

    @staticmethod
    def _extract_report_title(report: str) -> str:
        for line in (report or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return ""

    @classmethod
    def _split_report_sections(cls, report: str) -> List[Dict[str, str]]:
        text = (report or "").strip()
        if not text:
            return []
        matches = list(re.finditer(r"^(#{2,3})\s+(.+?)\s*$", text, flags=re.MULTILINE))
        if not matches:
            return [{"key": "full_review", "title": "Review", "markdown": text}]

        sections: List[Dict[str, str]] = []
        first_match = matches[0]
        starts_with_report_title = first_match.start() == 0 and first_match.group(1) == "##"
        content_start_index = 1 if starts_with_report_title else 0
        intro_start = first_match.end() if starts_with_report_title else 0
        intro_end = (
            matches[1].start()
            if starts_with_report_title and len(matches) > 1
            else (len(text) if starts_with_report_title else matches[0].start())
        )
        intro = text[intro_start:intro_end].strip()
        if intro:
            sections.append({"key": "overview", "title": "Overview", "markdown": intro})

        for index, match in enumerate(matches[content_start_index:], start=content_start_index):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            title = match.group(2).strip()
            markdown = text[start:end].strip()
            if not markdown:
                continue
            key = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", title).strip("_").lower()
            sections.append({
                "key": key or f"section_{index + 1}",
                "title": title,
                "markdown": markdown,
            })
        return sections

    @classmethod
    def _normalize_news_item(cls, item: Any) -> Dict[str, str]:
        return {
            "title": cls._compact_news_text(cls._get_news_field(item, "title"), limit=120),
            "snippet": cls._compact_news_text(cls._get_news_field(item, "snippet"), limit=260),
            "source": cls._compact_news_text(cls._get_news_field(item, "source"), limit=80),
            "published_date": cls._compact_news_text(cls._get_news_field(item, "published_date"), limit=40),
            "url": cls._compact_news_text(cls._get_news_field(item, "url"), limit=240),
        }
    
    def _inject_data_into_review(
        self,
        review: str,
        overview: MarketOverview,
        news: Optional[List] = None,
    ) -> str:
        """Inject structured data tables into the corresponding LLM prose sections."""
        # Build data blocks
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)
        patterns = (
            _ENGLISH_SECTION_PATTERNS
            if self._get_review_language() == "en"
            else _CHINESE_SECTION_PATTERNS
        )

        if stats_block:
            review = self._insert_after_section(
                review,
                patterns["market_summary"],
                stats_block,
            )

        if indices_block:
            review = self._insert_after_section(
                review,
                patterns["index_commentary"],
                indices_block,
            )

        if sector_block:
            original_review = review
            review = self._insert_after_section(
                review,
                patterns["sector_highlights"],
                sector_block,
            )
            if review == original_review and sector_block not in review:
                fallback_heading = (
                    "### 4. Sector Highlights"
                    if self._get_review_language() == "en"
                    else "### 三、板块主线"
                )
                review = f"{review.rstrip()}\n\n{fallback_heading}\n{sector_block}\n"

        return review

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        """Insert a data block at the end of a markdown section (before the next ### heading)."""
        import re
        # Find the heading
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        # Find the next ### heading after this one
        next_heading = re.search(r'\n###\s', text[start:])
        if next_heading:
            insert_pos = start + next_heading.start()
        else:
            # No next heading — append at end
            insert_pos = len(text)
        # Insert the block before the next heading, with spacing
        return text[:insert_pos].rstrip() + '\n\n' + block + '\n\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        """Build market statistics block."""
        has_stats = overview.up_count or overview.down_count or overview.total_amount
        has_market_signal = bool(self._supports_market_light() and (overview.indices or has_stats))
        if not has_stats and not has_market_signal:
            return ""
        light = self.build_market_light_snapshot(overview) if has_market_signal else None
        if self._get_review_language() == "en":
            lines = []
            if isinstance(light, dict):
                lines.extend(
                    [
                        f"- **Market Signal**: {light['score']}/100 "
                        f"({light['temperature_label']}, {light['label']})",
                        f"- **Drivers**: {'; '.join(light['reasons'])}",
                        f"- **Guidance**: {light['guidance']}",
                    ]
                )
            if has_stats:
                if lines:
                    lines.append("")
                lines.append(
                    f"- **Breadth**: Advancers {overview.up_count} / Decliners {overview.down_count} / "
                    f"Flat {overview.flat_count}; "
                    f"Limit-up {overview.limit_up_count} / Limit-down {overview.limit_down_count}; "
                    f"Turnover {overview.total_amount:.0f} ({self._get_turnover_unit_label()})"
                )
            return "\n".join(lines)
        lines = []
        score = light["score"] if isinstance(light, dict) else None
        label = light["temperature_label"] if isinstance(light, dict) else ""
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else 0.0
        limit_spread = overview.limit_up_count - overview.limit_down_count
        if isinstance(light, dict) and score is not None:
            lines.extend(
                [
                    f"- **盘面信号**：{score}/100（{label}，{light['label']}）",
                    f"- **信号依据**：{'；'.join(light['reasons'])}",
                    f"- **操作建议**：{light['guidance']}",
                ]
            )
        if has_stats:
            if lines:
                lines.append("")
            lines.extend(
                [
                    "| 指标 | 数值 | 观察 |",
                    "|------|------|------|",
                    f"| 上涨/下跌/平盘 | {overview.up_count} / {overview.down_count} / {overview.flat_count} | 上涨占比(不含平盘) {up_ratio:.1%} |",
                    f"| 涨停/跌停 | {overview.limit_up_count} / {overview.limit_down_count} | 涨跌停差 {limit_spread:+d} |",
                    f"| 两市成交额 | {overview.total_amount:.0f} 亿 | {self._describe_turnover(overview.total_amount)} |",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _get_directional_index_changes(overview: MarketOverview) -> List[float]:
        """Exclude VIX volatility returns from the directional index average."""
        return [
            idx.change_pct
            for idx in overview.indices
            if idx.change_pct is not None
            and str(idx.code or "").strip().upper() not in {"VIX", "^VIX"}
        ]

    def build_market_light_snapshot(self, overview: MarketOverview) -> Dict[str, Any]:
        """Build a deterministic market-light snapshot from structured breadth data."""
        scores = self._build_market_light_scores(overview)
        score = int(scores["score"])
        temperature_label = str(scores["temperature_label"])
        if score >= 60:
            status = "green"
        elif score >= 40:
            status = "yellow"
        else:
            status = "red"

        if self._get_review_language() == "en":
            label_map = {
                "green": "risk-on",
                "yellow": "balanced",
                "red": "risk-off",
            }
            guidance_map = {
                "green": "Risk appetite is acceptable; focus on leading themes and position discipline.",
                "yellow": "Signals are mixed; keep position sizing moderate and wait for confirmation.",
                "red": "Risk is elevated; prioritize drawdown control and avoid chasing weak rebounds.",
            }
            reasons = self._build_market_light_reasons_en(overview, score)
        else:
            label_map = {
                "green": "可进攻",
                "yellow": "需观察",
                "red": "偏防守",
            }
            guidance_map = {
                "green": "风险偏好尚可，关注主线延续与仓位纪律。",
                "yellow": "信号分化，控制仓位并等待量价确认。",
                "red": "风险偏高，优先控制回撤，避免追高弱反弹。",
            }
            reasons = self._build_market_light_reasons_zh(overview, score)

        snapshot = MarketLightSnapshot(
            region=self.region,
            trade_date=overview.date,
            status=status,
            label=label_map[status],
            score=score,
            temperature_label=temperature_label,
            reasons=reasons,
            guidance=guidance_map[status],
            dimensions=scores["dimensions"],
            data_quality=str(scores["data_quality"]),
        )
        return snapshot.model_dump()

    def _build_market_light_reasons_zh(self, overview: MarketOverview, score: int) -> List[str]:
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = []
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，赚钱效应扩散")
            elif up_ratio <= 0.4:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，亏钱效应较强")
            else:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，市场分化")
        index_changes = self._get_directional_index_changes(overview)
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            reasons.append(f"主要指数平均涨跌幅 {avg_change:+.2f}%")
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"涨跌停差 {overview.limit_up_count - overview.limit_down_count:+d}")
        if not reasons and overview.total_amount:
            reasons.append(f"成交额 {overview.total_amount:.0f} 亿，{self._describe_turnover(overview.total_amount)}")
        if not reasons:
            reasons.append("结构化涨跌数据有限，按可用行情综合判断")
        return reasons[:4]

    def _build_market_light_reasons_en(self, overview: MarketOverview, score: int) -> List[str]:
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = []
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is expanding")
            elif up_ratio <= 0.4:
                reasons.append(f"advancers ratio {up_ratio:.0%}, downside pressure dominates")
            else:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is mixed")
        index_changes = self._get_directional_index_changes(overview)
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            reasons.append(f"average major-index change {avg_change:+.2f}%")
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"limit-up/down spread {overview.limit_up_count - overview.limit_down_count:+d}")
        if not reasons and overview.total_amount:
            reasons.append(f"turnover {overview.total_amount:.0f} ({self._get_turnover_unit_label()})")
        if not reasons:
            reasons.append("limited structured breadth data; using available market inputs")
        return reasons[:4]

    def _build_indices_block(self, overview: MarketOverview) -> str:
        """构建指数行情表格"""
        if not overview.indices:
            return ""
        if self._get_review_language() == "en":
            lines = [
                f"| Index | Last | Change % | Open | High | Low | Amplitude | Turnover ({self._get_turnover_unit_label()}) |",
                "|-------|------|----------|------|------|-----|-----------|-----------------|",
            ]
        else:
            lines = [
                "| 指数 | 最新 | 涨跌幅 | 开盘 | 最高 | 最低 | 振幅 | 成交额(亿) |",
                "|------|------|--------|------|------|------|------|-----------|",
            ]
        for idx in overview.indices:
            arrow = self._get_index_change_arrow(idx.change_pct)
            amount_raw = idx.amount or 0.0
            amount_str = self._format_turnover_value(amount_raw)
            lines.append(
                f"| {idx.name} | {idx.current:.2f} | {arrow} {idx.change_pct:+.2f}% | "
                f"{self._format_optional_number(idx.open)} | {self._format_optional_number(idx.high)} | "
                f"{self._format_optional_number(idx.low)} | {self._format_optional_pct(idx.amplitude)} | {amount_str} |"
            )
        return "\n".join(lines)

    def _build_sector_block(self, overview: MarketOverview) -> str:
        """Build industry and concept ranking blocks."""
        if (
            not overview.top_sectors
            and not overview.bottom_sectors
            and not overview.top_concepts
            and not overview.bottom_concepts
        ):
            return ""
        lines = []
        language = self._get_review_language()

        def append_ranking(title: str, name_label: str, rows: List[Dict]) -> None:
            if not rows:
                return
            if lines:
                lines.append("")
            lines.extend([
                title,
                f"| {'Rank' if language == 'en' else '排名'} | {name_label} | {'Change' if language == 'en' else '涨跌幅'} |",
                "|------|------|--------|",
            ])
            for rank, item in enumerate(rows[:5], 1):
                lines.append(
                    f"| {rank} | {item.get('name', '-')} | {self._format_signed_pct(item.get('change_pct'))} |"
                )

        if language == "en":
            append_ranking("#### Leading Industry Sectors", "Sector", overview.top_sectors)
            append_ranking("#### Lagging Industry Sectors", "Sector", overview.bottom_sectors)
            append_ranking("#### Leading Concept Themes", "Concept", overview.top_concepts)
            append_ranking("#### Lagging Concept Themes", "Concept", overview.bottom_concepts)
        else:
            append_ranking("#### 行业板块领涨 Top 5", "行业板块", overview.top_sectors)
            append_ranking("#### 行业板块领跌 Top 5", "行业板块", overview.bottom_sectors)
            append_ranking("#### 概念板块领涨 Top 5", "概念板块", overview.top_concepts)
            append_ranking("#### 概念板块领跌 Top 5", "概念板块", overview.bottom_concepts)
        return "\n".join(lines)

    def _build_news_block(self, news: List) -> str:
        """Build a compact source-aware news catalyst list for the rendered report."""
        if not news:
            return ""
        language = self._get_review_language()
        if language == "en":
            lines = [
                "#### News Catalysts",
            ]
        else:
            lines = [
                "#### 近三日市场线索",
            ]

        for idx, item in enumerate(news[:5], 1):
            lines.append(self._format_news_catalyst_line(idx, item, language=language))
        return "\n".join(lines)

    @staticmethod
    def _get_news_field(item: Any, field: str) -> str:
        if hasattr(item, field):
            value = getattr(item, field, "") or ""
        elif isinstance(item, dict):
            value = item.get(field, "") or ""
        else:
            value = ""
        return str(value).strip()

    @classmethod
    def _format_news_catalyst_line(cls, idx: int, item: Any, *, language: str = "zh") -> str:
        fallback_title = "Untitled catalyst" if language == "en" else "未命名线索"
        title = cls._compact_news_text(cls._get_news_field(item, "title"), limit=90) or fallback_title
        source = cls._compact_news_text(cls._get_news_field(item, "source"), limit=40)
        date_text = cls._compact_news_text(cls._get_news_field(item, "published_date"), limit=24)
        url = cls._compact_news_text(cls._get_news_field(item, "url"), limit=0)
        title_text = cls._escape_markdown_link_label(title)
        if url:
            title_text = f"[{title_text}]({url})"
        meta_parts = [part for part in (source, date_text) if part]
        if language == "en":
            meta = f" ({' / '.join(meta_parts)})" if meta_parts else ""
        else:
            meta = f"（{' / '.join(meta_parts)}）" if meta_parts else ""
        return f"- {idx}. {title_text}{meta}"

    @staticmethod
    def _compact_news_text(value: str, *, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if limit <= 0 or len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _format_optional_number(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}"

    @staticmethod
    def _format_optional_pct(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}%"

    @staticmethod
    def _format_signed_pct(value: Any) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{numeric_value:+.2f}%"

    @classmethod
    def _format_ranking_summary(cls, rows: List[Dict], limit: int = 3) -> str:
        parts = []
        for item in (rows or [])[:limit]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            parts.append(f"{name}({cls._format_signed_pct(item.get('change_pct'))})")
        return ", ".join(parts)

    @staticmethod
    def _escape_markdown_link_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")

    @staticmethod
    def _describe_turnover(total_amount: float) -> str:
        if total_amount >= 15000:
            return "高活跃度"
        if total_amount >= 9000:
            return "中等活跃"
        if total_amount > 0:
            return "缩量观望"
        return "暂无数据"

    def _build_market_light_scores(self, overview: MarketOverview) -> Dict[str, Any]:
        """Build the canonical Market Light scores used by reports and alerts."""

        participants = overview.up_count + overview.down_count
        breadth_available = bool(self.profile.has_market_stats and participants > 0)
        breadth_score = 50
        if breadth_available:
            breadth_score = int(overview.up_count / participants * 100)

        index_changes = self._get_directional_index_changes(overview)
        index_available = bool(index_changes)
        index_score = 50
        if index_available:
            avg_change = sum(index_changes) / len(index_changes)
            index_score = int(max(0, min(100, 50 + avg_change * 12)))

        limit_total = overview.limit_up_count + overview.limit_down_count
        limit_available = bool(self.profile.has_market_stats and limit_total > 0)
        limit_score = 50
        if limit_available:
            limit_score = int(overview.limit_up_count / limit_total * 100)

        dimensions = {
            "breadth": {"score": breadth_score, "available": breadth_available},
            "index": {"score": index_score, "available": index_available},
            "limit": {"score": limit_score, "available": limit_available},
        }

        if not index_available:
            data_quality = "unavailable"
        elif all(dimension["available"] for dimension in dimensions.values()):
            data_quality = "ok"
        else:
            data_quality = "partial"

        score = int(round(breadth_score * 0.45 + index_score * 0.35 + limit_score * 0.20))
        if self._get_review_language() == "en":
            if score >= 70:
                label = "risk-on"
            elif score >= 55:
                label = "constructive"
            elif score >= 40:
                label = "mixed"
            else:
                label = "defensive"
        else:
            if score >= 70:
                label = "强势"
            elif score >= 55:
                label = "偏暖"
            elif score >= 40:
                label = "震荡"
            else:
                label = "偏弱"
        return {
            "score": score,
            "temperature_label": label,
            "dimensions": dimensions,
            "data_quality": data_quality,
        }

    def _build_market_temperature(self, overview: MarketOverview) -> tuple[int, str]:
        scores = self._build_market_light_scores(overview)
        score = int(scores["score"])
        label = str(scores["temperature_label"])
        return score, label

    def _build_output_template_sections(self, review_language: str) -> str:
        """Build LLM output sections according to market data capabilities."""
        if self.region == "us":
            if review_language == "en":
                return """### 3. News Catalysts
(Connect only dated, attributable news to the supplied index moves. Separate facts from interpretation.)

### 4. Outlook
(Describe the near-term market regime and the evidence that could change it.)

### 5. Risk Alerts
(List material risks and clearly state what data is unavailable.)

### 6. Strategy Plan
(Give a qualitative risk-on / neutral / risk-off assessment and key watchpoints only. Do not provide portfolio position percentages, precise entry/exit levels, stop-losses, price targets, or trading triggers.)"""
            return """### 三、消息催化
（只把有明确来源和日期的新闻与已提供指数走势联系起来，区分事实与推断。）

### 四、后市观察
（概括市场风险状态及可能改变判断的证据。）

### 五、风险提示
（列出主要风险，并明确说明缺失的数据。）

### 六、策略判断
（只给出定性的积极/中性/防守判断和观察重点。不得提供组合仓位百分比、精确买卖点、止损位、目标价或短线交易触发点。）"""

        if review_language == "en":
            if self.profile.has_market_stats and self.profile.has_sector_rankings:
                return """### 3. Fund Flows
(Interpret what turnover, participation, and flow signals imply.)

### 4. Sector Highlights
(Distinguish industry-sector moves from concept/theme moves, then analyze drivers and persistence.)

### 5. Outlook
(Provide the near-term outlook based on price action and news.)

### 6. Risk Alerts
(List the main risks to monitor.)

### 7. Strategy Plan
(Provide an offensive/balanced/defensive stance, a position-sizing guideline, one invalidation trigger, and end with "For reference only, not investment advice.")"""

            section_number = 3
            sections: List[str] = []
            if self.profile.has_market_stats:
                sections.append(f"""### {section_number}. Fund Flows
(Interpret only the provided turnover, participation, breadth, and flow signals.)""")
                section_number += 1
            if self.profile.has_sector_rankings:
                sections.append(f"""### {section_number}. Sector Highlights
(Analyze only the provided industry-sector and concept/theme rankings.)""")
                section_number += 1
            sections.extend([
                f"""### {section_number}. News Catalysts
(Connect recent news to index price action and macro/external-market clues. Do not infer unsupported breadth, fund-flow, or sector-ranking data.)""",
                f"""### {section_number + 1}. Outlook
(Provide the near-term outlook based on index price action and the available news.)""",
                f"""### {section_number + 2}. Risk Alerts
(List the main risks to monitor.)""",
                f"""### {section_number + 3}. Strategy Plan
(Provide an offensive/balanced/defensive stance, a position-sizing guideline, one invalidation trigger, and end with "For reference only, not investment advice.")""",
            ])
            return "\n\n".join(sections)

        if self.profile.has_market_stats and self.profile.has_sector_rankings:
            return """### 三、板块主线
（区分行业板块与概念题材，分析领涨/领跌背后的逻辑、持续性和是否形成主线）

### 四、资金与情绪
（解读成交额、涨跌停结构、市场宽度和风险偏好）

### 五、消息催化
（结合近三日新闻，提炼真正影响明日交易的催化或扰动）

### 六、明日交易计划
（给出进攻/均衡/防守结论、仓位区间、关注方向、回避方向和一个触发失效条件）

### 七、风险提示
（列出需要关注的风险点；最后补充“建议仅供参考，不构成投资建议”。）"""

        numerals = ["一", "二", "三", "四", "五", "六", "七", "八"]
        section_number = 3
        sections: List[str] = []

        def add_section(title: str, hint: str) -> None:
            nonlocal section_number
            sections.append(f"### {numerals[section_number - 1]}、{title}\n{hint}")
            section_number += 1

        if self.profile.has_sector_rankings:
            add_section("板块主线", "（仅分析已提供的行业板块与概念题材榜单，不扩展未提供的数据）")
        if self.profile.has_market_stats:
            add_section("资金与情绪", "（仅解读已提供的成交额、涨跌停结构、市场宽度和风险偏好数据）")
        add_section(
            "消息催化",
            "（结合近三日新闻和指数表现，提炼真正影响明日交易的催化或扰动；不要推断未提供的资金流、市场宽度或板块榜）",
        )
        add_section("明日交易计划", "（给出进攻/均衡/防守结论、仓位区间、关注方向、回避方向和一个触发失效条件）")
        add_section("风险提示", "（列出需要关注的风险点；最后补充“建议仅供参考，不构成投资建议”。）")
        return "\n\n".join(sections)

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """构建复盘报告 Prompt"""
        review_language = self._get_review_language()
        # Korean reuses the English structural template but the model is told to
        # write the entire shell, headings, guidance and conclusion in Korean.
        shell_language_label = "Korean (한국어)" if self._get_output_language() == "ko" else "English"

        # 指数行情信息（简洁格式，不用emoji）
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        top_sectors_text = self._format_ranking_summary(overview.top_sectors)
        bottom_sectors_text = self._format_ranking_summary(overview.bottom_sectors)
        top_concepts_text = self._format_ranking_summary(overview.top_concepts)
        bottom_concepts_text = self._format_ranking_summary(overview.bottom_concepts)
        
        # 新闻信息 - 支持 SearchResult 对象或字典
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            # 兼容 SearchResult 对象和字典
            title = self._compact_news_text(self._get_news_field(n, "title"), limit=90)
            snippet = self._compact_news_text(self._get_news_field(n, "snippet"), limit=220)
            source = self._compact_news_text(self._get_news_field(n, "source"), limit=60)
            published_date = self._compact_news_text(self._get_news_field(n, "published_date"), limit=30)
            url = self._compact_news_text(self._get_news_field(n, "url"), limit=180)
            meta_parts = [part for part in (source, published_date) if part]
            meta = f" ({' / '.join(meta_parts)})" if meta_parts else ""
            url_line = f"\n   URL: {url}" if url else ""
            news_text += f"{i}. {title}{meta}\n   {snippet or '-'}{url_line}\n"
        
        # 按 region 组装市场概况与板块区块（美股/港股/日韩无涨跌家数、板块数据）
        stats_block = ""
        sector_block = ""
        data_limits_block = ""
        if review_language == "en":
            if self.profile.has_market_stats:
                stats_block = f"""## Market Breadth
- Advancers: {overview.up_count} | Decliners: {overview.down_count} | Flat: {overview.flat_count}
- Limit-up: {overview.limit_up_count} | Limit-down: {overview.limit_down_count}
- Turnover: {overview.total_amount:.0f} ({self._get_turnover_unit_label()})"""

            if self.profile.has_sector_rankings:
                sector_block = f"""## Sector / Theme Performance
Industry leading: {top_sectors_text if top_sectors_text else "N/A"}
Industry lagging: {bottom_sectors_text if bottom_sectors_text else "N/A"}
Concept leading: {top_concepts_text if top_concepts_text else "N/A"}
Concept lagging: {bottom_concepts_text if bottom_concepts_text else "N/A"}"""

            data_limit_lines = []
            if not self.profile.has_market_stats:
                data_limit_lines.append(
                    "- Market breadth, aggregate turnover, participation, and fund-flow signals are not available for this market."
                )
            if not self.profile.has_sector_rankings:
                data_limit_lines.append("- Sector/theme ranking data is not available for this market.")
            if data_limit_lines:
                data_limits_block = "## Data Limits\n" + "\n".join(data_limit_lines)
        else:
            if self.profile.has_market_stats:
                stats_block = f"""## 市场概况
- 上涨: {overview.up_count} 家 | 下跌: {overview.down_count} 家 | 平盘: {overview.flat_count} 家
- 涨停: {overview.limit_up_count} 家 | 跌停: {overview.limit_down_count} 家
- 两市成交额: {overview.total_amount:.0f} 亿元"""

            if self.profile.has_sector_rankings:
                sector_block = f"""## 板块表现
行业领涨: {top_sectors_text if top_sectors_text else "暂无数据"}
行业领跌: {bottom_sectors_text if bottom_sectors_text else "暂无数据"}
概念领涨: {top_concepts_text if top_concepts_text else "暂无数据"}
概念领跌: {bottom_concepts_text if bottom_concepts_text else "暂无数据"}"""

            data_limit_lines = []
            if not self.profile.has_market_stats:
                data_limit_lines.append("- 该市场暂无涨跌家数、涨跌停、成交额汇总、参与度或资金流信号。")
            if not self.profile.has_sector_rankings:
                data_limit_lines.append("- 该市场暂无行业板块/概念题材涨跌榜。")
            if data_limit_lines:
                data_limits_block = "## 数据边界\n" + "\n".join(data_limit_lines)

        data_no_indices_hint = (
            "注意：由于行情数据获取失败，请主要根据【市场新闻】进行定性分析和总结，不要编造具体的指数点位。"
            if not indices_text
            else ""
        )
        if review_language == "en":
            data_no_indices_hint = (
                "Note: Market data fetch failed. Rely mainly on [Market News] for qualitative analysis. Do not invent index levels."
                if not indices_text
                else ""
            )
            indices_placeholder = indices_text if indices_text else "No index data (API error)"
            news_placeholder = news_text if news_text else "No relevant news"
            data_boundary_requirement = (
                "- Respect Data Limits: do not invent or over-interpret unsupported breadth, fund-flow, turnover, participation, or sector-ranking data.\n"
                if data_limits_block
                else ""
            )
            if self.region == "us":
                data_limit_lines.append(
                    "- The supplied index table contains only the displayed index levels and daily changes. "
                    "Volume/turnover, valuation, P/E, forward P/E and market breadth are unavailable unless "
                    "explicitly provided as separate fields."
                )
                data_limit_lines.append(
                    "- Never infer volume contraction/expansion, valuation level, or fund flows from missing data. "
                    "Missing values mean unknown, not zero."
                )
                data_limits_block = "## Data Limits\n" + "\n".join(data_limit_lines)
            data_boundary_requirement = (
                "- Evidence rule: do not claim volume, valuation, P/E, forward P/E, market breadth, or fund-flow signals "
                "unless the exact metric is explicitly supplied with a usable source/date. Missing means unknown, not zero. "
                "Do not invent any numeric macro value, analyst target, index level, or trading statistic absent from the supplied data. "
                "Do not claim that a news item caused an index move unless the supplied source explicitly establishes that relationship. "
                "Treat all quote values as the latest available values, not automatically as the report-date close. "
                "Do not provide portfolio sizing, entries, exits, stop-losses, price targets, or short-term trading triggers.\n"
                if self.region == "us" else ""
            )
            data_boundary_requirement += (
                "- Respect Data Limits: do not invent or over-interpret unsupported breadth, fund-flow, turnover, participation, or sector-ranking data.\n"
                if data_limits_block and review_language == "en" else ""
            )
            market_summary_hint = (
                "2-3 sentences summarizing overall market tone, index moves, and liquidity."
                if self.profile.has_market_stats
                else "2-3 sentences summarizing overall market tone, index moves, and available news context."
            )
        else:
            indices_placeholder = indices_text if indices_text else "暂无指数数据（接口异常）"
            news_placeholder = news_text if news_text else "暂无相关新闻"
            data_boundary_requirement = (
                "- 严格遵守数据边界：未提供涨跌家数、资金流、成交额汇总或板块榜时，不要编造或过度解读。\n"
                if data_limits_block
                else ""
            )
            if self.region == "us":
                data_limit_lines.append(
                    "- 当前指数表只提供明确列出的指数点位和日涨跌幅；除非输入中另有明确字段，"
                    "否则成交量、成交额、估值、市盈率 PE、远期 PE 和市场宽度均不可用。"
                )
                data_limit_lines.append(
                    "- 不得根据缺失数据推断放量/缩量、估值高低或资金流向；缺失值代表未知，不代表零。"
                )
                data_limits_block = "## 数据边界\\n" + "\\n".join(data_limit_lines)
            data_boundary_requirement = (
                "- 证据规则：除非明确提供了带可靠来源和日期的具体指标，否则不得判断成交量、估值、"
                "PE、远期PE或资金流。不得基于不完整/降级数据编造精确点位或仓位比例。\\n"
                if self.region == "us" else ""
            )
            data_boundary_requirement += (
                "- 严格遵守数据边界：未提供涨跌家数、资金流、成交额汇总或板块榜时，不要编造或过度解读。\\n"
                if data_limits_block and review_language != "en" else ""
            )
            market_summary_hint = (
                "2-3句话概括指数、涨跌家数、成交额和情绪温度，明确“强势/偏暖/震荡/偏弱”判断"
                if self.profile.has_market_stats
                else "2-3句话概括指数表现、新闻线索和整体风险状态，不要补写未提供的市场宽度或资金流数据"
            )

        output_template_sections = self._build_output_template_sections(review_language)
        zh_market_scope_name = self._get_market_scope_name("zh")
        zh_report_title = f"{overview.date} 大盘复盘"
        if self.region in ("jp", "kr"):
            zh_report_title = f"{overview.date} {zh_market_scope_name}大盘复盘"
        workflow_hint = (
            "报告要像交易员盘后工作台：先给结论，再按数据表、主线、催化、计划展开"
            if self.profile.has_market_stats or self.profile.has_sector_rankings
            else "报告要像交易员盘后工作台：先给结论，再按指数、新闻催化和计划展开"
        )

        if review_language == "en":
            report_title = self._get_review_title(overview.date).removeprefix("## ").strip()
            return f"""You are a professional {self._get_market_scope_name('en')} analyst. Please produce a concise market recap report based on the data below.

[Requirements]
- Output pure Markdown only
- No JSON
- No code blocks
- Use emoji sparingly in headings (at most one per heading)
- The entire fixed shell, headings, guidance, and conclusion must be in {shell_language_label}
{data_boundary_requirement}

---

# Today's Market Data

## Date
{overview.date}

## Major Indices
{indices_placeholder}

{stats_block}

{sector_block}

{data_limits_block}

## Market News
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# Output Template (follow this structure)

## {report_title}

### 1. Market Summary
({market_summary_hint})

### 2. Index Commentary
({self._get_index_hint()})

{output_template_sections}

---

Output the report content directly, no extra commentary.
"""

        # A 股场景使用中文提示语
        return f"""你是一位专业的{self._get_market_scope_name('zh')}分析师，请根据以下数据生成一份结构化的{self._get_market_scope_name('zh')}大盘复盘报告。

【重要】输出要求：
- 必须输出纯 Markdown 文本格式
- 禁止输出 JSON 格式
- 禁止输出代码块
- emoji 仅在标题处少量使用（每个标题最多1个）
- {workflow_hint}
- 不要重复列出已由系统注入的表格数据；正文负责解释表格背后的含义
{data_boundary_requirement}

---

# 今日市场数据

## 日期
{overview.date}

## 主要指数
{indices_placeholder}

{stats_block}

{sector_block}

{data_limits_block}

## 市场新闻
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# 输出格式模板（请严格按此格式输出）

## {zh_report_title}

> 一句话给出今日市场状态、核心矛盾和明日优先观察方向。

### 一、盘面总览
（{market_summary_hint}）

### 二、指数结构
（{self._get_index_hint()}，说明谁在护盘、谁在拖累，以及关键支撑/压力）

{output_template_sections}

---

请直接输出复盘报告内容，不要输出其他说明文字。
"""
    
    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """使用模板生成复盘报告（无大模型时的备选方案）"""
        template_language = self._get_template_review_language()
        mood_code = self.profile.mood_index_code
        # 根据 mood_index_code 查找对应指数
        # cn: mood_code="000001"，idx.code 可能为 "sh000001"（以 mood_code 结尾）
        # us: mood_code="SPX"，idx.code 直接为 "SPX"
        mood_index = next(
            (
                idx
                for idx in overview.indices
                if idx.code == mood_code or idx.code.endswith(mood_code)
            ),
            None,
        )
        if mood_index:
            if mood_index.change_pct > 1:
                market_mood = self._get_market_mood_text("strong_up", template_language)
            elif mood_index.change_pct > 0:
                market_mood = self._get_market_mood_text("mild_up", template_language)
            elif mood_index.change_pct > -1:
                market_mood = self._get_market_mood_text("mild_down", template_language)
            else:
                market_mood = self._get_market_mood_text("strong_down", template_language)
        else:
            market_mood = self._get_market_mood_text("range", template_language)
        
        # 指数行情（简洁格式）
        indices_text = ""
        for idx in overview.indices[:4]:
            marker = self._get_index_change_arrow(idx.change_pct)
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({marker} {idx.change_pct:+.2f}%)\n"
        
        # 板块信息
        separator = ", " if template_language == "en" else "、"
        top_text = separator.join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = separator.join([s['name'] for s in overview.bottom_sectors[:3]])
        top_concept_text = separator.join([s['name'] for s in overview.top_concepts[:3]])
        bottom_concept_text = separator.join([s['name'] for s in overview.bottom_concepts[:3]])

        if template_language == "en":
            stats_section = ""
            if self._supports_market_light() or self.profile.has_market_stats:
                stats_block = self._build_stats_block(overview)
                if stats_block:
                    stats_section = f"""
### 3. Breadth & Liquidity
{stats_block}
"""
            sector_section = ""
            if self.profile.has_sector_rankings and (top_text or bottom_text or top_concept_text or bottom_concept_text):
                sector_section = f"""
### 4. Sector / Theme Highlights
- **Industry Leaders**: {top_text or "N/A"}
- **Industry Laggards**: {bottom_text or "N/A"}
- **Concept Leaders**: {top_concept_text or "N/A"}
- **Concept Laggards**: {bottom_concept_text or "N/A"}
"""
            market_names = {
                "us": "US Market Recap",
                "hk": "HK Market Recap",
                "jp": "Japan Market Recap",
                "kr": "Korea Market Recap",
            }
            market_name = market_names.get(self.region, "A-share Market Recap")
            report = f"""## {overview.date} {market_name}

### 1. Market Summary
Today's {self._get_market_scope_name(template_language)} showed **{market_mood}**.

### 2. Major Indices
{indices_text or "- No index data available"}
{stats_section}
{sector_section}
### 5. Risk Alerts
Market conditions can change quickly. The data above is for reference only and does not constitute investment advice.

{self._get_strategy_markdown_block(template_language)}

---
*Review Time: {datetime.now().strftime('%H:%M')}*
"""
            return report

        market_labels = {"cn": "A股", "us": "美股", "hk": "港股", "jp": "日股", "kr": "韩股"}
        market_label = market_labels.get(self.region, "A股")
        dashboard_block = (
            self._build_stats_block(overview)
            if self._supports_market_light() or self.profile.has_market_stats
            else ""
        )
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview) if self.profile.has_sector_rankings else ""
        summary_focus = (
            "指数承接、成交额变化和板块持续性"
            if self.profile.has_market_stats and self.profile.has_sector_rankings
            else "指数承接、消息催化和整体风险状态"
        )
        market_summary_block = (
            dashboard_block
            if dashboard_block
            else (
                "暂无市场宽度数据。"
                if self.profile.has_market_stats
                else "- 当前以主要指数与可用新闻线索评估整体风险状态。"
            )
        )
        sector_section = (
            f"""
### 三、板块主线
{sector_block or "- 暂无板块涨跌榜数据。"}
"""
            if self.profile.has_sector_rankings
            else ""
        )
        funds_section = (
            """
### 四、资金与情绪
- 结合成交额和涨跌家数看，当前更适合等待确认，避免仅凭单一热点追高。
"""
            if self.profile.has_market_stats
            else ""
        )
        return f"""## {overview.date} 大盘复盘

> 今日{market_label}市场整体呈现**{market_mood}**态势，优先观察{summary_focus}。

### 一、盘面总览
{market_summary_block}

### 二、指数结构
{indices_block or indices_text or "暂无指数数据。"}
{sector_section}
{funds_section}

### 五、消息催化
- 暂无可用新闻时，应降低对题材持续性的确定性判断。

{self._get_strategy_markdown_block(template_language)}

### 七、风险提示
- 市场有风险，投资需谨慎。以上数据仅供参考，不构成投资建议。

---
*复盘时间: {datetime.now().strftime('%H:%M')}*
"""
    
    def _run_daily_review_parts(self) -> MarketLightReviewResult:
        """Run market review once and keep report/snapshot on the same overview."""
        logger.info("========== 开始大盘复盘分析 ==========")

        # 1. 获取市场概览
        overview = self.get_market_overview()

        # 2. 搜索市场新闻
        news = self.search_market_news()
        news = self._merge_persisted_market_intelligence(news)

        # 3. 生成复盘报告
        report = self.generate_market_review(overview, news)
        snapshot = self.build_market_light_snapshot(overview) if self._supports_market_light() else None
        structured_payload = self.build_market_review_payload(
            overview,
            news,
            report,
            snapshot,
        )

        logger.info("========== 大盘复盘分析完成 ==========")

        return MarketLightReviewResult(
            overview=overview,
            report=report,
            market_light_snapshot=snapshot,
            structured_payload=structured_payload,
        )

    def _merge_persisted_market_intelligence(self, news: List) -> List:
        """Merge local persisted market intelligence and search news with bounded prompt/payload slot preservation."""
        search_news = list(news or [])
        merged_local = []
        seen_urls = {
            self._get_news_field(item, "url")
            for item in search_news
            if self._get_news_field(item, "url")
        }
        try:
            service = IntelligenceService(config=self.config)
            service.refresh_auto_sources()
            payload = service.list_items(
                scope_type="market",
                market=self.region,
                published_days=max(1, int(self.config.get_effective_news_window_days() or 1)),
                page=1,
                page_size=6,
            )
            for item in payload.get("items", []):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "")
                if url and url in seen_urls:
                    continue
                seen_urls.add(url)
                merged_local.append({
                    "title": item.get("title") or "未命名资讯",
                    "snippet": item.get("summary") or "",
                    "source": item.get("source") or item.get("source_name") or "local-intel",
                    "published_date": item.get("published_at") or "",
                    "url": "" if url.startswith("no-url:intel:") else url,
                })
        except Exception as exc:
            logger.debug("[大盘] %s action=load_local_intelligence status=failed error=%s", self._log_context(), exc)
        merged_news = []
        merged_local_index = 0
        merged_search_index = 0
        while merged_local_index < len(merged_local) or merged_search_index < len(search_news):
            if merged_local_index < len(merged_local):
                merged_news.append(merged_local[merged_local_index])
                merged_local_index += 1
            if merged_search_index < len(search_news):
                merged_news.append(search_news[merged_search_index])
                merged_search_index += 1
        return merged_news

    def run_daily_review(self) -> str:
        """
        执行每日大盘复盘流程

        Returns:
            复盘报告文本
        """
        return self.run_daily_review_with_snapshot().report

    def run_daily_review_with_snapshot(self) -> MarketLightReviewResult:
        """Run daily review and return the report plus its structured Market Light snapshot."""
        return self._run_daily_review_parts()


# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    analyzer = MarketAnalyzer()
    
    # 测试获取市场概览
    overview = analyzer.get_market_overview()
    print(f"\n=== 市场概览 ===")
    print(f"日期: {overview.date}")
    print(f"指数数量: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"上涨: {overview.up_count} | 下跌: {overview.down_count}")
    print(f"成交额: {overview.total_amount:.0f}亿")
    
    # 测试生成模板报告
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== 复盘报告 ===")
    print(report)
