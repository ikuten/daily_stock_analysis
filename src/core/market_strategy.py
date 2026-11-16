# -*- coding: utf-8 -*-
"""Market strategy blueprints for CN/HK/US daily market recap."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class StrategyDimension:
    """Single strategy dimension used by market recap prompts."""

    name: str
    objective: str
    checkpoints: List[str]


@dataclass(frozen=True)
class MarketStrategyBlueprint:
    """Region specific market strategy blueprint."""

    region: str
    title: str
    positioning: str
    principles: List[str]
    dimensions: List[StrategyDimension]
    action_framework: List[str]

    def to_prompt_block(self) -> str:
        """Render blueprint as prompt instructions."""
        principles_text = "\n".join([f"- {item}" for item in self.principles])
        action_text = "\n".join([f"- {item}" for item in self.action_framework])

        dims = []
        for dim in self.dimensions:
            checkpoints = "\n".join([f"  - {cp}" for cp in dim.checkpoints])
            dims.append(f"- {dim.name}: {dim.objective}\n{checkpoints}")
        dimensions_text = "\n".join(dims)

        return (
            f"## Strategy Blueprint: {self.title}\n"
            f"{self.positioning}\n\n"
            f"### Strategy Principles\n{principles_text}\n\n"
            f"### Analysis Dimensions\n{dimensions_text}\n\n"
            f"### Action Framework\n{action_text}"
        )

    def to_markdown_block(self) -> str:
        """Render blueprint as markdown section for template fallback report."""
        dims = "\n".join([f"- **{dim.name}**: {dim.objective}" for dim in self.dimensions])
        section_title = "### VI. Strategy Framework" if self.region == "us" else "### 六、策略框架"
        return f"{section_title}\n{dims}\n"


CN_BLUEPRINT = MarketStrategyBlueprint(
    region="cn",
    title="A股市场三段式复盘策略",
    positioning="聚焦指数趋势、资金博弈与板块轮动，形成次日交易计划。",
    principles=[
        "先看指数方向，再看量能结构，最后看板块持续性。",
        "结论必须映射到仓位、节奏与风险控制动作。",
        "判断使用当日数据与近3日新闻，不臆测未验证信息。",
    ],
    dimensions=[
        StrategyDimension(
            name="趋势结构",
            objective="判断市场处于上升、震荡还是防守阶段。",
            checkpoints=["上证/深证/创业板是否同向", "放量上涨或缩量下跌是否成立", "关键支撑阻力是否被突破"],
        ),
        StrategyDimension(
            name="资金情绪",
            objective="识别短线风险偏好与情绪温度。",
            checkpoints=["涨跌家数与涨跌停结构", "成交额是否扩张", "高位股是否出现分歧"],
        ),
        StrategyDimension(
            name="主线板块",
            objective="提炼可交易主线与规避方向。",
            checkpoints=["领涨板块是否具备事件催化", "板块内部是否有龙头带动", "领跌板块是否扩散"],
        ),
    ],
    action_framework=[
        "进攻：指数共振上行 + 成交额放大 + 主线强化。",
        "均衡：指数分化或缩量震荡，控制仓位并等待确认。",
        "防守：指数转弱 + 领跌扩散，优先风控与减仓。",
    ],
)

US_BLUEPRINT = MarketStrategyBlueprint(
    region="us",
    title="US Market Regime Strategy",
    positioning="Describe the broad US equity risk regime from verified index, volatility, and macro data; this is a market observation, not a trading plan.",
    principles=[
        "Read S&P 500, Nasdaq-100, Nasdaq Composite, and Dow separately; never substitute one index for another.",
        "Separate observed data from interpretation and cite the supplied source/date for material news.",
        "Do not infer volume, valuation, breadth, or capital flows when the corresponding source fields are absent.",
        "Do not provide portfolio sizing, entries, exits, stop-losses, price targets, or short-term trading triggers.",
    ],
    dimensions=[
        StrategyDimension(
            name="Trend Regime",
            objective="Classify the broad market as advancing, range-bound, mixed, or defensive using supplied index closes.",
            checkpoints=[
                "Are SPX, NDX, IXIC, and DJI directionally aligned",
                "Is VIX rising or falling based on supplied values",
                "Are index moves consistent across the same completed trading session",
            ],
        ),
        StrategyDimension(
            name="Macro Context",
            objective="Summarize only sourced policy, rates, inflation, and currency information.",
            checkpoints=[
                "Treasury yield and USD data with explicit source/date",
                "Fed communication and scheduled macro releases",
                "Distinguish confirmed facts from possible market implications",
            ],
        ),
        StrategyDimension(
            name="Sector Themes",
            objective="Describe sourced sector and earnings themes without asserting unsupported index causality.",
            checkpoints=[
                "AI/semiconductor/software news with identifiable source and date",
                "Energy/financials sensitivity only when supported by supplied evidence",
                "Volatility context from VIX and dated market data",
            ],
        ),
    ],
    action_framework=[
        "Risk-on: broad indices advance together and the supplied evidence supports improving risk appetite.",
        "Neutral: index signals are mixed or material evidence is unavailable; describe what remains unknown.",
        "Risk-off: broad indices weaken and supplied volatility/macro evidence confirms elevated risk.",
    ],
)

HK_BLUEPRINT = MarketStrategyBlueprint(
    region="hk",
    title="港股市场三段式复盘策略",
    positioning="聚焦恒生指数趋势、南向资金博弈与板块轮动，形成次日交易计划。",
    principles=[
        "先看恒指/恒科/国企指数方向，再看南向资金情绪，最后看板块持续性。",
        "结论必须映射到仓位、节奏与风险控制动作。",
        "判断使用当日数据与近3日新闻，不臆测未验证信息。",
    ],
    dimensions=[
        StrategyDimension(
            name="趋势结构",
            objective="判断市场处于上升、震荡还是防守阶段。",
            checkpoints=["恒指/恒科/国企指数是否同向", "放量上涨或缩量下跌是否成立", "关键支撑阻力是否被突破"],
        ),
        StrategyDimension(
            name="资金情绪",
            objective="识别南向资金风险偏好与情绪温度。",
            checkpoints=["南向资金净流入方向与规模", "港元汇率与内地政策含义", "市场广度与龙头集中度"],
        ),
        StrategyDimension(
            name="主线板块",
            objective="提炼可交易主线与规避方向。",
            checkpoints=["科技/互联网平台趋势持续性", "金融/地产对政策转向的敏感度", "防御与成长因子轮动"],
        ),
    ],
    action_framework=[
        "进攻：恒指共振上行 + 南向资金持续流入 + 主线强化。",
        "均衡：指数分化或缩量震荡，控制仓位并等待确认。",
        "防守：指数转弱 + 波动率上升，优先风控与减仓。",
    ],
)


JP_BLUEPRINT = MarketStrategyBlueprint(
    region="jp",
    title="日本市场三段式复盘策略",
    positioning="聚焦日经225、东证指数、汇率与全球风险偏好，形成次日交易计划。",
    principles=[
        "先看日经225与TOPIX是否同向，再看日元、半导体/出口链与金融股表现。",
        "把指数结论映射到仓位、节奏与风险控制动作。",
        "只基于可得指数、新闻和价格行为判断，不臆造市场广度或板块统计。",
    ],
    dimensions=[
        StrategyDimension(
            name="趋势结构",
            objective="判断日本市场处于上攻、震荡还是防守阶段。",
            checkpoints=["日经225/TOPIX是否同向", "指数是否突破或跌破关键区间", "大盘权重与成长链是否共振"],
        ),
        StrategyDimension(
            name="宏观与汇率",
            objective="识别日元、利率和全球风险偏好对权益市场的影响。",
            checkpoints=["日元方向对出口链的影响", "日本央行和美债利率叙事", "海外科技股与半导体链映射"],
        ),
        StrategyDimension(
            name="主题线索",
            objective="提炼可延续主线与需要规避的拥挤方向。",
            checkpoints=["半导体/自动化/汽车链持续性", "金融与内需股是否轮动", "新闻催化是否支撑价格行为"],
        ),
    ],
    action_framework=[
        "进攻：主要指数共振上行 + 外部风险偏好改善 + 主线强化。",
        "均衡：指数分化或汇率扰动，降低追涨并等待确认。",
        "防守：主要指数转弱或外部风险升温，优先控制仓位。",
    ],
)

KR_BLUEPRINT = MarketStrategyBlueprint(
    region="kr",
    title="韩国市场三段式复盘策略",
    positioning="聚焦 KOSPI、KOSDAQ、半导体权重与全球科技风险偏好，形成次日交易计划。",
    principles=[
        "先看 KOSPI/KOSDAQ 是否同向，再看三星电子、SK 海力士等权重线索。",
        "区分指数 beta、半导体周期和成长股风险偏好的贡献。",
        "只基于可得指数、新闻和价格行为判断，不臆造市场广度或板块统计。",
    ],
    dimensions=[
        StrategyDimension(
            name="趋势结构",
            objective="判断韩国市场处于上攻、震荡还是防守阶段。",
            checkpoints=["KOSPI/KOSDAQ 是否同向", "权重股是否支撑指数", "关键支撑阻力是否被突破"],
        ),
        StrategyDimension(
            name="科技周期",
            objective="识别半导体、AI 硬件和全球科技股对韩国市场的映射。",
            checkpoints=["存储/半导体链新闻催化", "美股科技方向联动", "外资风险偏好变化"],
        ),
        StrategyDimension(
            name="主题线索",
            objective="提炼可延续主线与需要规避的拥挤方向。",
            checkpoints=["电池/汽车/互联网是否轮动", "KOSDAQ 成长股风险偏好", "新闻催化是否支撑价格行为"],
        ),
    ],
    action_framework=[
        "进攻：KOSPI/KOSDAQ 共振上行 + 科技权重确认 + 外部风险偏好改善。",
        "均衡：指数或权重股分化，控制仓位并等待确认。",
        "防守：科技权重转弱或外部风险升温，优先控制回撤。",
    ],
)

def get_market_strategy_blueprint(region: str) -> MarketStrategyBlueprint:
    """Return strategy blueprint by market region."""
    if region == "us":
        return US_BLUEPRINT
    if region == "hk":
        return HK_BLUEPRINT
    if region == "jp":
        return JP_BLUEPRINT
    if region == "kr":
        return KR_BLUEPRINT
    return CN_BLUEPRINT
