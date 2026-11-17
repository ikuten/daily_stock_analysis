# -*- coding: utf-8 -*-
"""US index quote source identity tests."""

import sys
from types import SimpleNamespace

import pandas as pd
from unittest.mock import MagicMock, patch

# Optional dependency used by some data-provider adapters.
sys.modules.setdefault("fake_useragent", MagicMock())

from data_provider.realtime_types import RealtimeSource
from data_provider.yfinance_fetcher import YfinanceFetcher


def test_yfinance_is_reported_as_primary_source_for_us_index_quotes():
    ticker = MagicMock()
    ticker.fast_info = SimpleNamespace(
        lastPrice=28000.0,
        previousClose=27900.0,
        open=27950.0,
        dayHigh=28050.0,
        dayLow=27920.0,
        lastVolume=123456,
    )
    ticker.info = {"currency": "USD"}
    yf = MagicMock()
    yf.Ticker.return_value = ticker

    with patch.dict(sys.modules, {"yfinance": yf}):
        quote = YfinanceFetcher().get_realtime_quote("NDX")

    assert quote is not None
    assert quote.market == "us"
    assert quote.source is RealtimeSource.YFINANCE
    assert quote.source.value == "yfinance"
    assert quote.fallback_from is None
    assert quote.price == 28000.0



def test_yfinance_daily_index_bar_exposes_its_actual_quote_date():
    ticker = MagicMock()
    ticker.history.return_value = pd.DataFrame(
        {
            "Open": [27900.0, 28000.0],
            "High": [28100.0, 28200.0],
            "Low": [27800.0, 27900.0],
            "Close": [27950.0, 28050.0],
            "Volume": [1000.0, 1200.0],
        },
        index=pd.to_datetime(["2026-11-12", "2026-11-13"]),
    )
    yf = MagicMock()
    yf.Ticker.return_value = ticker

    item = YfinanceFetcher()._fetch_yf_ticker_data(yf, "^NDX", "纳斯达克100指数", "NDX")

    assert item is not None
    assert item["quote_date"] == "2026-11-13"
