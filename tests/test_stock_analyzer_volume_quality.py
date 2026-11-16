# -*- coding: utf-8 -*-
"""Regression tests for missing/invalid volume handling."""

import numpy as np
import pandas as pd
import pytest

from src.stock_analyzer import StockTrendAnalyzer, TrendAnalysisResult, VolumeStatus


@pytest.mark.parametrize("volumes", [
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0, 0, 0, 0, 0, 0],
    [100, 100, 100, 100, np.nan, 20],
])
def test_missing_or_invalid_volume_is_not_classified_as_shrink(volumes):
    frame = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=6),
        "close": [100, 101, 100, 102, 101, 99],
        "volume": volumes,
    })
    result = TrendAnalysisResult(code="NDX")

    StockTrendAnalyzer()._analyze_volume(frame, result)

    assert result.volume_status == VolumeStatus.NORMAL
    assert result.volume_ratio_5d == 0.0
    assert result.volume_data_available is False
    assert "未纳入量能判断" in result.volume_trend


def test_valid_volume_computes_ratio_against_previous_five_bars():
    frame = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=6),
        "close": [100, 101, 100, 102, 101, 99],
        "volume": [100, 100, 100, 100, 100, 50],
    })
    result = TrendAnalysisResult(code="NDX")

    StockTrendAnalyzer()._analyze_volume(frame, result)

    assert result.volume_data_available is True
    assert result.volume_ratio_5d == pytest.approx(0.5)
    assert result.volume_status == VolumeStatus.SHRINK_VOLUME_DOWN
