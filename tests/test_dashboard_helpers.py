from __future__ import annotations

import pandas as pd

from alpha_rotation.backtest import BacktestConfig, backtest_engine
from alpha_rotation.dashboard_helpers import build_research_artifacts, prepare_download_tables


def make_panel() -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=320, freq="B")
    rows = []
    for i, date in enumerate(dates):
        for ticker, bias in [("AAA", 0.0), ("BBB", 1.0), ("CCC", 2.0)]:
            price = 100 + i + bias
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "Open": price,
                    "High": price + 1,
                    "Low": price - 1,
                    "Close": price,
                    "Volume": 1000 + i,
                }
            )
    return pd.DataFrame(rows)


def make_snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"effective_date": pd.Timestamp("2023-01-01"), "ticker": "AAA"},
            {"effective_date": pd.Timestamp("2023-01-01"), "ticker": "BBB"},
            {"effective_date": pd.Timestamp("2023-01-01"), "ticker": "CCC"},
        ]
    )


def test_build_research_artifacts_includes_custom_factor_diagnostics():
    artifacts = build_research_artifacts(
        panel=make_panel(),
        custom_factors={"spread": "Close - Open"},
        lookback_periods=5,
    )
    assert not artifacts.diagnostics.empty
    assert any(result.name == "spread" and result.is_valid for result in artifacts.custom_factor_results)
    assert "spread" in artifacts.factor_panel.columns


def test_prepare_download_tables_returns_expected_frames():
    artifacts = build_research_artifacts(panel=make_panel(), lookback_periods=5)
    result = backtest_engine(
        panel=artifacts.factor_panel,
        snapshots=make_snapshots(),
        config=BacktestConfig(
            factor_cols=["mom_1m"],
            rebalance_freq=5,
            long_quantile=0.5,
            commission_bps=0.0,
        ),
    )
    tables = prepare_download_tables(diagnostics=artifacts.diagnostics, result=result)
    assert {"factor_diagnostics", "performance", "rebalance_log", "holdings_history"}.issubset(tables.keys())


def test_build_research_artifacts_respects_diagnostic_range():
    artifacts = build_research_artifacts(
        panel=make_panel(),
        lookback_periods=5,
        diagnostic_start="2024-01-01",
        diagnostic_end="2024-03-31",
    )
    assert not artifacts.diagnostics.empty
    dates = pd.to_datetime(artifacts.diagnostics["selection_date"])
    assert dates.min() >= pd.Timestamp("2024-01-01")
    assert dates.max() <= pd.Timestamp("2024-03-31")
