from __future__ import annotations

import pandas as pd

from alpha_rotation.backtest import BacktestConfig, backtest_engine
from alpha_rotation.reporting import (
    rebalance_detail,
    summarize_factor_diagnostics,
    summarize_rebalances,
)


def make_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    rows = []
    for i, date in enumerate(dates):
        for ticker, factor in [("AAA", 2.0), ("BBB", 1.0)]:
            price = 100 + i + (1 if ticker == "BBB" else 0)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "Open": price,
                    "High": price + 1,
                    "Low": price - 1,
                    "Close": price,
                    "Volume": 1000,
                    "ret_1d": 0.001,
                    "fwd_ret_5d": 0.01 if ticker == "AAA" else 0.005,
                    "factor_a": factor + i * 0.001,
                }
            )
    return pd.DataFrame(rows)


def test_reporting_handles_rebalance_outputs():
    snapshots = pd.DataFrame(
        [
            {"effective_date": "2024-01-01", "ticker": "AAA"},
            {"effective_date": "2024-01-01", "ticker": "BBB"},
        ]
    ).assign(effective_date=lambda x: pd.to_datetime(x["effective_date"]))

    result = backtest_engine(
        panel=make_panel(),
        snapshots=snapshots,
        config=BacktestConfig(
            factor_cols=["factor_a"],
            rebalance_freq=5,
            long_quantile=0.5,
            commission_bps=5.0,
        ),
    )

    summary = summarize_rebalances(result)
    detail = rebalance_detail(result, result.rebalance_log.index[0])

    assert not summary.empty
    assert "turnover" in summary.columns
    assert not detail.empty
    assert set(["ticker", "old_weight", "new_weight", "weight_change", "action"]).issubset(detail.columns)


def test_summarize_factor_diagnostics_filters_by_date():
    diagnostics = pd.DataFrame(
        [
            {
                "selection_date": pd.Timestamp("2024-02-01"),
                "factor": "factor_a",
                "mean_ic": 0.1,
                "ic_std": 0.05,
                "icir": 2.0,
                "recommended_sign": "original",
                "recommended_factor_name": "factor_a",
                "original_candidate": "factor_a",
                "inverse_candidate": "inv_factor_a",
            },
            {
                "selection_date": pd.Timestamp("2024-03-01"),
                "factor": "factor_b",
                "mean_ic": -0.1,
                "ic_std": 0.05,
                "icir": -2.0,
                "recommended_sign": "inverse",
                "recommended_factor_name": "inv_factor_b",
                "original_candidate": "factor_b",
                "inverse_candidate": "inv_factor_b",
            },
        ]
    )

    summary = summarize_factor_diagnostics(diagnostics, "2024-03-01")
    assert len(summary) == 1
    assert summary.iloc[0]["recommended_factor_name"] == "inv_factor_b"
