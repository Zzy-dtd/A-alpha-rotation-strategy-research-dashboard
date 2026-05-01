from __future__ import annotations

import pandas as pd

from alpha_rotation.backtest import (
    BacktestConfig,
    backtest_engine,
    calculate_turnover,
    monthly_factor_diagnostics,
    prepare_clean_panel,
)


def make_test_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    rows = []
    tickers = ["AAA", "BBB", "CCC"]
    for i, date in enumerate(dates):
        for j, ticker in enumerate(tickers):
            base = 100 + (j * 5) + i
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "Open": base,
                    "High": base + 1,
                    "Low": base - 1,
                    "Close": base,
                    "Volume": 1_000_000 + j * 1000,
                    "ret_1d": 0.001 * (j + 1),
                    "fwd_ret_5d": 0.01 * (j + 1),
                    "factor_a": 3 - j + (i * 0.001),
                    "factor_b": j + (i * 0.001),
                }
            )
    return pd.DataFrame(rows)


def make_snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"effective_date": "2024-01-01", "ticker": "AAA"},
            {"effective_date": "2024-01-01", "ticker": "BBB"},
            {"effective_date": "2024-01-01", "ticker": "CCC"},
            {"effective_date": "2024-02-01", "ticker": "AAA"},
            {"effective_date": "2024-02-01", "ticker": "BBB"},
        ]
    )


def test_calculate_turnover_handles_weight_changes():
    previous_positions = pd.Series({"AAA": 0.5, "BBB": 0.5})
    target_positions = pd.Series({"AAA": 1.0})
    turnover = calculate_turnover(previous_positions, target_positions)
    assert turnover == 0.5


def test_backtest_commission_reduces_net_returns():
    panel = make_test_panel()
    snapshots = make_snapshots()

    gross_result = backtest_engine(
        panel=panel,
        snapshots=snapshots.assign(effective_date=pd.to_datetime(snapshots["effective_date"])),
        config=BacktestConfig(
            factor_cols=["factor_a", "factor_b"],
            rebalance_freq=5,
            long_quantile=0.5,
            commission_bps=0.0,
        ),
    )
    cost_result = backtest_engine(
        panel=panel,
        snapshots=snapshots.assign(effective_date=pd.to_datetime(snapshots["effective_date"])),
        config=BacktestConfig(
            factor_cols=["factor_a", "factor_b"],
            rebalance_freq=5,
            long_quantile=0.5,
            commission_bps=25.0,
        ),
    )

    assert gross_result.performance["gross_ret"].equals(cost_result.performance["gross_ret"])
    assert cost_result.performance["daily_ret"].sum() < gross_result.performance["daily_ret"].sum()
    assert cost_result.rebalance_log["commission_cost"].sum() > 0


def test_rebalance_log_reconciles_with_trade_log():
    panel = make_test_panel()
    snapshots = make_snapshots().assign(effective_date=lambda x: pd.to_datetime(x["effective_date"]))
    result = backtest_engine(
        panel=panel,
        snapshots=snapshots,
        config=BacktestConfig(
            factor_cols=["factor_a", "factor_b"],
            rebalance_freq=5,
            long_quantile=0.5,
            commission_bps=10.0,
        ),
    )

    first_rebalance = result.rebalance_log.iloc[0]
    trade_rows = result.rebalance_trades[result.rebalance_trades["date"] == result.rebalance_log.index[0]]

    assert sorted(first_rebalance["changed"]) == sorted(trade_rows.loc[trade_rows["weight_change"] != 0, "ticker"].tolist())
    assert abs(first_rebalance["turnover"] - result.performance.loc[result.rebalance_log.index[0], "turnover"]) < 1e-12


def test_backtest_accepts_inverted_factor_selection():
    panel = make_test_panel()
    snapshots = make_snapshots().assign(effective_date=lambda x: pd.to_datetime(x["effective_date"]))

    original = backtest_engine(
        panel=panel,
        snapshots=snapshots,
        config=BacktestConfig(
            factor_cols=["factor_a"],
            rebalance_freq=5,
            long_quantile=0.5,
            commission_bps=0.0,
        ),
    )
    inverted = backtest_engine(
        panel=panel,
        snapshots=snapshots,
        config=BacktestConfig(
            factor_cols=["inv_factor_a"],
            rebalance_freq=5,
            long_quantile=0.5,
            commission_bps=0.0,
        ),
    )

    assert list(original.factor_weight_history.columns) == ["factor_a"]
    assert list(inverted.factor_weight_history.columns) == ["inv_factor_a"]
    assert not original.performance["gross_ret"].equals(inverted.performance["gross_ret"])


def test_monthly_factor_diagnostics_reports_inverse_candidates():
    panel = make_test_panel()
    panel["factor_neg"] = -panel["factor_a"]

    diagnostics = monthly_factor_diagnostics(
        panel=panel,
        factor_cols=["factor_a", "factor_neg"],
        lookback_periods=5,
    )

    assert not diagnostics.empty
    assert {"factor", "icir", "recommended_sign", "recommended_factor_name", "inverse_candidate"}.issubset(diagnostics.columns)
    neg_rows = diagnostics[diagnostics["factor"] == "factor_neg"]
    assert not neg_rows.empty
    assert set(neg_rows["inverse_candidate"]) == {"inv_factor_neg"}


def test_prepare_clean_panel_keeps_dates_without_global_ticker_intersection():
    panel = pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "AAA", "Close": 100, "factor_a": 1.0, "fwd_ret_5d": 0.01},
            {"date": "2024-01-02", "ticker": "AAA", "Close": 101, "factor_a": 1.1, "fwd_ret_5d": 0.02},
            {"date": "2024-01-02", "ticker": "BBB", "Close": 50, "factor_a": 0.5, "fwd_ret_5d": 0.03},
            {"date": "2024-01-03", "ticker": "BBB", "Close": 51, "factor_a": 0.6, "fwd_ret_5d": 0.01},
            {"date": "2024-01-03", "ticker": "CCC", "Close": 75, "factor_a": 0.8, "fwd_ret_5d": 0.04},
        ]
    )
    cleaned = prepare_clean_panel(
        panel,
        BacktestConfig(factor_cols=["factor_a"], price_col="Close"),
    )

    kept_dates = sorted(pd.to_datetime(cleaned["date"]).dt.strftime("%Y-%m-%d").unique().tolist())
    assert kept_dates == ["2024-01-01", "2024-01-02"]
