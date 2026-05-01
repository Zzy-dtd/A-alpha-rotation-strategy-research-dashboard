from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestConfig, BacktestResult, backtest_engine, monthly_factor_diagnostics
from .custom_factors import apply_custom_factors
from .data import (
    load_benchmark_history,
    load_ohlcv_panel_from_bytes,
    load_price_history,
    load_universe_snapshots,
    load_universe_snapshots_from_bytes,
    panel_from_yfinance,
)
from .factors import BASE_PANEL_COLUMNS, DEFAULT_FACTOR_COLUMNS, build_factor_panel
from .reporting import combine_with_benchmark, summarize_factor_diagnostics, summarize_rebalances


@dataclass
class ResearchRunArtifacts:
    factor_panel: pd.DataFrame
    diagnostics: pd.DataFrame
    custom_factor_results: list


def load_snapshots_input(path: str | None = None, uploaded: tuple[str, bytes] | None = None) -> pd.DataFrame:
    if uploaded is not None:
        file_name, payload = uploaded
        return load_universe_snapshots_from_bytes(file_name, payload)
    if path:
        return load_universe_snapshots(path)
    raise ValueError("Provide either a universe path or an uploaded universe file.")


def load_panel_input(
    mode: str,
    snapshots: pd.DataFrame,
    start: str,
    end: str | None,
    uploaded_ohlcv: tuple[str, bytes] | None = None,
) -> pd.DataFrame:
    if mode == "yfinance":
        raw = load_price_history(snapshots=snapshots, start=start, end=end)
        return panel_from_yfinance(raw)
    if mode == "upload":
        if uploaded_ohlcv is None:
            raise ValueError("Upload an OHLCV file to use upload mode.")
        file_name, payload = uploaded_ohlcv
        return load_ohlcv_panel_from_bytes(file_name, payload)
    raise ValueError(f"Unsupported panel load mode: {mode}")


def build_research_artifacts(
    panel: pd.DataFrame,
    factor_cols: list[str] | None = None,
    lookback_periods: int = 20,
    custom_factors: dict[str, str] | None = None,
    diagnostic_start: str | None = None,
    diagnostic_end: str | None = None,
) -> ResearchRunArtifacts:
    factor_panel = build_factor_panel(panel)
    factor_library = list(factor_cols or DEFAULT_FACTOR_COLUMNS)
    custom_factor_results = []

    if custom_factors:
        factor_panel, custom_factor_results = apply_custom_factors(
            factor_panel,
            custom_factors=custom_factors,
            allowed_columns=[column for column in factor_panel.columns if column != "date"],
        )
        factor_library = factor_library + [result.name for result in custom_factor_results if result.is_valid]

    diagnostic_panel = factor_panel.copy()
    if diagnostic_start:
        diagnostic_panel = diagnostic_panel[
            pd.to_datetime(diagnostic_panel["date"]) >= pd.Timestamp(diagnostic_start)
        ].copy()
    if diagnostic_end:
        diagnostic_panel = diagnostic_panel[
            pd.to_datetime(diagnostic_panel["date"]) <= pd.Timestamp(diagnostic_end)
        ].copy()

    diagnostics = monthly_factor_diagnostics(
        panel=diagnostic_panel,
        factor_cols=factor_library,
        lookback_periods=lookback_periods,
    )
    return ResearchRunArtifacts(
        factor_panel=factor_panel,
        diagnostics=diagnostics,
        custom_factor_results=custom_factor_results,
    )


def run_backtest_with_outputs(
    factor_panel: pd.DataFrame,
    snapshots: pd.DataFrame,
    chosen_factors: list[str],
    benchmark_symbol: str,
    start: str,
    end: str | None,
    rebalance_freq: int,
    long_quantile: float,
    short_quantile: float,
    commission_bps: float,
) -> tuple[BacktestResult, pd.DataFrame]:
    panel_selected = factor_panel.copy()
    config = BacktestConfig(
        factor_cols=chosen_factors,
        rebalance_freq=rebalance_freq,
        long_quantile=long_quantile,
        short_quantile=short_quantile,
        commission_bps=commission_bps,
    )
    result = backtest_engine(panel=panel_selected, snapshots=snapshots, config=config)
    benchmark = load_benchmark_history(benchmark_symbol, start=start, end=end)
    combined = combine_with_benchmark(result, benchmark)
    return result, combined


def prepare_download_tables(
    diagnostics: pd.DataFrame | None = None,
    result: BacktestResult | None = None,
    benchmark_results: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    if diagnostics is not None:
        tables["factor_diagnostics"] = diagnostics.copy()
        if not diagnostics.empty:
            latest_date = diagnostics["selection_date"].max()
            tables["latest_factor_diagnostics"] = summarize_factor_diagnostics(diagnostics, latest_date)
    if result is not None:
        tables["performance"] = result.performance.reset_index()
        tables["rebalance_summary"] = summarize_rebalances(result).reset_index()
        tables["rebalance_log"] = result.rebalance_log.reset_index()
        tables["rebalance_trades"] = result.rebalance_trades.copy()
        tables["holdings_history"] = result.holdings_history.copy()
        tables["factor_weight_history"] = result.factor_weight_history.reset_index()
        tables["universe_history"] = result.universe_history.reset_index()
    if benchmark_results is not None:
        tables["benchmark_comparison"] = benchmark_results.reset_index()
    return tables
