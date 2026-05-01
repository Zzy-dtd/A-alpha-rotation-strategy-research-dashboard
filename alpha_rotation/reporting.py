from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import BacktestResult


def combine_with_benchmark(result: BacktestResult, benchmark: pd.DataFrame) -> pd.DataFrame:
    combined = result.performance.join(benchmark[["bench_ret"]], how="left").dropna(subset=["bench_ret"])
    combined["Strategy_Cum"] = (1 + combined["daily_ret"]).cumprod() - 1
    combined["Benchmark_Cum"] = (1 + combined["bench_ret"]).cumprod() - 1
    combined["Excess_Cum"] = combined["Strategy_Cum"] - combined["Benchmark_Cum"]
    return combined


def plot_performance(final_results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(final_results.index, final_results["Strategy_Cum"], label="Strategy", color="royalblue", linewidth=2)
    ax.plot(final_results.index, final_results["Benchmark_Cum"], label="Benchmark", color="grey", linestyle="--", alpha=0.7)
    ax.fill_between(final_results.index, final_results["Excess_Cum"], color="orange", alpha=0.15, label="Excess Return")
    ax.set_title("Performance Comparison: Strategy vs Benchmark", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


def summarize_rebalances(result: BacktestResult) -> pd.DataFrame:
    if result.rebalance_log.empty:
        return pd.DataFrame(columns=["holdings_count", "long_count", "short_count", "turnover", "commission_cost", "adds", "drops"])

    summary = result.rebalance_log[["holdings_count", "long_count", "short_count", "turnover", "commission_cost"]].copy()
    summary["adds"] = result.rebalance_log["added"].apply(len)
    summary["drops"] = result.rebalance_log["removed"].apply(len)
    return summary


def rebalance_detail(result: BacktestResult, date: str | pd.Timestamp) -> pd.DataFrame:
    target_date = pd.Timestamp(date)
    trades = result.rebalance_trades.copy()
    if trades.empty:
        return trades
    detail = trades[trades["date"] == target_date].copy()
    return detail.sort_values(["action", "ticker"]).reset_index(drop=True)


def plot_rebalance_activity(result: BacktestResult):
    summary = summarize_rebalances(result)
    fig, ax = plt.subplots(figsize=(12, 6))

    if summary.empty:
        ax.set_title("No rebalance events available")
        return fig, (ax,)

    ax.bar(summary.index, summary["turnover"], color="steelblue", alpha=0.75, label="Turnover")
    ax.set_ylabel("Turnover", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.grid(True, alpha=0.3)

    cost_ax = ax.twinx()
    cost_ax.plot(summary.index, summary["commission_cost"], color="darkred", marker="o", label="Commission cost")
    cost_ax.set_ylabel("Commission Cost", color="darkred")
    cost_ax.tick_params(axis="y", labelcolor="darkred")
    ax.set_title("Turnover and Commission Cost by Rebalance")
    ax.set_xlabel("Rebalance Date")

    handles, labels = ax.get_legend_handles_labels()
    cost_handles, cost_labels = cost_ax.get_legend_handles_labels()
    ax.legend(handles + cost_handles, labels + cost_labels, loc="upper left")
    return fig, (ax, cost_ax)


def plot_position_days_by_year(result: BacktestResult, year: int):
    fig, ax = plt.subplots(figsize=(12, 7))
    holdings = result.holdings_history.copy()
    if holdings.empty:
        ax.set_title("No holdings history available")
        return fig, ax

    holdings["date"] = pd.to_datetime(holdings["date"])
    active = holdings[(holdings["date"].dt.year == year) & (holdings["weight"] != 0)]
    counts = active.groupby("ticker")["date"].nunique().sort_values(ascending=False).head(20)
    if counts.empty:
        ax.set_title(f"No positions recorded in {year}")
        return fig, ax

    counts.plot(kind="bar", ax=ax, color="darkgreen", alpha=0.8)
    ax.set_title(f"Top 20 Stocks by Days in Position for {year}")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Days in Position")
    ax.grid(True, axis="y", alpha=0.3)
    return fig, ax


def plot_factor_weight_timeseries(result: BacktestResult):
    fig, ax = plt.subplots(figsize=(12, 7))
    weights = result.factor_weight_history.copy()
    if weights.empty:
        ax.set_title("No factor weight history available")
        return fig, ax

    for column in weights.columns:
        ax.plot(weights.index, weights[column], label=column, linewidth=1.8)
    ax.set_title("Factor Weight Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    return fig, ax


def summarize_factor_diagnostics(diagnostics: pd.DataFrame, selection_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics

    summary = diagnostics.copy()
    if selection_date is not None:
        summary = summary[summary["selection_date"] == pd.Timestamp(selection_date)]
    return summary.sort_values(["selection_date", "icir"], ascending=[True, False]).reset_index(drop=True)
