from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from alpha_rotation.dashboard_helpers import (
    build_research_artifacts,
    load_panel_input,
    load_snapshots_input,
    prepare_download_tables,
    run_backtest_with_outputs,
)
from alpha_rotation.factors import DEFAULT_FACTOR_COLUMNS
from alpha_rotation.reporting import (
    plot_performance,
    plot_factor_weight_timeseries,
    plot_position_days_by_year,
    plot_rebalance_activity,
    rebalance_detail,
    summarize_factor_diagnostics,
)


DEFAULT_START = "2014-01-01"
DEFAULT_UNIVERSE_PATH = "inputs/sp500_historical_constituents_semiannual_2014_2026.csv"
DEFAULT_BENCHMARK = "^GSPC"


st.set_page_config(page_title="Alpha Research Dashboard", layout="wide")
st.title("Alpha Research Dashboard")
st.caption("Discover alphas, review IC/ICIR diagnostics, test signed factors, and inspect detailed backtest outputs.")


def dataframe_download(df: pd.DataFrame, label: str, file_name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=file_name, mime="text/csv")


@st.cache_data(show_spinner=False)
def cached_load_snapshots(path: str | None, uploaded_name: str | None, uploaded_bytes: bytes | None):
    uploaded = (uploaded_name, uploaded_bytes) if uploaded_name and uploaded_bytes is not None else None
    return load_snapshots_input(path=path, uploaded=uploaded)


@st.cache_data(show_spinner=False)
def cached_load_panel(mode: str, snapshots: pd.DataFrame, start: str, end: str | None, uploaded_name: str | None, uploaded_bytes: bytes | None):
    uploaded = (uploaded_name, uploaded_bytes) if uploaded_name and uploaded_bytes is not None else None
    return load_panel_input(
        mode=mode,
        snapshots=snapshots,
        start=start,
        end=end,
        uploaded_ohlcv=uploaded,
    )


@st.cache_data(show_spinner=False)
def cached_build_artifacts(
    panel: pd.DataFrame,
    factor_cols: list[str],
    lookback_periods: int,
    custom_factor_items: tuple[tuple[str, str], ...],
    diagnostic_start: str | None,
    diagnostic_end: str | None,
):
    return build_research_artifacts(
        panel=panel,
        factor_cols=factor_cols,
        lookback_periods=lookback_periods,
        custom_factors=dict(custom_factor_items),
        diagnostic_start=diagnostic_start,
        diagnostic_end=diagnostic_end,
    )


@st.cache_data(show_spinner=False)
def cached_run_backtest(
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
):
    return run_backtest_with_outputs(
        factor_panel=factor_panel,
        snapshots=snapshots,
        chosen_factors=chosen_factors,
        benchmark_symbol=benchmark_symbol,
        start=start,
        end=end,
        rebalance_freq=rebalance_freq,
        long_quantile=long_quantile,
        short_quantile=short_quantile,
        commission_bps=commission_bps,
    )


with st.sidebar:
    st.header("Data Source")
    panel_mode = st.radio(
        "Data source",
        options=["yfinance", "upload"],
        format_func=lambda value: {
            "yfinance": "yfinance",
            "upload": "Upload file",
        }[value],
    )
    start_date = st.text_input("Start date", DEFAULT_START)
    end_date_text = st.text_input("End date (optional)", "")
    end_date = end_date_text or None
    benchmark_symbol = st.text_input("Benchmark symbol", DEFAULT_BENCHMARK)

    st.header("Universe")
    universe_path = st.text_input("Universe snapshot path", DEFAULT_UNIVERSE_PATH)
    universe_upload = st.file_uploader("Or upload universe snapshot CSV/XLSX", type=["csv", "xlsx", "xls"])

    st.header("Backtest Config")
    lookback_periods = st.number_input("IC/ICIR lookback periods", min_value=5, max_value=252, value=20, step=1)
    rebalance_freq = st.number_input("Rebalance frequency (days)", min_value=1, max_value=63, value=5, step=1)
    long_quantile = st.slider("Long quantile", min_value=0.0, max_value=0.50, value=0.20, step=0.05)
    short_quantile = st.slider("Short quantile", min_value=0.0, max_value=0.50, value=0.0, step=0.05)
    commission_bps = st.number_input("Commission (bps)", min_value=0.0, max_value=500.0, value=10.0, step=1.0)

    uploaded_ohlcv = None
    if panel_mode == "upload":
        uploaded_ohlcv = st.file_uploader("Upload data CSV/XLSX", type=["csv", "xlsx", "xls"])


st.subheader("1. Load Data")
load_clicked = st.button("Load universe and data", type="primary")

if "snapshots" not in st.session_state:
    st.session_state["snapshots"] = None
if "panel" not in st.session_state:
    st.session_state["panel"] = None
if "artifacts" not in st.session_state:
    st.session_state["artifacts"] = None

if load_clicked:
    try:
        uploaded_universe = None
        if universe_upload is not None:
            uploaded_universe = (universe_upload.name, universe_upload.getvalue())
        snapshots = cached_load_snapshots(
            path=universe_path if uploaded_universe is None else None,
            uploaded_name=uploaded_universe[0] if uploaded_universe else None,
            uploaded_bytes=uploaded_universe[1] if uploaded_universe else None,
        )
        uploaded_panel_tuple = None
        if uploaded_ohlcv is not None:
            uploaded_panel_tuple = (uploaded_ohlcv.name, uploaded_ohlcv.getvalue())
        panel = cached_load_panel(
            mode=panel_mode,
            snapshots=snapshots,
            start=start_date,
            end=end_date,
            uploaded_name=uploaded_panel_tuple[0] if uploaded_panel_tuple else None,
            uploaded_bytes=uploaded_panel_tuple[1] if uploaded_panel_tuple else None,
        )
        st.session_state["snapshots"] = snapshots
        st.session_state["panel"] = panel
        st.success("Data loaded successfully.")
    except Exception as exc:
        st.error(str(exc))

if st.session_state["snapshots"] is not None:
    feature_columns = [
        column for column in st.session_state["panel"].columns
        if column not in ["date", "ticker"]
    ]
    st.write("Available input features")
    st.write(feature_columns)


st.subheader("2. Alpha Lab")
default_custom_text = st.session_state.get(
    "custom_factor_text",
    "custom_spread = Close - Open\ncustom_range = High - Low",
)
custom_factor_text = st.text_area(
    "Custom factors (one per line: name = formula)",
    value=default_custom_text,
    height=110,
)
st.session_state["custom_factor_text"] = custom_factor_text

diagnostic_col1, diagnostic_col2 = st.columns(2)
with diagnostic_col1:
    diagnostic_start_text = st.text_input(
        "Diagnostic start date",
        value=st.session_state.get("diagnostic_start_text", start_date),
        key="diagnostic_start_text",
    )
with diagnostic_col2:
    diagnostic_end_text = st.text_input(
        "Diagnostic end date",
        value=st.session_state.get("diagnostic_end_text", end_date_text),
        key="diagnostic_end_text",
    )

build_clicked = st.button("Build factor diagnostics")


def parse_custom_factor_text(raw_text: str) -> dict[str, str]:
    custom_factors: dict[str, str] = {}
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            raise ValueError(f"Invalid custom factor line: {stripped}")
        name, formula = stripped.split("=", 1)
        custom_factors[name.strip()] = formula.strip()
    return custom_factors


if build_clicked:
    try:
        if st.session_state["panel"] is None:
            raise ValueError("Load data first.")
        custom_factors = parse_custom_factor_text(custom_factor_text)
        artifacts = cached_build_artifacts(
            panel=st.session_state["panel"],
            factor_cols=DEFAULT_FACTOR_COLUMNS,
            lookback_periods=int(lookback_periods),
            custom_factor_items=tuple(custom_factors.items()),
            diagnostic_start=diagnostic_start_text or None,
            diagnostic_end=diagnostic_end_text or None,
        )
        st.session_state["artifacts"] = artifacts
        st.success("Factor diagnostics built successfully.")
    except Exception as exc:
        st.error(str(exc))

artifacts = st.session_state.get("artifacts")
if artifacts is not None:
    st.write("Custom factor validation")
    validation_rows = [
        {
            "name": item.name,
            "formula": item.formula,
            "is_valid": item.is_valid,
            "message": item.message,
        }
        for item in artifacts.custom_factor_results
    ]
    if validation_rows:
        st.dataframe(pd.DataFrame(validation_rows), use_container_width=True)

    diagnostics = artifacts.diagnostics
    if diagnostics.empty:
        st.warning("No diagnostics available yet. Try a longer date range or simpler factor set.")
    else:
        range_label = (
            f"{diagnostic_start_text or 'beginning'} to {diagnostic_end_text or 'latest'}"
        )
        latest_selection_date = diagnostics["selection_date"].max()
        latest_summary = summarize_factor_diagnostics(diagnostics, latest_selection_date)[
            ["factor", "mean_ic", "ic_std", "icir", "inverse_candidate"]
        ]

        st.write(f"Diagnostics range: {range_label}")
        st.write(f"Latest diagnostics in selected range for {latest_selection_date.date()}")
        st.dataframe(latest_summary, use_container_width=True, height=360)
        dataframe_download(latest_summary, "Download latest diagnostics", "latest_factor_diagnostics.csv")

        all_candidates = []
        diagnostics_source = summarize_factor_diagnostics(diagnostics, latest_selection_date)
        for row in diagnostics_source.itertuples(index=False):
            all_candidates.append(row.factor)
            all_candidates.append(row.inverse_candidate)
        unique_candidates = list(dict.fromkeys(all_candidates))
        recommended = diagnostics_source["recommended_factor_name"].head(3).tolist()
        chosen_factors = st.multiselect(
            "Choose signed factors for the backtest",
            options=unique_candidates,
            default=recommended,
        )

        st.subheader("3. Run Backtest")
        run_clicked = st.button("Run backtest")
        if run_clicked:
            try:
                if not chosen_factors:
                    raise ValueError("Choose at least one factor before running the backtest.")
                result, combined = cached_run_backtest(
                    factor_panel=artifacts.factor_panel,
                    snapshots=st.session_state["snapshots"],
                    chosen_factors=chosen_factors,
                    benchmark_symbol=benchmark_symbol,
                    start=start_date,
                    end=end_date,
                    rebalance_freq=int(rebalance_freq),
                    long_quantile=float(long_quantile),
                    short_quantile=float(short_quantile),
                    commission_bps=float(commission_bps),
                )
                st.session_state["result"] = result
                st.session_state["combined"] = combined
                st.session_state["chosen_factors"] = chosen_factors
                st.success("Backtest completed.")
            except Exception as exc:
                st.error(str(exc))


result = st.session_state.get("result")
combined = st.session_state.get("combined")
if result is not None and combined is not None:
    st.subheader("4. Results")
    st.write("Chosen factors")
    st.write(st.session_state.get("chosen_factors", []))

    col1, col2 = st.columns(2)
    with col1:
        fig, _ = plot_performance(combined)
        st.pyplot(fig, clear_figure=True)
    with col2:
        fig2, _ = plot_rebalance_activity(result)
        st.pyplot(fig2, clear_figure=True)

    fig3, _ = plot_factor_weight_timeseries(result)
    st.pyplot(fig3, clear_figure=True)

    available_years = sorted(pd.to_datetime(result.holdings_history["date"]).dt.year.unique().tolist())
    if available_years:
        selected_year = st.selectbox("Position-day count year", options=available_years)
        fig4, _ = plot_position_days_by_year(result, selected_year)
        st.pyplot(fig4, clear_figure=True)

    if not result.rebalance_log.empty:
        selected_date = st.selectbox(
            "Inspect one rebalance date",
            options=list(result.rebalance_log.index),
            format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"),
        )
        st.dataframe(rebalance_detail(result, selected_date), use_container_width=True, height=240)

    tables = prepare_download_tables(
        diagnostics=artifacts.diagnostics if artifacts is not None else None,
        result=result,
        benchmark_results=combined,
    )
    st.subheader("5. Downloads")
    for name, table in tables.items():
        dataframe_download(table, f"Download {name}", f"{name}.csv")
