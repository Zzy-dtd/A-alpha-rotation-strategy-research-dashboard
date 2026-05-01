from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


REQUIRED_UNIVERSE_COLUMNS = {"effective_date", "ticker"}
REQUIRED_OHLCV_COLUMNS = {"date", "ticker", "Open", "High", "Low", "Close", "Volume"}


@dataclass(frozen=True)
class UniverseSnapshot:
    effective_date: pd.Timestamp
    tickers: tuple[str, ...]


def load_universe_snapshots(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Universe snapshot file not found: {file_path}")

    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        snapshots = pd.read_excel(file_path)
    else:
        snapshots = pd.read_csv(file_path)

    missing = REQUIRED_UNIVERSE_COLUMNS - set(snapshots.columns)
    if missing:
        raise ValueError(f"Universe snapshot file is missing columns: {sorted(missing)}")

    snapshots = snapshots.copy()
    snapshots["effective_date"] = pd.to_datetime(snapshots["effective_date"])
    snapshots["ticker"] = snapshots["ticker"].astype(str).str.upper().str.strip()
    snapshots = snapshots.dropna(subset=["effective_date", "ticker"])
    snapshots = snapshots.sort_values(["effective_date", "ticker"]).reset_index(drop=True)

    if snapshots.empty:
        raise ValueError("Universe snapshot file is empty.")

    return snapshots


def load_universe_snapshots_from_bytes(file_name: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    buffer = BytesIO(payload)
    if suffix in {".xlsx", ".xls"}:
        snapshots = pd.read_excel(buffer)
    else:
        snapshots = pd.read_csv(buffer)

    missing = REQUIRED_UNIVERSE_COLUMNS - set(snapshots.columns)
    if missing:
        raise ValueError(f"Universe snapshot file is missing columns: {sorted(missing)}")

    snapshots = snapshots.copy()
    snapshots["effective_date"] = pd.to_datetime(snapshots["effective_date"])
    snapshots["ticker"] = snapshots["ticker"].astype(str).str.upper().str.strip()
    snapshots = snapshots.dropna(subset=["effective_date", "ticker"])
    snapshots = snapshots.sort_values(["effective_date", "ticker"]).reset_index(drop=True)
    if snapshots.empty:
        raise ValueError("Universe snapshot file is empty.")
    return snapshots


def get_snapshot_schedule(snapshots: pd.DataFrame) -> list[UniverseSnapshot]:
    grouped = snapshots.groupby("effective_date")["ticker"]
    schedule = [
        UniverseSnapshot(effective_date=dt, tickers=tuple(sorted(set(tickers))))
        for dt, tickers in grouped
    ]
    if not schedule:
        raise ValueError("No valid universe snapshots were found.")
    return schedule


def resolve_universe_by_date(
    snapshots: pd.DataFrame,
    trading_dates: Iterable[pd.Timestamp],
) -> dict[pd.Timestamp, tuple[str, ...]]:
    schedule = get_snapshot_schedule(snapshots)
    resolved: dict[pd.Timestamp, tuple[str, ...]] = {}

    for raw_date in trading_dates:
        date = pd.Timestamp(raw_date)
        eligible = [snapshot for snapshot in schedule if snapshot.effective_date <= date]
        if not eligible:
            first_date = schedule[0].effective_date.strftime("%Y-%m-%d")
            raise ValueError(
                f"No universe snapshot is available for {date.strftime('%Y-%m-%d')}. "
                f"First snapshot starts at {first_date}."
            )
        resolved[date] = eligible[-1].tickers

    return resolved


def unique_universe_tickers(snapshots: pd.DataFrame) -> list[str]:
    return sorted(snapshots["ticker"].dropna().astype(str).str.upper().unique().tolist())


def load_price_history(
    snapshots: pd.DataFrame,
    start: str,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    tickers = unique_universe_tickers(snapshots)
    if not tickers:
        raise ValueError("Universe snapshots did not contain any tickers.")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise ValueError("No price data was returned by yfinance.")
    return raw


def load_ohlcv_panel(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"OHLCV file not found: {file_path}")

    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        panel = pd.read_excel(file_path)
    else:
        panel = pd.read_csv(file_path)
    return normalize_ohlcv_panel(panel)


def load_ohlcv_panel_from_bytes(file_name: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    buffer = BytesIO(payload)
    if suffix in {".xlsx", ".xls"}:
        panel = pd.read_excel(buffer)
    else:
        panel = pd.read_csv(buffer)
    return normalize_ohlcv_panel(panel)


def normalize_ohlcv_panel(panel: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_OHLCV_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(f"OHLCV file is missing columns: {sorted(missing)}")

    normalized = panel.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper().str.strip()
    normalized = normalized.dropna(subset=["date", "ticker"])
    normalized = normalized.sort_values(["date", "ticker"]).reset_index(drop=True)
    return normalized


def panel_from_yfinance(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("Price history is empty.")

    close = raw["Close"]
    start_date = close.index[0]
    valid_tickers = close.columns[close.loc[start_date].notna()].tolist()

    if not valid_tickers:
        raise ValueError("No valid tickers had data on the start date.")

    fields = raw.columns.levels[0]
    raw_clean = pd.concat(
        {
            field: raw[field][valid_tickers]
            for field in fields
            if field in raw.columns.get_level_values(0)
        },
        axis=1,
    ).sort_index(axis=1)

    panel = raw_clean.stack(level=1).rename_axis(["date", "ticker"]).reset_index()
    panel.columns = ["date", "ticker", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ticker"] = panel["ticker"].astype(str)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_benchmark_history(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    benchmark = yf.download(symbol, start=start, end=end)["Close"].reset_index()
    if benchmark.empty:
        raise ValueError(f"No benchmark data was returned for {symbol}.")

    benchmark["date"] = pd.to_datetime(benchmark["Date"])
    benchmark = benchmark.sort_values("date").set_index("date")
    benchmark["bench_ret"] = benchmark[symbol].pct_change()
    return benchmark[[symbol, "bench_ret"]]
