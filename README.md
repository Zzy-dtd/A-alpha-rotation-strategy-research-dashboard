# Alpha Research Dashboard

Local research dashboard for discovering alphas, reviewing factor diagnostics, and running long-only or long/short backtests on equity universes driven by dated snapshot files.

The app is built on top of the `alpha_rotation` package and exposes a Streamlit interface for:

- loading universe snapshots and market data
- building built-in and custom alpha features
- reviewing IC / ICIR diagnostics on a user-chosen date range
- selecting raw or inverted factors
- running backtests with commission, turnover, long, and short controls
- inspecting performance, factor weights, rebalance activity, and position persistence

## Main Files

- [dashboard_app.py](./dashboard_app.py): Streamlit dashboard entrypoint
- [alpha_rotation](./alpha_rotation): core data, factor, diagnostics, backtest, and reporting logic
- [inputs](./inputs): example universe files
- [tests](./tests): regression tests

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the dashboard:

```powershell
streamlit run dashboard_app.py
```

4. Open the local Streamlit URL shown in the terminal.

## What the Dashboard Does

The dashboard has four main steps:

1. Load a universe file plus data from either `yfinance` or an uploaded file.
2. Build the Alpha Lab diagnostics over a user-selected diagnostic date range.
3. Choose factors, including inverse factors such as `inv_mom_3m`.
4. Run the backtest and inspect the outputs.

## Input Files

There are two user-controlled inputs:

- a universe snapshot file
- optionally an uploaded market-data file

### 1. Universe Snapshot File

This file defines the stock pool over time.

Required columns:

- `effective_date`
- `ticker`

Optional columns:

- `source`
- `note`
- any other metadata columns; they are ignored by the backtest logic

Supported file types:

- `.csv`
- `.xlsx`
- `.xls`

Behavior:

- each row means the ticker is active starting from `effective_date`
- for any trading date, the active stock pool is the latest snapshot whose `effective_date` is less than or equal to that trading date
- snapshots should usually contain the full universe for that date, not only additions or removals

Minimal example:

```csv
effective_date,ticker,source,note
2023-01-01,AAPL,manual,starting universe
2023-01-01,MSFT,manual,starting universe
2023-07-01,AAPL,manual,updated universe
2023-07-01,MSFT,manual,updated universe
2023-07-01,NVDA,manual,updated universe
```

Important:

- tickers should match `yfinance` naming where possible, for example `BRK-B` rather than `BRK.B`
- the first snapshot date must be on or before the backtest start date

### 2. Uploaded Market-Data File

If you choose `upload` instead of `yfinance`, the uploaded file must contain at least these columns:

- `date`
- `ticker`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

Supported file types:

- `.csv`
- `.xlsx`
- `.xls`

Additional columns are allowed and preserved. This is useful when your uploaded data contains extra features, fundamentals, alternative data fields, or precomputed signals that you want to use in custom alpha formulas.

Minimal example:

```csv
date,ticker,Open,High,Low,Close,Volume
2024-01-02,AAPL,185.64,188.44,183.89,185.64,82488700
2024-01-02,MSFT,370.83,373.26,366.78,370.87,25258600
```

## Custom Factor Formulas

You can define custom factors in the Alpha Lab with one formula per line:

```text
spread = Close - Open
range_ratio = (High - Low) / Close
```

Rules:

- factor names must use only letters, numbers, and underscores
- factor names cannot start with `inv_`
- formulas may use only:
  - existing column names
  - numbers
  - `+`, `-`, `*`, `/`
  - parentheses
- arbitrary Python code is not allowed

If your uploaded data includes extra columns, those columns can also be used in custom formulas.

## Diagnostics

The Alpha Lab diagnostics are separate from the backtest weight-adjustment logic.

You can control:

- `IC/ICIR lookback periods`: rolling history used to compute diagnostics
- `Diagnostic start date`
- `Diagnostic end date`

The dashboard runs diagnostics only on the selected diagnostic range, then shows:

- `factor`
- `mean_ic`
- `ic_std`
- `icir`
- `inverse_candidate`

You can then choose factors manually from that result.

## Backtest Controls

Main controls available in the dashboard:

- benchmark symbol
- rebalance frequency
- long quantile
- short quantile
- commission in basis points

The backtest supports:

- long-only
- short-only
- long/short

## Current Outputs

The dashboard currently shows:

- strategy vs benchmark cumulative performance
- turnover and commission-cost chart
- factor weight time series
- yearly top-20 position-days-by-stock chart
- per-rebalance trade detail for a selected rebalance date

It also allows CSV downloads of diagnostics and backtest output tables.

## Example Universe Files

Example files are included in [inputs](./inputs):

- [sp500_universe_snapshots.csv](./inputs/sp500_universe_snapshots.csv): small starter example
- [sp500_historical_constituents_semiannual_2014_2026.csv](./inputs/sp500_historical_constituents_semiannual_2014_2026.csv): larger historical-style example

## Testing

Run the test suite with:

```powershell
pytest -q
```

## Notes

- `yfinance` coverage for old or delisted symbols may be incomplete.
- Large historical universes can take time to download.
- Uploaded files are often the best choice when you already have richer internal data.
