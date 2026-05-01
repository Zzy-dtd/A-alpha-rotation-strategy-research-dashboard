from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data import resolve_universe_by_date


@dataclass(frozen=True)
class BacktestConfig:
    factor_cols: list[str]
    rebalance_freq: int = 5
    long_quantile: float = 0.2
    short_quantile: float = 0.0
    commission_bps: float = 10.0
    weight_method: str = "icir"
    ic_method: str = "spearman"
    universe_refresh_rule: str = "effective_until_next_snapshot"
    price_col: str = "Close"
    label_col: str = "fwd_ret_5d"
    ticker_col: str = "ticker"
    date_col: str = "date"


@dataclass
class BacktestResult:
    performance: pd.DataFrame
    factor_weight_history: pd.DataFrame
    holdings_history: pd.DataFrame
    rebalance_log: pd.DataFrame
    rebalance_trades: pd.DataFrame
    ic_history: pd.DataFrame
    final_weights: pd.Series
    universe_history: pd.DataFrame


@dataclass(frozen=True)
class SignedFactor:
    source_factor: str
    display_name: str
    sign: int = 1


def cs_zscore(x: pd.Series) -> pd.Series:
    std = x.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=x.index)
    return (x - x.mean()) / std


def compute_weights(ic_window: pd.DataFrame, method: str = "icir") -> pd.Series:
    if ic_window.empty:
        return pd.Series(dtype=float)

    mu = ic_window.mean()
    if method == "mean_ic":
        raw_w = mu
    elif method == "icir":
        sigma = ic_window.std().replace(0, np.nan)
        raw_w = mu / sigma
    elif method == "mean_abs":
        raw_w = np.sign(mu) * np.abs(mu)
    else:
        raw_w = pd.Series(1.0, index=ic_window.columns)

    raw_w = raw_w.fillna(0.0)
    if raw_w.abs().sum() == 0:
        return pd.Series(1.0 / len(raw_w), index=raw_w.index)
    return raw_w / raw_w.abs().sum()


def prepare_clean_panel(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    clean = df.copy()
    clean[config.date_col] = pd.to_datetime(clean[config.date_col])
    clean = clean.dropna(subset=config.factor_cols + [config.label_col])
    if clean.empty:
        return clean

    for factor in config.factor_cols:
        clean[factor] = clean.groupby(config.date_col)[factor].transform(cs_zscore)

    clean["ret_1d_next"] = clean.groupby(config.ticker_col)[config.price_col].pct_change().shift(-1)
    clean = clean.dropna(subset=["ret_1d_next"])
    return clean.sort_values([config.date_col, config.ticker_col]).reset_index(drop=True)


def build_ic_history(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    records = []
    for dt, day_data in df.groupby(config.date_col):
        day_ic = {
            factor: day_data[factor].corr(day_data[config.label_col], method=config.ic_method)
            for factor in config.factor_cols
        }
        day_ic["date"] = dt
        records.append(day_ic)
    if not records:
        return pd.DataFrame(columns=config.factor_cols, index=pd.Index([], name="date"))
    return pd.DataFrame(records).set_index("date").sort_index()


def monthly_factor_diagnostics(
    panel: pd.DataFrame,
    factor_cols: list[str],
    ic_method: str = "spearman",
    lookback_periods: int = 20,
    date_col: str = "date",
    ticker_col: str = "ticker",
    label_col: str = "fwd_ret_5d",
) -> pd.DataFrame:
    config = BacktestConfig(
        factor_cols=factor_cols,
        ic_method=ic_method,
        date_col=date_col,
        ticker_col=ticker_col,
        label_col=label_col,
    )
    clean = prepare_clean_panel(panel, config)
    ic_history = build_ic_history(clean, config)
    if ic_history.empty:
        return pd.DataFrame(
            columns=[
                "selection_date",
                "factor",
                "mean_ic",
                "ic_std",
                "icir",
                "recommended_sign",
                "recommended_factor_name",
                "original_candidate",
                "inverse_candidate",
            ]
        )

    reweight_dates = sorted(monthly_reweight_dates(ic_history.index))
    records = []
    for selection_date in reweight_dates:
        if selection_date not in ic_history.index:
            continue
        end_loc = ic_history.index.get_loc(selection_date)
        end_idx = end_loc - 5
        start_idx = end_idx - lookback_periods
        if start_idx < 0:
            continue

        ic_window = ic_history.iloc[start_idx : end_idx + 1]
        mu = ic_window.mean()
        sigma = ic_window.std()
        icir = (mu / sigma.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

        for factor in factor_cols:
            mean_ic = mu.get(factor, np.nan)
            ic_std = sigma.get(factor, np.nan)
            factor_icir = icir.get(factor, np.nan)
            recommended_sign = "inverse" if pd.notna(factor_icir) and factor_icir < 0 else "original"
            recommended_factor_name = (
                f"inv_{factor}" if recommended_sign == "inverse" else factor
            )
            records.append(
                {
                    "selection_date": pd.Timestamp(selection_date),
                    "factor": factor,
                    "mean_ic": mean_ic,
                    "ic_std": ic_std,
                    "icir": factor_icir,
                    "recommended_sign": recommended_sign,
                    "recommended_factor_name": recommended_factor_name,
                    "original_candidate": factor,
                    "inverse_candidate": f"inv_{factor}",
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "selection_date",
                "factor",
                "mean_ic",
                "ic_std",
                "icir",
                "recommended_sign",
                "recommended_factor_name",
                "original_candidate",
                "inverse_candidate",
            ]
        )

    return pd.DataFrame(records).sort_values(
        ["selection_date", "icir"], ascending=[True, False]
    ).reset_index(drop=True)


def resolve_signed_factor_selection(selected_factors: list[str]) -> list[SignedFactor]:
    resolved = []
    for item in selected_factors:
        if item.startswith("inv_"):
            source_factor = item[4:]
            resolved.append(SignedFactor(source_factor=source_factor, display_name=item, sign=-1))
        else:
            resolved.append(SignedFactor(source_factor=item, display_name=item, sign=1))
    return resolved


def monthly_reweight_dates(trading_dates: Iterable[pd.Timestamp]) -> set[pd.Timestamp]:
    dates = pd.Series(pd.to_datetime(list(trading_dates)))
    return set(dates.groupby(dates.dt.to_period("M")).first().tolist())


def build_target_positions(
    df: pd.DataFrame,
    trading_dates: list[pd.Timestamp],
    date_index: int,
    current_weights: pd.Series,
    selected_factors: list[SignedFactor],
    active_universe: tuple[str, ...],
    config: BacktestConfig,
) -> pd.Series:
    start_smooth = max(0, date_index - (config.rebalance_freq - 1))
    smooth_window = trading_dates[start_smooth : date_index + 1]
    eligible = df[
        df[config.date_col].isin(smooth_window) & df[config.ticker_col].isin(active_universe)
    ]
    source_factors = list(dict.fromkeys(factor.source_factor for factor in selected_factors))
    alpha_avg = eligible.groupby(config.ticker_col)[source_factors].mean()
    if alpha_avg.empty:
        return pd.Series(dtype=float)

    signed_alpha = pd.DataFrame(index=alpha_avg.index)
    for factor in selected_factors:
        signed_alpha[factor.display_name] = alpha_avg[factor.source_factor] * factor.sign

    scores = (signed_alpha * current_weights.reindex(signed_alpha.columns)).sum(axis=1).sort_values(ascending=False)
    positions = pd.Series(dtype=float)

    if config.long_quantile > 0:
        n_long = max(1, int(len(scores) * config.long_quantile))
        long_names = scores.head(n_long).index
        long_scale = 1.0 if config.short_quantile <= 0 else 0.5
        positions = pd.concat(
            [positions, pd.Series(long_scale / n_long, index=long_names, dtype=float)]
        )

    if config.short_quantile > 0:
        n_short = max(1, int(len(scores) * config.short_quantile))
        short_names = scores.tail(n_short).index
        short_scale = 0.5 if config.long_quantile > 0 else 1.0
        short_positions = pd.Series(-short_scale / n_short, index=short_names, dtype=float)
        positions = pd.concat([positions, short_positions])

    if positions.empty:
        return positions

    positions = positions.groupby(level=0).sum()
    positions = positions[positions != 0]
    return positions.sort_index()


def calculate_turnover(previous_positions: pd.Series, target_positions: pd.Series) -> float:
    combined = pd.concat(
        [previous_positions.rename("old"), target_positions.rename("new")],
        axis=1,
    ).fillna(0.0)
    return 0.5 * (combined["new"] - combined["old"]).abs().sum()


def build_trade_log(
    date: pd.Timestamp,
    previous_positions: pd.Series,
    target_positions: pd.Series,
) -> pd.DataFrame:
    combined = pd.concat(
        [previous_positions.rename("old_weight"), target_positions.rename("new_weight")],
        axis=1,
    ).fillna(0.0)
    combined["weight_change"] = combined["new_weight"] - combined["old_weight"]
    combined["action"] = np.where(
        combined["new_weight"] > combined["old_weight"],
        "buy",
        np.where(combined["new_weight"] < combined["old_weight"], "sell", "hold"),
    )
    combined = combined.reset_index().rename(columns={"index": "ticker"})
    combined["date"] = pd.Timestamp(date)
    return combined[["date", "ticker", "old_weight", "new_weight", "weight_change", "action"]]


def backtest_engine(
    panel: pd.DataFrame,
    snapshots: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    selected_factors = resolve_signed_factor_selection(config.factor_cols)
    signed_panel = panel.copy()
    for factor in selected_factors:
        signed_panel[factor.display_name] = signed_panel[factor.source_factor] * factor.sign

    config = BacktestConfig(
        factor_cols=[factor.display_name for factor in selected_factors],
        rebalance_freq=config.rebalance_freq,
        long_quantile=config.long_quantile,
        short_quantile=config.short_quantile,
        commission_bps=config.commission_bps,
        weight_method=config.weight_method,
        ic_method=config.ic_method,
        universe_refresh_rule=config.universe_refresh_rule,
        price_col=config.price_col,
        label_col=config.label_col,
        ticker_col=config.ticker_col,
        date_col=config.date_col,
    )
    df = prepare_clean_panel(signed_panel, config)
    trading_dates = sorted(df[config.date_col].unique())
    universe_map = resolve_universe_by_date(snapshots, trading_dates)
    ic_history = build_ic_history(df, config)
    reweight_dates = monthly_reweight_dates(trading_dates)

    current_factor_weights = pd.Series(
        1.0 / len(config.factor_cols),
        index=config.factor_cols,
        dtype=float,
    )
    previous_positions = pd.Series(dtype=float)
    performance_records = []
    rebalance_records = []
    holdings_records = []
    trade_logs = []
    weight_history = []
    universe_records = []

    for i, today in enumerate(trading_dates):
        active_universe = universe_map[pd.Timestamp(today)]
        universe_records.append(
            {
                "date": pd.Timestamp(today),
                "universe_size": len(active_universe),
                "tickers": list(active_universe),
            }
        )

        if today in reweight_dates:
            end_idx = i - config.rebalance_freq
            start_idx = end_idx - 20
            if start_idx >= 0:
                ic_window = ic_history.iloc[start_idx : end_idx + 1]
                current_factor_weights = compute_weights(ic_window, method=config.weight_method)
            weight_history.append(
                {"date": pd.Timestamp(today), **current_factor_weights.to_dict()}
            )

        rebalance_cost = 0.0
        gross_return = 0.0
        target_positions = previous_positions

        if i % config.rebalance_freq == 0:
            target_positions = build_target_positions(
                df=df,
                trading_dates=trading_dates,
                date_index=i,
                current_weights=current_factor_weights,
                selected_factors=selected_factors,
                active_universe=active_universe,
                config=config,
            )
            turnover = calculate_turnover(previous_positions, target_positions)
            rebalance_cost = turnover * config.commission_bps / 10000.0
            trade_log = build_trade_log(today, previous_positions, target_positions)
            trade_logs.append(trade_log)

            added = sorted(set(target_positions.index) - set(previous_positions.index))
            removed = sorted(set(previous_positions.index) - set(target_positions.index))
            changed = trade_log.loc[trade_log["weight_change"] != 0, "ticker"].tolist()
            rebalance_records.append(
                {
                    "date": pd.Timestamp(today),
                    "holdings_count": int(target_positions.shape[0]),
                    "long_count": int((target_positions > 0).sum()),
                    "short_count": int((target_positions < 0).sum()),
                    "added": added,
                    "removed": removed,
                    "changed": changed,
                    "turnover": turnover,
                    "commission_cost": rebalance_cost,
                    "target_weights": target_positions.sort_index().to_dict(),
                    "previous_weights": previous_positions.sort_index().to_dict(),
                }
            )
            previous_positions = target_positions

        for ticker, weight in previous_positions.sort_index().items():
            holdings_records.append(
                {
                    "date": pd.Timestamp(today),
                    "ticker": ticker,
                    "weight": weight,
                    "side": "long" if weight > 0 else "short" if weight < 0 else "flat",
                }
            )

        day_rets = df[df[config.date_col] == today][[config.ticker_col, "ret_1d_next"]]
        pnl = previous_positions.rename("weight").reset_index().rename(
            columns={"index": config.ticker_col}
        ).merge(
            day_rets,
            on=config.ticker_col,
            how="left",
        )
        pnl["ret_1d_next"] = pnl["ret_1d_next"].fillna(0.0)
        if not pnl.empty:
            gross_return = float((pnl["weight"] * pnl["ret_1d_next"]).sum())

        net_return = gross_return - rebalance_cost
        performance_records.append(
            {
                "date": pd.Timestamp(today),
                "gross_ret": gross_return,
                "commission_cost": rebalance_cost,
                "daily_ret": net_return,
                "turnover": rebalance_records[-1]["turnover"] if rebalance_records and rebalance_records[-1]["date"] == pd.Timestamp(today) else 0.0,
            }
        )

    performance = pd.DataFrame(performance_records).set_index("date").sort_index()
    performance["cum_ret"] = (1 + performance["daily_ret"]).cumprod() - 1
    performance["gross_cum_ret"] = (1 + performance["gross_ret"]).cumprod() - 1

    factor_weight_history = pd.DataFrame(weight_history).set_index("date").sort_index() if weight_history else pd.DataFrame(columns=config.factor_cols)
    holdings_history = pd.DataFrame(holdings_records)
    rebalance_log = pd.DataFrame(rebalance_records).set_index("date").sort_index() if rebalance_records else pd.DataFrame(
        columns=["holdings_count", "long_count", "short_count", "added", "removed", "changed", "turnover", "commission_cost", "target_weights", "previous_weights"]
    )
    rebalance_trades = pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame(
        columns=["date", "ticker", "old_weight", "new_weight", "weight_change", "action"]
    )
    universe_history = pd.DataFrame(universe_records).set_index("date").sort_index()

    return BacktestResult(
        performance=performance,
        factor_weight_history=factor_weight_history,
        holdings_history=holdings_history,
        rebalance_log=rebalance_log,
        rebalance_trades=rebalance_trades,
        ic_history=ic_history,
        final_weights=current_factor_weights,
        universe_history=universe_history,
    )
