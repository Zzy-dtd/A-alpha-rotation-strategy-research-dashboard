from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_FACTOR_COLUMNS = [
    "mom_1m",
    "mom_3m",
    "ma_dist_20",
    "mom_accel",
    "rsi_14",
    "rev_5d",
    "rev_12m",
    "price_z_60",
    "realized_vol_20",
    "downside_vol_20",
    "atr_14",
    "volume_spike_20",
    "mfi_14",
    "pvt",
    "neg_mom_3m",
]

BASE_PANEL_COLUMNS = [
    "date",
    "ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "ret_1d",
    "fwd_ret_5d",
]


def calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_factor_panel(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy().sort_values(["date", "ticker"]).reset_index(drop=True)
    original_columns = df.columns.tolist()
    g = df.groupby("ticker", group_keys=False)

    df["ret_1d"] = g["Close"].pct_change()
    df["log_ret_1d"] = np.log(df["Close"] / g["Close"].shift(1))
    df["ret_5d"] = g["Close"].pct_change(5)
    df["ret_20d"] = g["Close"].pct_change(20)
    df["ret_60d"] = g["Close"].pct_change(60)
    df["ret_252d"] = g["Close"].pct_change(252)

    df["ma_20"] = g["Close"].transform(lambda x: x.rolling(20).mean())
    df["ma_60"] = g["Close"].transform(lambda x: x.rolling(60).mean())
    df["std_20"] = g["ret_1d"].transform(lambda x: x.rolling(20).std())
    df["std_60"] = g["ret_1d"].transform(lambda x: x.rolling(60).std())
    df["vol_ma_20"] = g["Volume"].transform(lambda x: x.rolling(20).mean())

    df["mom_1m"] = df["ret_20d"]
    df["mom_3m"] = df["ret_60d"]
    df["neg_mom_3m"] = -df["mom_3m"]
    df["ma_dist_20"] = (df["Close"] - df["ma_20"]) / df["ma_20"]
    df["mom_accel"] = df["ret_20d"] - df["ret_60d"]
    df["rsi_14"] = g["Close"].transform(lambda x: calc_rsi(x, 14))
    df["rev_5d"] = -df["ret_5d"]
    df["rev_12m"] = -df["ret_252d"]
    df["price_z_60"] = (df["Close"] - df["ma_60"]) / g["Close"].transform(
        lambda x: x.rolling(60).std()
    )
    df["realized_vol_20"] = df["std_20"]
    df["downside_vol_20"] = g["ret_1d"].transform(lambda x: x.clip(upper=0).rolling(20).std())

    prev_close = g["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = g["true_range"].transform(lambda x: x.rolling(14).mean())

    df["volume_spike_20"] = df["Volume"] / df["vol_ma_20"]
    df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["money_flow"] = df["typical_price"] * df["Volume"]

    tp_diff = g["typical_price"].diff()
    df["positive_mf"] = np.where(tp_diff > 0, df["money_flow"], 0.0)
    df["negative_mf"] = np.where(tp_diff < 0, df["money_flow"], 0.0)
    pos_mf_14 = g["positive_mf"].transform(lambda x: x.rolling(14).sum())
    neg_mf_14 = g["negative_mf"].transform(lambda x: x.rolling(14).sum())
    money_ratio = pos_mf_14 / neg_mf_14
    df["mfi_14"] = 100 - (100 / (1 + money_ratio))

    df["pvt_increment"] = df["ret_1d"] * df["Volume"]
    df["pvt"] = g["pvt_increment"].cumsum()
    df["fwd_ret_5d"] = g["Close"].pct_change(5).shift(-5)

    passthrough_columns = [
        column
        for column in original_columns
        if column not in BASE_PANEL_COLUMNS and column not in DEFAULT_FACTOR_COLUMNS
    ]
    ordered_columns = BASE_PANEL_COLUMNS + passthrough_columns + DEFAULT_FACTOR_COLUMNS
    return df[ordered_columns].copy()
