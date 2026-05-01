from __future__ import annotations

import pandas as pd
import pytest

from alpha_rotation.data import resolve_universe_by_date


def test_universe_switches_on_effective_date():
    snapshots = pd.DataFrame(
        [
            {"effective_date": "2024-01-01", "ticker": "AAA"},
            {"effective_date": "2024-01-01", "ticker": "BBB"},
            {"effective_date": "2024-07-01", "ticker": "BBB"},
            {"effective_date": "2024-07-01", "ticker": "CCC"},
        ]
    )
    snapshots["effective_date"] = pd.to_datetime(snapshots["effective_date"])

    resolved = resolve_universe_by_date(
        snapshots,
        [pd.Timestamp("2024-06-28"), pd.Timestamp("2024-07-01")],
    )

    assert resolved[pd.Timestamp("2024-06-28")] == ("AAA", "BBB")
    assert resolved[pd.Timestamp("2024-07-01")] == ("BBB", "CCC")


def test_universe_rejects_dates_before_first_snapshot():
    snapshots = pd.DataFrame(
        [
            {"effective_date": "2024-01-01", "ticker": "AAA"},
            {"effective_date": "2024-01-01", "ticker": "BBB"},
        ]
    )
    snapshots["effective_date"] = pd.to_datetime(snapshots["effective_date"])

    with pytest.raises(ValueError, match="First snapshot starts at 2024-01-01"):
        resolve_universe_by_date(snapshots, [pd.Timestamp("2023-12-31")])
