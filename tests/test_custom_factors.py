from __future__ import annotations

import pandas as pd

from alpha_rotation.custom_factors import (
    apply_custom_factors,
    evaluate_custom_factor_formula,
    validate_custom_factor_formula,
)


def make_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [11.0, 12.0, 13.0],
            "Low": [9.0, 10.0, 11.0],
            "Close": [10.5, 11.5, 12.5],
            "Volume": [100.0, 120.0, 140.0],
        }
    )


def test_evaluate_custom_factor_formula_supports_basic_arithmetic():
    panel = make_panel()
    result = evaluate_custom_factor_formula(panel, "(Close - Open) / Open", ["Open", "Close"])
    expected = (panel["Close"] - panel["Open"]) / panel["Open"]
    assert result.equals(expected.astype(float))


def test_validate_custom_factor_formula_rejects_unknown_column():
    panel = make_panel()
    validation = validate_custom_factor_formula(panel, "bad_factor", "Foo + Close", ["Close"])
    assert not validation.is_valid
    assert "Unknown column" in validation.message


def test_apply_custom_factors_adds_valid_factor_only():
    panel = make_panel()
    updated, validations = apply_custom_factors(
        panel,
        custom_factors={"spread": "Close - Open", "bad": "Foo + 1"},
        allowed_columns=["Open", "Close"],
    )
    assert "spread" in updated.columns
    assert "bad" not in updated.columns
    assert len(validations) == 2
