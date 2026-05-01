from __future__ import annotations

import ast
from dataclasses import dataclass

import pandas as pd


ALLOWED_BINARY_OPS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
}

ALLOWED_UNARY_OPS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


@dataclass(frozen=True)
class CustomFactorValidationResult:
    name: str
    formula: str
    is_valid: bool
    message: str
    preview: pd.Series | None = None


def validate_custom_factor_name(name: str) -> None:
    if not name or not name.strip():
        raise ValueError("Custom factor name cannot be empty.")
    if name.startswith("inv_"):
        raise ValueError("Custom factor names cannot start with 'inv_'.")
    if not name.replace("_", "").isalnum():
        raise ValueError("Custom factor name must contain only letters, numbers, and underscores.")


def evaluate_custom_factor_formula(
    panel: pd.DataFrame,
    formula: str,
    allowed_columns: list[str],
) -> pd.Series:
    expression = ast.parse(formula, mode="eval")
    allowed = set(allowed_columns)

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Name):
            if node.id not in allowed:
                raise ValueError(f"Unknown column in formula: {node.id}")
            return panel[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINARY_OPS:
            return ALLOWED_BINARY_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY_OPS:
            return ALLOWED_UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Formula contains unsupported syntax. Only +, -, *, /, parentheses, and column names are allowed.")

    result = _eval(expression)
    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=panel.index, dtype=float)
    return result.astype(float)


def validate_custom_factor_formula(
    panel: pd.DataFrame,
    name: str,
    formula: str,
    allowed_columns: list[str],
) -> CustomFactorValidationResult:
    try:
        validate_custom_factor_name(name)
        preview = evaluate_custom_factor_formula(panel, formula, allowed_columns)
        return CustomFactorValidationResult(
            name=name,
            formula=formula,
            is_valid=True,
            message="Formula is valid.",
            preview=preview.head(5),
        )
    except Exception as exc:
        return CustomFactorValidationResult(
            name=name,
            formula=formula,
            is_valid=False,
            message=str(exc),
            preview=None,
        )


def apply_custom_factors(
    panel: pd.DataFrame,
    custom_factors: dict[str, str],
    allowed_columns: list[str],
) -> tuple[pd.DataFrame, list[CustomFactorValidationResult]]:
    updated = panel.copy()
    validations: list[CustomFactorValidationResult] = []
    available_columns = list(dict.fromkeys(allowed_columns + list(updated.columns)))

    for name, formula in custom_factors.items():
        validation = validate_custom_factor_formula(updated, name, formula, available_columns)
        validations.append(validation)
        if validation.is_valid:
            updated[name] = evaluate_custom_factor_formula(updated, formula, available_columns)
            available_columns.append(name)

    return updated, validations
