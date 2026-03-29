"""
Task descriptions and shared utility functions used across the environment.
"""

import re
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_DESCRIPTIONS: Dict[str, str] = {
    "task_easy": (
        "Clean a small customer records dataset. "
        "Fix age values stored as strings, fill or drop rows with 'N/A' ages, "
        "remove duplicate rows, and strip whitespace from names."
    ),
    "task_medium": (
        "Normalize a product inventory dataset. "
        "Standardize inconsistent category casing (Electronics/electronics/ELECTRONICS → title case), "
        "remove negative prices, fill null stock values with 0, "
        "and generate placeholder product codes for missing entries."
    ),
    "task_hard": (
        "Validate and repair a multi-issue orders dataset. "
        "Standardize mixed date formats to YYYY-MM-DD, remove negative quantities, "
        "fix invalid customer emails, strip currency symbols from prices, "
        "and clamp out-of-range discount percentages to [0, 100]."
    ),
}

TASK_DIFFICULTIES: Dict[str, str] = {
    "task_easy":   "easy",
    "task_medium": "medium",
    "task_hard":   "hard",
}

MAX_STEPS: Dict[str, int] = {
    "task_easy":   10,
    "task_medium": 15,
    "task_hard":   20,
}

# ---------------------------------------------------------------------------
# Issue detection
# ---------------------------------------------------------------------------

_NULL_SENTINELS = {None, "N/A", "n/a", "", "null", "NULL", "None", "nan", "NaN"}


def _is_null(val: Any) -> bool:
    return val in _NULL_SENTINELS


def detect_issues(dataset: List[Dict[str, Any]], task_id: str) -> List[str]:
    """Inspect the current dataset and return a list of human-readable issues."""
    if not dataset:
        return ["Dataset is empty"]

    issues: List[str] = []
    columns = list(dataset[0].keys())

    for col in columns:
        values = [row.get(col) for row in dataset]
        non_null = [v for v in values if not _is_null(v)]

        # Null / missing values
        null_count = len(values) - len(non_null)
        if null_count > 0:
            issues.append(f"Column '{col}': {null_count} null/missing value(s)")

        # Numeric values stored as strings
        if all(isinstance(v, str) for v in non_null):
            numeric_strings = [
                v for v in non_null
                if re.match(r"^-?\d+(\.\d+)?$", str(v))
            ]
            if len(numeric_strings) == len(non_null) and non_null:
                issues.append(f"Column '{col}': numeric data stored as strings")

        # Negative numbers
        numeric_vals = [v for v in non_null if isinstance(v, (int, float))]
        neg_count = sum(1 for v in numeric_vals if v < 0)
        if neg_count > 0:
            issues.append(f"Column '{col}': {neg_count} negative value(s)")

        # Leading/trailing whitespace in strings
        ws_count = sum(
            1 for v in non_null if isinstance(v, str) and v != v.strip()
        )
        if ws_count > 0:
            issues.append(f"Column '{col}': {ws_count} value(s) with leading/trailing whitespace")

        # Inconsistent casing for categorical-looking columns
        if col == "category" or (
            all(isinstance(v, str) for v in non_null)
            and len(set(str(v).lower() for v in non_null)) < len(set(str(v) for v in non_null))
        ):
            if len(non_null) > 0:
                lower_set = set(str(v).lower() for v in non_null)
                exact_set = set(str(v) for v in non_null)
                if len(exact_set) > len(lower_set):
                    issues.append(
                        f"Column '{col}': inconsistent casing "
                        f"({len(exact_set)} variants for {len(lower_set)} unique value(s))"
                    )

    # Duplicate rows
    seen: list = []
    dupes = 0
    for row in dataset:
        key = tuple(sorted((k, str(v)) for k, v in row.items()))
        if key in seen:
            dupes += 1
        else:
            seen.append(key)
    if dupes > 0:
        issues.append(f"{dupes} duplicate row(s) detected")

    # Task-specific checks
    if task_id == "task_hard":
        date_col = [row.get("date") for row in dataset if row.get("date")]
        bad_dates = sum(
            1 for d in date_col
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(d))
        )
        if bad_dates > 0:
            issues.append(f"Column 'date': {bad_dates} non-ISO date format(s) (expected YYYY-MM-DD)")

        currency_prices = sum(
            1 for row in dataset
            if isinstance(row.get("price"), str)
            and any(c in row["price"] for c in ("$", "€", "£", "¥"))
        )
        if currency_prices > 0:
            issues.append(f"Column 'price': {currency_prices} value(s) contain currency symbols")

        invalid_emails = sum(
            1 for row in dataset
            if row.get("customer_email")
            and not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", str(row["customer_email"]).strip())
        )
        if invalid_emails > 0:
            issues.append(f"Column 'customer_email': {invalid_emails} invalid email address(es)")

        bad_discounts = sum(
            1 for row in dataset
            if isinstance(row.get("discount_pct"), (int, float))
            and not (0 <= row["discount_pct"] <= 100)
        )
        if bad_discounts > 0:
            issues.append(f"Column 'discount_pct': {bad_discounts} value(s) outside [0, 100]")

    return issues if issues else ["No issues detected — dataset looks clean!"]


# ---------------------------------------------------------------------------
# Column info helper
# ---------------------------------------------------------------------------

def get_column_info(dataset: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return per-column stats used in the Observation."""
    if not dataset:
        return {}

    columns = list(dataset[0].keys())
    info: Dict[str, Dict[str, Any]] = {}

    for col in columns:
        values = [row.get(col) for row in dataset]
        non_null = [v for v in values if not _is_null(v)]

        null_count = len(values) - len(non_null)
        unique_vals = list(dict.fromkeys(str(v) for v in non_null))  # preserve order

        # dtype detection
        if all(isinstance(v, int) for v in non_null):
            dtype = "int"
        elif all(isinstance(v, float) for v in non_null):
            dtype = "float"
        elif all(isinstance(v, (int, float)) for v in non_null):
            dtype = "numeric"
        else:
            dtype = "str"

        info[col] = {
            "dtype": dtype,
            "null_count": null_count,
            "unique_count": len(unique_vals),
            "sample_values": unique_vals[:6],
        }

    return info
