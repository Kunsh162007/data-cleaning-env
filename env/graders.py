"""
Deterministic graders for each task.
Every grader returns (score: float, breakdown: dict[str, float]).
All checks are inspections of the current dataset state — no gold label needed.
"""

import re
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------

_NULL_SENTINELS = {None, "N/A", "n/a", "", "null", "NULL", "None", "nan", "NaN"}


def _is_null(val: Any) -> bool:
    return val in _NULL_SENTINELS


def _has_currency(val: Any) -> bool:
    return isinstance(val, str) and any(c in val for c in ("$", "€", "£", "¥"))


def _is_valid_email(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", val.strip()))


def _is_iso_date(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", val.strip()))


# ---------------------------------------------------------------------------
# TASK EASY — Customer Records
# Checks (weights): no_null_ages(0.30) | no_duplicates(0.25)
#                   age_is_numeric(0.25) | no_name_whitespace(0.20)
# ---------------------------------------------------------------------------

def grade_easy(dataset: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}

    # 1. No null / NA ages
    null_ages = sum(1 for r in dataset if _is_null(r.get("age")))
    breakdown["no_null_ages"] = round(0.30 * max(0.0, 1.0 - null_ages / 3), 4)

    # 2. No duplicate rows
    seen: list = []
    dupes = 0
    for row in dataset:
        key = tuple(sorted((k, str(v)) for k, v in row.items()))
        if key in seen:
            dupes += 1
        else:
            seen.append(key)
    breakdown["no_duplicates"] = 0.25 if dupes == 0 else 0.0

    # 3. age column is integer / float (not string)
    valid_ages = [r.get("age") for r in dataset if not _is_null(r.get("age"))]
    numeric_count = sum(1 for v in valid_ages if isinstance(v, (int, float)))
    breakdown["age_is_numeric"] = (
        0.25 if valid_ages and numeric_count == len(valid_ages) else 0.0
    )

    # 4. No leading/trailing whitespace in name column
    ws_names = sum(
        1 for r in dataset
        if isinstance(r.get("name"), str) and r["name"] != r["name"].strip()
    )
    breakdown["no_name_whitespace"] = 0.20 if ws_names == 0 else 0.0

    score = min(1.0, sum(breakdown.values()))
    return round(score, 4), breakdown


# ---------------------------------------------------------------------------
# TASK MEDIUM — Product Inventory
# Checks: consistent_categories(0.25) | no_negative_prices(0.25)
#         no_null_stocks(0.25)        | no_null_product_codes(0.25)
# ---------------------------------------------------------------------------

def grade_medium(dataset: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}

    # 1. All categories in title case (consistent normalisation)
    categories = [r.get("category") for r in dataset if not _is_null(r.get("category"))]
    bad_case = sum(1 for c in categories if isinstance(c, str) and c != c.title())
    breakdown["consistent_categories"] = (
        0.25 if bad_case == 0 else round(0.25 * max(0.0, 1.0 - bad_case / len(categories)), 4)
    )

    # 2. No negative prices
    prices = [r.get("price") for r in dataset if not _is_null(r.get("price"))]
    neg_prices = sum(1 for p in prices if isinstance(p, (int, float)) and p < 0)
    breakdown["no_negative_prices"] = (
        0.25 if neg_prices == 0 else round(0.25 * max(0.0, 1.0 - neg_prices / 3), 4)
    )

    # 3. No null stocks (should be filled with 0)
    null_stocks = sum(1 for r in dataset if _is_null(r.get("stock")))
    breakdown["no_null_stocks"] = (
        0.25 if null_stocks == 0 else round(0.25 * max(0.0, 1.0 - null_stocks / 5), 4)
    )

    # 4. No null product codes
    null_codes = sum(1 for r in dataset if _is_null(r.get("product_code")))
    breakdown["no_null_product_codes"] = (
        0.25 if null_codes == 0 else round(0.25 * max(0.0, 1.0 - null_codes / 3), 4)
    )

    score = min(1.0, sum(breakdown.values()))
    return round(score, 4), breakdown


# ---------------------------------------------------------------------------
# TASK HARD — Orders
# Checks: iso_dates(0.20) | no_negative_qty(0.20) | valid_emails(0.20)
#         numeric_prices(0.20) | valid_discounts(0.20)
# ---------------------------------------------------------------------------

def grade_hard(dataset: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}
    n = len(dataset) if dataset else 1

    # 1. All dates in YYYY-MM-DD format
    bad_dates = sum(1 for r in dataset if not _is_iso_date(r.get("date")))
    # original has 12 non-iso dates out of 20
    breakdown["iso_dates"] = (
        0.20 if bad_dates == 0 else round(0.20 * max(0.0, 1.0 - bad_dates / 12), 4)
    )

    # 2. No negative quantities
    neg_qty = sum(
        1 for r in dataset
        if isinstance(r.get("quantity"), (int, float)) and r["quantity"] < 0
    )
    breakdown["no_negative_quantities"] = (
        0.20 if neg_qty == 0 else round(0.20 * max(0.0, 1.0 - neg_qty / 4), 4)
    )

    # 3. All emails valid (contain @, no spaces)
    invalid_emails = sum(1 for r in dataset if not _is_valid_email(r.get("customer_email")))
    breakdown["valid_emails"] = (
        0.20 if invalid_emails == 0 else round(0.20 * max(0.0, 1.0 - invalid_emails / 4), 4)
    )

    # 4. All prices are numeric floats (no currency symbols)
    currency_prices = sum(1 for r in dataset if _has_currency(r.get("price")))
    breakdown["numeric_prices"] = (
        0.20 if currency_prices == 0 else round(0.20 * max(0.0, 1.0 - currency_prices / n), 4)
    )

    # 5. All discount_pct in [0, 100]
    bad_discounts = sum(
        1 for r in dataset
        if isinstance(r.get("discount_pct"), (int, float))
        and not (0 <= r["discount_pct"] <= 100)
    )
    breakdown["valid_discounts"] = (
        0.20 if bad_discounts == 0 else round(0.20 * max(0.0, 1.0 - bad_discounts / 5), 4)
    )

    score = min(1.0, sum(breakdown.values()))
    return round(score, 4), breakdown


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def grade_task(task_id: str, dataset: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    graders = {
        "task_easy":   grade_easy,
        "task_medium": grade_medium,
        "task_hard":   grade_hard,
    }
    grader = graders.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id '{task_id}'")
    score, breakdown = grader(dataset)
    return _clamp(score), breakdown


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0.001, 0.999)."""
    return round(max(0.001, min(0.999, score)), 4)
