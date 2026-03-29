"""
Tests for the Data Cleaning Environment.
Run with: python -m pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from env import Action, DataCleaningEnv
from env.graders import grade_easy, grade_medium, grade_hard, grade_task
from env.datasets import get_dataset


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def test_dataset_loading():
    for tid in ("task_easy", "task_medium", "task_hard"):
        data = get_dataset(tid)
        assert isinstance(data, list)
        assert len(data) > 0

def test_dataset_is_fresh_copy():
    d1 = get_dataset("task_easy")
    d2 = get_dataset("task_easy")
    d1[0]["name"] = "MUTATED"
    assert d2[0]["name"] != "MUTATED"


# ---------------------------------------------------------------------------
# Graders on untouched (dirty) data — must not be 1.0
# ---------------------------------------------------------------------------

def test_easy_initial_score_below_1():
    score, _ = grade_easy(get_dataset("task_easy"))
    assert 0.0 <= score < 1.0

def test_medium_initial_score_below_1():
    score, _ = grade_medium(get_dataset("task_medium"))
    assert 0.0 <= score < 1.0

def test_hard_initial_score_below_1():
    score, _ = grade_hard(get_dataset("task_hard"))
    assert 0.0 <= score < 1.0


# ---------------------------------------------------------------------------
# Graders on fully cleaned data — must return 1.0
# ---------------------------------------------------------------------------

def _clean_easy():
    """Return a perfectly clean version of the easy dataset."""
    return [
        {"name": "Alice",   "age": 28, "email": "alice@email.com",   "city": "New York"},
        {"name": "Charlie", "age": 35, "email": "charlie@email.com", "city": "Paris"},
        {"name": "Eve",     "age": 22, "email": "eve@email.com",     "city": "Sydney"},
        {"name": "Frank",   "age": 31, "email": "frank@email.com",   "city": "Berlin"},
        {"name": "Henry",   "age": 45, "email": "henry@email.com",   "city": "Madrid"},
        {"name": "Iris",    "age": 29, "email": "iris@email.com",    "city": "Amsterdam"},
    ]

def test_easy_perfect_score():
    score, breakdown = grade_easy(_clean_easy())
    assert score == 1.0, f"Expected 1.0, got {score}. Breakdown: {breakdown}"

def test_easy_breakdown_keys():
    _, breakdown = grade_easy(get_dataset("task_easy"))
    assert set(breakdown) == {"no_null_ages", "no_duplicates", "age_is_numeric", "no_name_whitespace"}

def test_medium_breakdown_keys():
    _, breakdown = grade_medium(get_dataset("task_medium"))
    assert set(breakdown) == {
        "consistent_categories", "no_negative_prices", "no_null_stocks", "no_null_product_codes"
    }

def test_hard_breakdown_keys():
    _, breakdown = grade_hard(get_dataset("task_hard"))
    assert set(breakdown) == {
        "iso_dates", "no_negative_quantities", "valid_emails",
        "numeric_prices", "valid_discounts",
    }


# ---------------------------------------------------------------------------
# Environment API
# ---------------------------------------------------------------------------

def test_reset_returns_step_result():
    env = DataCleaningEnv()
    result = env.reset("task_easy")
    assert result.done is False
    assert result.observation.step == 0
    assert result.observation.task_id == "task_easy"
    assert result.observation.score >= 0.0

def test_reset_resets_state():
    env = DataCleaningEnv()
    env.reset("task_easy")
    env.step(Action(action_type="remove_duplicates"))
    env.reset("task_easy")
    assert env.step_count == 0
    assert env.dataset == get_dataset("task_easy")

def test_step_increments_counter():
    env = DataCleaningEnv()
    env.reset("task_easy")
    result = env.step(Action(action_type="remove_duplicates"))
    assert result.observation.step == 1

def test_step_invalid_action():
    env = DataCleaningEnv()
    env.reset("task_easy")
    result = env.step(Action(action_type="nonexistent_action"))
    assert result.reward.value == pytest.approx(-0.05)

def test_step_done_action():
    env = DataCleaningEnv()
    env.reset("task_easy")
    result = env.step(Action(action_type="done"))
    assert result.done is True

def test_state_endpoint():
    env = DataCleaningEnv()
    env.reset("task_medium")
    s = env.state()
    assert s.task_id == "task_medium"
    assert s.step == 0
    assert s.done is False

def test_step_after_done():
    env = DataCleaningEnv()
    env.reset("task_easy")
    env.step(Action(action_type="done"))
    result = env.step(Action(action_type="remove_duplicates"))
    assert result.done is True
    assert result.reward.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Action correctness
# ---------------------------------------------------------------------------

def test_remove_duplicates():
    env = DataCleaningEnv()
    env.reset("task_easy")
    before = len(env.dataset)
    env.step(Action(action_type="remove_duplicates"))
    assert len(env.dataset) < before

def test_fill_nulls_zero():
    env = DataCleaningEnv()
    env.reset("task_medium")
    env.step(Action(action_type="fill_nulls", params={"column": "stock", "strategy": "zero"}))
    null_stocks = sum(1 for r in env.dataset if r.get("stock") is None)
    assert null_stocks == 0

def test_fix_type_age():
    env = DataCleaningEnv()
    env.reset("task_easy")
    env.step(Action(action_type="drop_nulls", params={"column": "age"}))
    env.step(Action(action_type="fix_type", params={"column": "age", "dtype": "int"}))
    valid_ages = [r["age"] for r in env.dataset if r.get("age") is not None]
    assert all(isinstance(v, int) for v in valid_ages)

def test_standardize_dates():
    env = DataCleaningEnv()
    env.reset("task_hard")
    env.step(Action(action_type="standardize_dates", params={"column": "date"}))
    import re
    bad = [r["date"] for r in env.dataset if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(r.get("date", "")))]
    assert len(bad) == 0

def test_strip_currency():
    env = DataCleaningEnv()
    env.reset("task_hard")
    env.step(Action(action_type="strip_currency", params={"column": "price"}))
    currency_remaining = [r["price"] for r in env.dataset if isinstance(r["price"], str) and any(c in r["price"] for c in "$€£¥")]
    assert currency_remaining == []

def test_normalize_column_auto_titlecase():
    env = DataCleaningEnv()
    env.reset("task_medium")
    env.step(Action(action_type="normalize_column", params={"column": "category"}))
    cats = [r["category"] for r in env.dataset]
    for c in cats:
        assert c == c.title(), f"'{c}' is not title-case"

def test_score_improves_after_cleaning():
    env = DataCleaningEnv()
    result = env.reset("task_easy")
    initial_score = result.observation.score
    env.step(Action(action_type="remove_duplicates"))
    env.step(Action(action_type="drop_nulls", params={"column": "age"}))
    env.step(Action(action_type="fix_type", params={"column": "age", "dtype": "int"}))
    result = env.step(Action(action_type="strip_whitespace", params={"column": "name"}))
    assert result.observation.score > initial_score


# ---------------------------------------------------------------------------
# Task dispatcher
# ---------------------------------------------------------------------------

def test_grade_task_dispatcher():
    for tid in ("task_easy", "task_medium", "task_hard"):
        score, breakdown = grade_task(tid, get_dataset(tid))
        assert 0.0 <= score <= 1.0
        assert isinstance(breakdown, dict)
        assert all(0.0 <= v <= 1.0 for v in breakdown.values())

def test_grade_task_unknown_raises():
    with pytest.raises(ValueError):
        grade_task("task_unknown", [])
