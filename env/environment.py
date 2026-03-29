"""
Core DataCleaningEnv — implements the OpenEnv step/reset/state interface.

Supported actions
-----------------
drop_nulls         {"column": str}
fill_nulls         {"column": str, "value": any}  | {"column": str, "strategy": "zero"|"mean"|"unknown"}
fix_type           {"column": str, "dtype": "int"|"float"|"str"}
normalize_column   {"column": str, "mapping": {old: new, ...}}
remove_duplicates  {}  |  {"subset": [col, ...]}
clip_outliers      {"column": str, "lower": num, "upper": num}
strip_whitespace   {"column": str}  |  {}  (all string columns)
filter_rows        {"column": str, "operator": ">="|"<="|">"|"<"|"=="|"!=", "value": any}
standardize_dates  {"column": str}   (→ YYYY-MM-DD)
strip_currency     {"column": str}   (removes $€£¥, converts to float)
done               {}
"""

import copy
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .datasets import get_dataset
from .graders import grade_task
from .models import Action, EnvState, Observation, Reward, StepResult
from .tasks import (
    MAX_STEPS,
    TASK_DESCRIPTIONS,
    detect_issues,
    get_column_info,
)

_NULL_SENTINELS = {None, "N/A", "n/a", "", "null", "NULL", "None", "nan", "NaN"}

_DATE_FORMATS = [
    "%Y-%m-%d",   # 2024-01-15  ← target
    "%m/%d/%Y",   # 01/15/2024
    "%d/%m/%Y",   # 15/01/2024
    "%b %d %Y",   # Jan 15 2024
    "%b %d, %Y",  # Jan 15, 2024
    "%B %d %Y",   # January 15 2024
    "%d-%m-%Y",   # 15-01-2024
]


def _is_null(val: Any) -> bool:
    return val in _NULL_SENTINELS


class DataCleaningEnv:
    """Data Cleaning & Validation reinforcement-learning environment."""

    def __init__(self) -> None:
        self.task_id: Optional[str] = None
        self.dataset: List[Dict[str, Any]] = []
        self.step_count: int = 0
        self._done: bool = False
        self._prev_score: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task_easy") -> StepResult:
        """Initialise (or re-initialise) the environment for a given task."""
        if task_id not in MAX_STEPS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(MAX_STEPS)}")

        self.task_id = task_id
        self.dataset = get_dataset(task_id)
        self.step_count = 0
        self._done = False
        self._prev_score, _ = grade_task(task_id, self.dataset)

        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=Reward(value=0.0, breakdown={}, message="Episode started — good luck!"),
            done=False,
            info={"task_id": task_id},
        )

    def step(self, action: Action) -> StepResult:
        """Apply an action and return the next observation + reward."""
        if self._done:
            obs = self._make_observation()
            return StepResult(
                observation=obs,
                reward=Reward(value=0.0, breakdown={}, message="Episode already finished."),
                done=True,
                info={"error": "Call reset() to start a new episode."},
            )

        self.step_count += 1
        max_s = MAX_STEPS[self.task_id]

        # Execute the action
        action_info = self._execute_action(action)

        # Score after action
        new_score, breakdown = grade_task(self.task_id, self.dataset)
        delta = new_score - self._prev_score

        # Reward shaping
        if action_info.get("error"):
            reward_val = -0.05
            msg = f"Invalid action ({action_info['error']})"
        elif action.action_type == "done":
            self._done = True
            # Bonus if agent finishes with high score; small penalty if quitting early
            bonus = 0.10 if new_score >= 0.80 else -0.05
            reward_val = round(new_score + bonus, 4)
            msg = f"Agent called done. Final score: {new_score:.4f}"
        else:
            # Delta reward minus tiny step cost → encourages efficiency
            reward_val = round(delta - 0.01, 4)
            msg = action_info.get("message", f"Executed {action.action_type}")

        self._prev_score = new_score

        # Hard time-limit
        if self.step_count >= max_s and not self._done:
            self._done = True
            msg += f" | Max steps ({max_s}) reached."

        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=Reward(value=reward_val, breakdown=breakdown, message=msg),
            done=self._done,
            info=action_info,
        )

    def state(self) -> EnvState:
        """Return a lightweight snapshot of the current episode state."""
        score, _ = (
            grade_task(self.task_id, self.dataset)
            if self.task_id
            else (0.0, {})
        )
        issues = detect_issues(self.dataset, self.task_id) if self.task_id else []
        return EnvState(
            task_id=self.task_id,
            step=self.step_count,
            score=score,
            done=self._done,
            dataset_size=len(self.dataset),
            issues_count=len(issues),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        issues = detect_issues(self.dataset, self.task_id)
        col_info = get_column_info(self.dataset)
        score, _ = grade_task(self.task_id, self.dataset)
        return Observation(
            task_id=self.task_id,
            task_description=TASK_DESCRIPTIONS.get(self.task_id, ""),
            dataset=self.dataset,
            column_info=col_info,
            issues_detected=issues,
            step=self.step_count,
            max_steps=MAX_STEPS.get(self.task_id, 15),
            score=score,
        )

    def _execute_action(self, action: Action) -> Dict[str, Any]:
        handlers = {
            "drop_nulls":        self._drop_nulls,
            "fill_nulls":        self._fill_nulls,
            "fix_type":          self._fix_type,
            "normalize_column":  self._normalize_column,
            "remove_duplicates": self._remove_duplicates,
            "clip_outliers":     self._clip_outliers,
            "strip_whitespace":  self._strip_whitespace,
            "filter_rows":       self._filter_rows,
            "standardize_dates": self._standardize_dates,
            "strip_currency":    self._strip_currency,
            "drop_invalid":      self._drop_invalid,
            "done":              lambda _p: {"message": "Agent signalled completion"},
        }
        handler = handlers.get(action.action_type)
        if handler is None:
            return {"error": f"Unknown action '{action.action_type}'"}
        try:
            return handler(action.params)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _drop_nulls(self, params: Dict) -> Dict:
        column = params.get("column")
        if column and column not in (self.dataset[0].keys() if self.dataset else []):
            return {"error": f"Column '{column}' not found"}
        before = len(self.dataset)
        if column:
            self.dataset = [r for r in self.dataset if not _is_null(r.get(column))]
        else:
            self.dataset = [
                r for r in self.dataset
                if not any(_is_null(v) for v in r.values())
            ]
        dropped = before - len(self.dataset)
        return {"message": f"Dropped {dropped} row(s) with null values"}

    def _fill_nulls(self, params: Dict) -> Dict:
        column = params.get("column")
        value = params.get("value")
        strategy = params.get("strategy", "")
        if column and column not in (self.dataset[0].keys() if self.dataset else []):
            return {"error": f"Column '{column}' not found"}

        # Compute fill value from strategy
        if strategy:
            col_vals = [
                r.get(column) for r in self.dataset
                if column and not _is_null(r.get(column)) and isinstance(r.get(column), (int, float))
            ]
            if strategy == "mean" and col_vals:
                value = sum(col_vals) / len(col_vals)
            elif strategy == "zero":
                value = 0
            elif strategy == "unknown":
                value = "Unknown"
            else:
                value = 0

        filled = 0
        for row in self.dataset:
            if column:
                if _is_null(row.get(column)):
                    row[column] = value
                    filled += 1
            else:
                for col in list(row.keys()):
                    if _is_null(row[col]):
                        row[col] = value
                        filled += 1
        return {"message": f"Filled {filled} null value(s) with {value!r}"}

    def _fix_type(self, params: Dict) -> Dict:
        column = params.get("column")
        dtype = params.get("dtype", "str")
        if not column:
            return {"error": "Parameter 'column' is required"}
        if column not in (self.dataset[0].keys() if self.dataset else []):
            return {"error": f"Column '{column}' not found"}

        converters = {"int": int, "float": float, "str": str}
        conv = converters.get(dtype)
        if conv is None:
            return {"error": f"Unknown dtype '{dtype}'. Use int, float, or str"}

        converted, errors = 0, 0
        for row in self.dataset:
            val = row.get(column)
            if _is_null(val):
                continue
            try:
                # Strip whitespace from strings before numeric conversion
                if dtype in ("int", "float") and isinstance(val, str):
                    val = val.strip()
                row[column] = conv(val)
                converted += 1
            except (ValueError, TypeError):
                errors += 1

        return {"message": f"Converted {converted} value(s) to {dtype}; {errors} failed"}

    def _normalize_column(self, params: Dict) -> Dict:
        column = params.get("column")
        mapping = params.get("mapping", {})
        if not column:
            return {"error": "Parameter 'column' is required"}
        if column not in (self.dataset[0].keys() if self.dataset else []):
            return {"error": f"Column '{column}' not found"}
        if not mapping:
            # Auto title-case for string columns
            changed = 0
            for row in self.dataset:
                val = row.get(column)
                if isinstance(val, str) and not _is_null(val):
                    new_val = val.title()
                    if new_val != val:
                        row[column] = new_val
                        changed += 1
            return {"message": f"Title-cased {changed} value(s) in '{column}'"}

        changed = 0
        for row in self.dataset:
            val = row.get(column)
            if val in mapping:
                row[column] = mapping[val]
                changed += 1
            elif isinstance(val, str) and val.lower() in {k.lower(): v for k, v in mapping.items()}:
                lower_map = {k.lower(): v for k, v in mapping.items()}
                row[column] = lower_map[val.lower()]
                changed += 1
        return {"message": f"Mapped {changed} value(s) in '{column}'"}

    def _remove_duplicates(self, params: Dict) -> Dict:
        subset = params.get("subset", [])
        before = len(self.dataset)
        seen: list = []
        unique: List[Dict] = []
        for row in self.dataset:
            if subset:
                key = tuple((k, str(row.get(k))) for k in subset)
            else:
                key = tuple(sorted((k, str(v)) for k, v in row.items()))
            if key not in seen:
                seen.append(key)
                unique.append(row)
        self.dataset = unique
        removed = before - len(self.dataset)
        return {"message": f"Removed {removed} duplicate row(s)"}

    def _clip_outliers(self, params: Dict) -> Dict:
        column = params.get("column")
        lower = params.get("lower")
        upper = params.get("upper")
        if not column:
            return {"error": "Parameter 'column' is required"}
        if column not in (self.dataset[0].keys() if self.dataset else []):
            return {"error": f"Column '{column}' not found"}

        clipped = 0
        for row in self.dataset:
            val = row.get(column)
            if not isinstance(val, (int, float)):
                continue
            new_val = val
            if lower is not None and val < lower:
                new_val = lower
            if upper is not None and val > upper:
                new_val = upper
            if new_val != val:
                row[column] = new_val
                clipped += 1
        return {"message": f"Clipped {clipped} value(s) in '{column}' to [{lower}, {upper}]"}

    def _strip_whitespace(self, params: Dict) -> Dict:
        column = params.get("column")
        stripped = 0
        cols = [column] if column else (list(self.dataset[0].keys()) if self.dataset else [])
        for row in self.dataset:
            for col in cols:
                val = row.get(col)
                if isinstance(val, str):
                    new_val = val.strip()
                    if new_val != val:
                        row[col] = new_val
                        stripped += 1
        return {"message": f"Stripped whitespace from {stripped} value(s)"}

    def _filter_rows(self, params: Dict) -> Dict:
        column = params.get("column")
        operator = params.get("operator", ">=")
        value = params.get("value")
        if not column:
            return {"error": "Parameter 'column' is required"}
        if column not in (self.dataset[0].keys() if self.dataset else []):
            return {"error": f"Column '{column}' not found"}

        ops = {
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            ">":  lambda a, b: a > b,
            "<":  lambda a, b: a < b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        op_fn = ops.get(operator)
        if op_fn is None:
            return {"error": f"Unknown operator '{operator}'"}

        before = len(self.dataset)
        self.dataset = [
            r for r in self.dataset
            if isinstance(r.get(column), (int, float)) and op_fn(r[column], value)
        ]
        removed = before - len(self.dataset)
        return {"message": f"Filtered out {removed} row(s) where '{column}' {operator} {value} was False"}

    def _standardize_dates(self, params: Dict) -> Dict:
        column = params.get("column", "date")
        if column not in (self.dataset[0].keys() if self.dataset else []):
            return {"error": f"Column '{column}' not found"}

        converted, already_good, failed = 0, 0, 0
        for row in self.dataset:
            val = row.get(column)
            if _is_null(val):
                continue
            val_str = str(val).strip()
            if re.match(r"^\d{4}-\d{2}-\d{2}$", val_str):
                already_good += 1
                continue
            parsed = None
            for fmt in _DATE_FORMATS:
                try:
                    parsed = datetime.strptime(val_str, fmt)
                    break
                except ValueError:
                    pass
            if parsed:
                row[column] = parsed.strftime("%Y-%m-%d")
                converted += 1
            else:
                failed += 1

        return {
            "message": (
                f"Standardized {converted} date(s); "
                f"{already_good} already in ISO format; "
                f"{failed} could not be parsed"
            )
        }

    def _strip_currency(self, params: Dict) -> Dict:
        column = params.get("column", "price")
        if column not in (self.dataset[0].keys() if self.dataset else []):
            return {"error": f"Column '{column}' not found"}

        stripped, failed = 0, 0
        for row in self.dataset:
            val = row.get(column)
            if not isinstance(val, str):
                continue
            cleaned = re.sub(r"[$€£¥\s]", "", val).strip()
            try:
                row[column] = float(cleaned)
                stripped += 1
            except ValueError:
                failed += 1

        return {"message": f"Stripped currency symbols from {stripped} value(s); {failed} failed"}

    def _drop_invalid(self, params: Dict) -> Dict:
        """Drop rows where a column fails a named validation rule.

        Supported rules:
          email     — must match x@y.z
          positive  — must be > 0
          iso_date  — must match YYYY-MM-DD
        """
        column = params.get("column")
        rule = params.get("rule", "")
        if not column:
            return {"error": "Parameter 'column' is required"}
        if column not in (self.dataset[0].keys() if self.dataset else []):
            return {"error": f"Column '{column}' not found"}

        rules = {
            "email":    lambda v: bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", str(v).strip())),
            "positive": lambda v: isinstance(v, (int, float)) and v > 0,
            "iso_date": lambda v: bool(re.match(r"^\d{4}-\d{2}-\d{2}$", str(v).strip())),
        }
        validator = rules.get(rule)
        if validator is None:
            return {"error": f"Unknown rule '{rule}'. Use: email, positive, iso_date"}

        before = len(self.dataset)
        self.dataset = [
            row for row in self.dataset
            if not _is_null(row.get(column)) and validator(row.get(column))
        ]
        dropped = before - len(self.dataset)
        return {"message": f"Dropped {dropped} row(s) failing '{rule}' check on '{column}'"}
