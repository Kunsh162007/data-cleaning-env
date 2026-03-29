"""
Typed Pydantic models for the Data Cleaning Environment.
Implements the OpenEnv observation/action/reward interface.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Action(BaseModel):
    """
    Agent action: a named cleaning operation plus optional parameters.

    Examples
    --------
    {"action_type": "remove_duplicates", "params": {}}
    {"action_type": "fill_nulls", "params": {"column": "stock", "value": 0}}
    {"action_type": "fix_type",   "params": {"column": "age",  "dtype": "int"}}
    """

    action_type: str = Field(..., description="Name of the cleaning operation")
    params: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")


class Observation(BaseModel):
    """Full view of the environment returned after every reset / step."""

    task_id: str = Field(..., description="Active task identifier")
    task_description: str = Field(..., description="Human-readable task goal")
    dataset: List[Dict[str, Any]] = Field(..., description="Current dataset rows as dicts")
    column_info: Dict[str, Dict[str, Any]] = Field(
        ..., description="Per-column stats: dtype, null_count, unique_count, sample_values"
    )
    issues_detected: List[str] = Field(
        ..., description="Automatically-detected data quality issues"
    )
    step: int = Field(..., description="Current step number (0 = after reset)")
    max_steps: int = Field(..., description="Maximum steps allowed for this task")
    score: float = Field(..., description="Current grader score in [0.0, 1.0]")


class Reward(BaseModel):
    """Reward signal returned alongside every step."""

    value: float = Field(..., description="Scalar reward for this step")
    breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Per-check scores from the grader"
    )
    message: str = Field(..., description="Human-readable explanation of the reward")


class StepResult(BaseModel):
    """Complete return value of step() and reset()."""

    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class EnvState(BaseModel):
    """Lightweight snapshot returned by state()."""

    task_id: Optional[str]
    step: int
    score: float
    done: bool
    dataset_size: int
    issues_count: int
