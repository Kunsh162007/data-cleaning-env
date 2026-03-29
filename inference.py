"""
Inference Script — Data Cleaning Environment
=============================================

Mandatory environment variables
--------------------------------
API_BASE_URL  : LLM API base URL  (default: https://router.huggingface.co/v1)
MODEL_NAME    : Model identifier
HF_TOKEN      : Hugging Face / API key

Usage
-----
    python inference.py

The script runs a single episode for each of the three tasks, prints step-by-step
progress, and reports final grader scores.  Total wall-time is well under 20 min.
"""

import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment import (direct — no HTTP overhead)
# ---------------------------------------------------------------------------
from env import Action, DataCleaningEnv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME: str   = os.getenv("MODEL_NAME", "")

MAX_STEPS_PER_TASK = 12   # Hard cap per episode (env may end earlier)
TEMPERATURE        = 0.1
MAX_TOKENS         = 256
HISTORY_WINDOW     = 5    # Last N actions shown to model

FALLBACK_ACTION = {"action_type": "done", "params": {}}

TASKS = ["task_easy", "task_medium", "task_hard"]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data cleaning agent. You receive information about a messy dataset
and must apply cleaning operations one at a time to fix all detected issues.

Available actions (respond with ONLY a valid JSON object — no markdown, no prose):

{"action_type": "remove_duplicates",  "params": {}}
{"action_type": "drop_nulls",         "params": {"column": "<col>"}}
{"action_type": "fill_nulls",         "params": {"column": "<col>", "value": <val>}}
{"action_type": "fill_nulls",         "params": {"column": "<col>", "strategy": "zero"}}
{"action_type": "fix_type",           "params": {"column": "<col>", "dtype": "int"}}
{"action_type": "fix_type",           "params": {"column": "<col>", "dtype": "float"}}
{"action_type": "normalize_column",   "params": {"column": "<col>", "mapping": {"old": "new"}}}
{"action_type": "normalize_column",   "params": {"column": "<col>"}}   ← auto title-case
{"action_type": "clip_outliers",      "params": {"column": "<col>", "lower": 0, "upper": 9999}}
{"action_type": "strip_whitespace",   "params": {"column": "<col>"}}
{"action_type": "standardize_dates",  "params": {"column": "<col>"}}
{"action_type": "strip_currency",     "params": {"column": "<col>"}}
{"action_type": "filter_rows",        "params": {"column": "<col>", "operator": ">=", "value": 0}}
{"action_type": "drop_invalid",       "params": {"column": "<col>", "rule": "email"}}   ← rules: email | positive | iso_date
{"action_type": "done",               "params": {}}   ← call when dataset is clean

Rules:
- Output ONLY the JSON object, nothing else.
- Do not include markdown fences or extra text.
- Call "done" when all issues are resolved or you cannot improve further.
""").strip()


def build_user_prompt(
    step: int,
    obs: Any,
    history: List[str],
) -> str:
    # Show a compact dataset sample (first 4 rows max)
    sample = obs.dataset[:4]
    sample_str = json.dumps(sample, indent=2)

    col_info_str = json.dumps(obs.column_info, indent=2)

    issues_str = "\n".join(f"  • {i}" for i in obs.issues_detected)

    history_str = (
        "\n".join(history[-HISTORY_WINDOW:]) if history else "  (none yet)"
    )

    return textwrap.dedent(f"""
        TASK:   {obs.task_description}
        STEP:   {step} / {obs.max_steps}
        SCORE:  {obs.score:.4f}

        ISSUES DETECTED:
        {issues_str}

        COLUMN INFO:
        {col_info_str}

        DATASET SAMPLE (first 4 rows):
        {sample_str}

        PREVIOUS ACTIONS:
        {history_str}

        What is your next cleaning action?
    """).strip()


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(response_text: str) -> Dict[str, Any]:
    """Extract the first JSON object from the model response."""
    if not response_text:
        return FALLBACK_ACTION

    # Try direct parse first
    text = response_text.strip()
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    try:
        obj = json.loads(text)
        if "action_type" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Regex extraction fallback
    match = _JSON_OBJECT_RE.search(response_text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if "action_type" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    print(f"  [warn] Could not parse action from: {response_text!r}")
    return FALLBACK_ACTION


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> float:
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'='*60}")

    env = DataCleaningEnv()
    result = env.reset(task_id)
    obs = result.observation
    history: List[str] = []

    print(f"  Description: {obs.task_description}")
    print(f"  Issues at start ({len(obs.issues_detected)}):")
    for issue in obs.issues_detected:
        print(f"    • {issue}")
    print(f"  Initial score: {obs.score:.4f}")
    print()

    final_score = obs.score

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        if result.done:
            print(f"  Episode done at step {step - 1}.")
            break

        # Build messages
        user_prompt = build_user_prompt(step, obs, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        # LLM call
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] LLM call failed: {exc}. Using fallback.")
            response_text = json.dumps(FALLBACK_ACTION)

        action_dict = parse_action(response_text)
        action = Action(**action_dict)

        print(f"  Step {step:>2}: {action.action_type}({json.dumps(action.params)})")

        result = env.step(action)
        obs    = result.observation
        reward = result.reward

        final_score = obs.score
        history.append(
            f"Step {step}: {action.action_type}({json.dumps(action.params)}) "
            f"→ reward={reward.value:+.4f}, score={obs.score:.4f}"
        )

        print(f"           reward={reward.value:+.4f}  score={obs.score:.4f}  done={result.done}")
        if reward.message:
            print(f"           msg: {reward.message}")

        if result.done:
            break
    else:
        print(f"  Reached max inference steps ({MAX_STEPS_PER_TASK}).")

    # Print final breakdown
    _, breakdown = __import__("env.graders", fromlist=["grade_task"]).grade_task(
        task_id, env.dataset
    )
    print(f"\n  Final score: {final_score:.4f}")
    print("  Score breakdown:")
    for check, val in breakdown.items():
        print(f"    {check:<30} {val:.4f}")

    return final_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Validate config
    missing = []
    if not API_KEY:
        missing.append("HF_TOKEN (or API_KEY)")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if missing:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing)}")
        print("  Set them before running:")
        print("    export HF_TOKEN=hf_...")
        print("    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct")
        print("    export API_BASE_URL=https://router.huggingface.co/v1  # optional")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("\n" + "="*60)
    print("  Data Cleaning Environment — Baseline Inference")
    print("="*60)
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Base URL: {API_BASE_URL}")

    scores: Dict[str, float] = {}
    for task_id in TASKS:
        scores[task_id] = run_task(client, task_id)

    # Summary
    print("\n" + "="*60)
    print("  FINAL SCORES SUMMARY")
    print("="*60)
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:<14}  {score:.4f}  {bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average score: {avg:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
