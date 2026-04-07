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
"""

import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List

# Ensure repo root is on the path so `env` package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from env import Action, DataCleaningEnv
from env.graders import grade_task

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME: str   = os.getenv("MODEL_NAME", "")

MAX_STEPS_PER_TASK = 12
TEMPERATURE        = 0.1
MAX_TOKENS         = 256
HISTORY_WINDOW     = 5

FALLBACK_ACTION = {"action_type": "done", "params": {}}
TASKS = ["task_easy", "task_medium", "task_hard"]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data cleaning agent. You receive information about a messy dataset
and must apply cleaning operations one at a time to fix all detected issues.

Available actions (respond with ONLY a valid JSON object - no markdown, no prose):

{"action_type": "remove_duplicates",  "params": {}}
{"action_type": "drop_nulls",         "params": {"column": "<col>"}}
{"action_type": "fill_nulls",         "params": {"column": "<col>", "value": 0}}
{"action_type": "fill_nulls",         "params": {"column": "<col>", "strategy": "zero"}}
{"action_type": "fix_type",           "params": {"column": "<col>", "dtype": "int"}}
{"action_type": "fix_type",           "params": {"column": "<col>", "dtype": "float"}}
{"action_type": "normalize_column",   "params": {"column": "<col>"}}
{"action_type": "clip_outliers",      "params": {"column": "<col>", "lower": 0, "upper": 9999}}
{"action_type": "strip_whitespace",   "params": {"column": "<col>"}}
{"action_type": "standardize_dates",  "params": {"column": "<col>"}}
{"action_type": "strip_currency",     "params": {"column": "<col>"}}
{"action_type": "drop_invalid",       "params": {"column": "<col>", "rule": "email"}}
{"action_type": "filter_rows",        "params": {"column": "<col>", "operator": ">=", "value": 0}}
{"action_type": "done",               "params": {}}

Rules:
- Output ONLY the JSON object, nothing else.
- Do not include markdown fences or extra text.
- Call "done" when all issues are resolved or you cannot improve further.
""").strip()


def build_user_prompt(step: int, obs: Any, history: List[str]) -> str:
    sample = obs.dataset[:4]
    sample_str = json.dumps(sample, indent=2)
    col_info_str = json.dumps(obs.column_info, indent=2)
    issues_str = "\n".join(f"  - {i}" for i in obs.issues_detected)
    history_str = "\n".join(history[-HISTORY_WINDOW:]) if history else "  (none yet)"

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
    if not response_text:
        return FALLBACK_ACTION
    text = response_text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        obj = json.loads(text)
        if "action_type" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    match = _JSON_OBJECT_RE.search(response_text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if "action_type" in obj:
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    print(f"  [warn] Could not parse action from: {response_text!r}")
    return FALLBACK_ACTION


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> float:
    print(f"START {task_id}")

    try:
        env = DataCleaningEnv()
        result = env.reset(task_id)
        obs = result.observation
        history: List[str] = []

        final_score = obs.score

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if result.done:
                break

            try:
                user_prompt = build_user_prompt(step, obs, history)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ]
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"  [warn] LLM call failed: {exc}. Using fallback.")
                response_text = json.dumps(FALLBACK_ACTION)

            try:
                action_dict = parse_action(response_text)
                action = Action(**action_dict)
            except Exception as exc:
                print(f"  [warn] Action parse failed: {exc}. Using fallback.")
                action = Action(**FALLBACK_ACTION)

            print(f"STEP {step} action={action.action_type} params={json.dumps(action.params)}")

            try:
                result = env.step(action)
                obs    = result.observation
                reward = result.reward
                final_score = obs.score
            except Exception as exc:
                print(f"  [warn] env.step failed: {exc}")
                break

            history.append(
                f"Step {step}: {action.action_type}({json.dumps(action.params)}) "
                f"-> reward={reward.value:+.4f}, score={obs.score:.4f}"
            )

            if result.done:
                break
        else:
            print(f"  [warn] Reached max steps ({MAX_STEPS_PER_TASK}).")

        try:
            _, breakdown = grade_task(task_id, env.dataset)
            for check, val in breakdown.items():
                print(f"  score {check}={val:.4f}")
        except Exception as exc:
            print(f"  [warn] Could not compute breakdown: {exc}")

        print(f"END {task_id} score={final_score:.4f}")
        return final_score

    except Exception as exc:
        print(f"  [error] Task {task_id} failed: {exc}")
        print(f"END {task_id} score=0.0000")
        return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    missing = []
    if not API_KEY:
        missing.append("HF_TOKEN (or API_KEY)")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if missing:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[ERROR] Failed to create OpenAI client: {exc}")
        sys.exit(1)

    print("=" * 60)
    print("Data Cleaning Environment - Baseline Inference")
    print("=" * 60)
    print(f"Model:    {MODEL_NAME}")
    print(f"Base URL: {API_BASE_URL}")
    print()

    scores: Dict[str, float] = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(client, task_id)
        except Exception as exc:
            print(f"[error] Unhandled exception in {task_id}: {exc}")
            print(f"END {task_id} score=0.0000")
            scores[task_id] = 0.0

    print()
    print("=" * 60)
    print("FINAL SCORES SUMMARY")
    print("=" * 60)
    for task_id, score in scores.items():
        bar = "X" * int(score * 20)
        print(f"{task_id:<14}  {score:.4f}  {bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"Average score: {avg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
