"""
Inference Script - Data Cleaning Environment

Mandatory environment variables:
    API_BASE_URL  : LLM API base URL  (default: https://router.huggingface.co/v1)
    MODEL_NAME    : Model identifier
    HF_TOKEN      : Hugging Face / API key
"""

import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "")

MAX_STEPS_PER_TASK = 12
TEMPERATURE        = 0.1
MAX_TOKENS         = 256
HISTORY_WINDOW     = 5
FALLBACK_ACTION    = {"action_type": "done", "params": {}}
TASKS              = ["task_easy", "task_medium", "task_hard"]

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
- Call done when all issues are resolved or you cannot improve further.
""").strip()


def build_user_prompt(step, obs, history):
    try:
        sample_str   = json.dumps(obs.dataset[:4], indent=2)
        col_info_str = json.dumps(obs.column_info, indent=2)
        issues_str   = "\n".join(f"  - {i}" for i in obs.issues_detected)
        history_str  = "\n".join(history[-HISTORY_WINDOW:]) if history else "  (none yet)"
        return (
            f"TASK: {obs.task_description}\n"
            f"STEP: {step} / {obs.max_steps}\n"
            f"SCORE: {obs.score:.4f}\n\n"
            f"ISSUES:\n{issues_str}\n\n"
            f"COLUMN INFO:\n{col_info_str}\n\n"
            f"DATASET SAMPLE:\n{sample_str}\n\n"
            f"PREVIOUS ACTIONS:\n{history_str}\n\n"
            f"What is your next cleaning action?"
        )
    except Exception:
        return "What is your next cleaning action?"


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(text):
    if not text:
        return FALLBACK_ACTION
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        obj = json.loads(text)
        if "action_type" in obj:
            return obj
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "action_type" in obj:
                return obj
        except Exception:
            pass
    return FALLBACK_ACTION


def run_task(client, task_id):
    print(f"START {task_id}")
    final_score = 0.0
    try:
        from env import Action, DataCleaningEnv
        from env.graders import grade_task

        env    = DataCleaningEnv()
        result = env.reset(task_id)
        obs    = result.observation
        history = []
        final_score = obs.score

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if result.done:
                break

            response_text = json.dumps(FALLBACK_ACTION)
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_prompt(step, obs, history)},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as e:
                print(f"  [warn] LLM error: {e}")

            try:
                action = Action(**parse_action(response_text))
            except Exception:
                action = Action(**FALLBACK_ACTION)

            print(f"STEP {step} action={action.action_type} params={json.dumps(action.params)}")

            try:
                result      = env.step(action)
                obs         = result.observation
                reward      = result.reward
                final_score = obs.score
                history.append(
                    f"Step {step}: {action.action_type} "
                    f"reward={reward.value:+.4f} score={obs.score:.4f}"
                )
            except Exception as e:
                print(f"  [warn] step error: {e}")
                break

            if result.done:
                break

        try:
            _, breakdown = grade_task(task_id, env.dataset)
            for check, val in breakdown.items():
                print(f"  score {check}={val:.4f}")
        except Exception:
            pass

    except Exception as e:
        print(f"  [error] task failed: {e}")

    print(f"END {task_id} score={final_score:.4f}")
    return final_score


def main():
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[ERROR] Could not create client: {e}")
        for task_id in TASKS:
            print(f"START {task_id}")
            print(f"END {task_id} score=0.0000")
        return

    print("=" * 60)
    print("Data Cleaning Environment - Baseline Inference")
    print("=" * 60)
    print(f"Model:    {MODEL_NAME}")
    print(f"Base URL: {API_BASE_URL}")
    print()

    scores = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(client, task_id)
        except Exception as e:
            print(f"[error] {task_id}: {e}")
            print(f"END {task_id} score=0.0000")
            scores[task_id] = 0.0

    print()
    print("=" * 60)
    print("FINAL SCORES")
    print("=" * 60)
    for task_id, score in scores.items():
        print(f"{task_id}: {score:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"Average: {avg:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
    sys.exit(0)
