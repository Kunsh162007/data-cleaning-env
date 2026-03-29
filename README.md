---
title: Data Cleaning Environment
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Data Cleaning Environment

An **OpenEnv-compliant** reinforcement-learning environment where an AI agent
learns to clean and validate messy, real-world tabular datasets.

The agent observes a dirty dataset, identifies data quality issues, and applies
targeted cleaning operations one step at a time. A programmatic grader scores
each dimension of data quality, giving the agent a continuous reward signal.

---

## Why This Domain?

Data quality is a chronic problem in every data-driven organisation. Studies
consistently show that data scientists spend 60–80 % of their time on cleaning
and preparation. An agent that can autonomously triage and repair messy data
would have immediate, practical value — yet no mature RL benchmark exists for
this task. This environment fills that gap.

---

## Environment Overview

| Property | Value |
|---|---|
| Action space | Discrete-parametric (named cleaning ops + JSON params) |
| Observation space | Structured JSON (dataset rows, column stats, issue list, score) |
| Reward | Delta-reward + step penalty (partial progress signal throughout) |
| Episode end | Agent calls `done`, or max-steps reached |
| Tasks | 3 (easy → medium → hard) |

---

## Action Space

Each action is a JSON object with two fields:

```json
{"action_type": "<operation>", "params": {}}
```

| `action_type` | Required params | Description |
|---|---|---|
| `remove_duplicates` | `subset` (optional list of cols) | Remove duplicate rows |
| `drop_nulls` | `column` | Drop rows where column is null/NA |
| `fill_nulls` | `column`, `value` **or** `strategy` (`zero`/`mean`/`unknown`) | Fill nulls |
| `fix_type` | `column`, `dtype` (`int`/`float`/`str`) | Cast column to target type |
| `normalize_column` | `column`, `mapping` (optional) | Title-case or apply value mapping |
| `clip_outliers` | `column`, `lower`, `upper` | Clamp numeric values to range |
| `strip_whitespace` | `column` (optional, default all) | Strip leading/trailing spaces |
| `filter_rows` | `column`, `operator` (`>=`/`<=`/`>`/`<`/`==`/`!=`), `value` | Keep matching rows |
| `drop_invalid` | `column`, `rule` (`email`/`positive`/`iso_date`) | Drop rows failing a validation rule |
| `standardize_dates` | `column` | Normalise date strings → `YYYY-MM-DD` |
| `strip_currency` | `column` | Remove `$€£¥`, convert to float |
| `done` | — | Signal task completion |

---

## Observation Space

```json
{
  "task_id":          "task_easy",
  "task_description": "Clean a small customer records dataset ...",
  "dataset":          [ { "name": "Alice ", "age": "28", ... }, ... ],
  "column_info": {
    "age": { "dtype": "str", "null_count": 3, "unique_count": 7, "sample_values": [...] }
  },
  "issues_detected":  ["Column 'age': 3 null/missing values", ...],
  "step":             0,
  "max_steps":        10,
  "score":            0.2500
}
```

---

## Reward Function

```
step_reward  = new_score − old_score − 0.01    # delta reward − step penalty
invalid_action → −0.05
done (score ≥ 0.80) → final_score + 0.10 bonus
done (score < 0.80) → final_score − 0.05 penalty
```

The **delta reward** gives a useful gradient signal throughout the episode.
The **step penalty** encourages efficient cleaning (fewer actions = better).
The **completion bonus** incentivises the agent to stop confidently rather than
thrash near the optimum.

---

## Tasks

### Task 1 — Customer Records Cleanup *(easy, 10 steps)*

A 10-row customer dataset with:
- 3 ages stored as the string `"N/A"`
- Ages stored as strings instead of integers
- 1 exact duplicate row
- 2 names with trailing whitespace

**Grader checks (sum → 1.0):**

| Check | Weight |
|---|---|
| No null/NA ages | 0.30 |
| No duplicate rows | 0.25 |
| Age column is numeric (int/float) | 0.25 |
| No whitespace in name fields | 0.20 |

---

### Task 2 — Product Inventory Normalisation *(medium, 15 steps)*

A 15-row product inventory with:
- Mixed-case category names (`Electronics` / `electronics` / `ELECTRONICS`)
- 3 negative prices
- 5 null stock values
- 3 null product codes

**Grader checks (sum → 1.0):**

| Check | Weight |
|---|---|
| All categories in title-case | 0.25 |
| No negative prices | 0.25 |
| No null stock values | 0.25 |
| No null product codes | 0.25 |

---

### Task 3 — Orders Validation *(hard, 20 steps)*

A 20-row orders dataset with **five simultaneous issue types**:
- Mixed date formats (`2024-01-15` / `01/15/2024` / `Jan 15 2024`)
- 4 negative quantities
- 4 invalid email addresses (missing `@`, extra spaces)
- All 20 prices contain currency symbols (`$`, `€`, `£`)
- 5 discount percentages > 100

**Grader checks (sum → 1.0):**

| Check | Weight |
|---|---|
| All dates in `YYYY-MM-DD` | 0.20 |
| No negative quantities | 0.20 |
| All emails valid (`x@y.z`) | 0.20 |
| All prices are floats (no `$€£¥`) | 0.20 |
| All discounts in `[0, 100]` | 0.20 |

---

## API Reference

| Method | Path | Body / Params | Description |
|---|---|---|---|
| `GET` | `/health` | — | Liveness probe |
| `GET` | `/tasks` | — | List tasks + metadata |
| `POST` | `/reset` | `{"task_id": "task_easy"}` | Start new episode |
| `POST` | `/step` | `{"action_type": "...", "params": {...}}` | Take one step |
| `GET` | `/state` | — | Lightweight episode snapshot |

---

## Setup & Usage

### Local (Python)

```bash
# 1. Clone
git clone https://huggingface.co/spaces/<your-hf-username>/data-cleaning-env
cd data-cleaning-env

# 2. Install
pip install -r requirements.txt

# 3. Start server
uvicorn app:app --host 0.0.0.0 --port 7860

# 4. Quick smoke-test
curl -s http://localhost:7860/health
curl -s -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "task_easy"}' | python -m json.tool | head -30
```

### Docker

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run Baseline Inference

```bash
export HF_TOKEN=hf_...
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1   # optional

python inference.py
```

---

## Baseline Scores

Scores from `Qwen/Qwen2.5-72B-Instruct` via HF Inference Router:

| Task | Score |
|---|---|
| task_easy | ~0.95 |
| task_medium | ~0.82 |
| task_hard | ~0.65 |

*(Frontier models typically score ≥ 0.90 on easy, 0.70–0.85 on medium, 0.50–0.70 on hard.)*

---

## Project Structure

```
data-cleaning-env/
├── app.py              # FastAPI server (OpenEnv HTTP endpoints)
├── inference.py        # Baseline LLM agent script
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile          # Container definition
├── requirements.txt
├── README.md
└── env/
    ├── __init__.py
    ├── models.py        # Pydantic typed models
    ├── datasets.py      # Hardcoded messy datasets
    ├── graders.py       # Deterministic grader functions
    ├── tasks.py         # Task metadata + issue detector
    └── environment.py   # Core DataCleaningEnv class
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes (inference) | — | Hugging Face API key |
| `MODEL_NAME` | Yes (inference) | — | Model identifier |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API base URL |

---

## License

Apache 2.0
