"""
FastAPI server — exposes the Data Cleaning Environment via HTTP.

Endpoints
---------
POST /reset           body: {"task_id": "task_easy"|"task_medium"|"task_hard"}
POST /step            body: {"action_type": "...", "params": {...}}
GET  /state           returns lightweight episode snapshot
GET  /tasks           lists available tasks and their metadata
GET  /health          liveness probe
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import DataCleaningEnv
from env.models import Action, EnvState, StepResult
from env.tasks import TASK_DESCRIPTIONS, TASK_DIFFICULTIES, MAX_STEPS


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

env = DataCleaningEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm with default task so first reset is instant
    env.reset("task_easy")
    yield


app = FastAPI(
    title="Data Cleaning Environment",
    description=(
        "An OpenEnv-compliant reinforcement-learning environment "
        "where agents learn to clean and validate messy datasets."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_easy"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "environment": "data-cleaning-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id":     tid,
                "description": TASK_DESCRIPTIONS[tid],
                "difficulty":  TASK_DIFFICULTIES[tid],
                "max_steps":   MAX_STEPS[tid],
            }
            for tid in ["task_easy", "task_medium", "task_hard"]
        ]
    }


@app.post("/reset", response_model=StepResult)
def reset(body: Optional[ResetRequest] = None):
    task_id = body.task_id if body else "task_easy"
    try:
        result = env.reset(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/step", response_model=StepResult)
def step(action: Action):
    if env.task_id is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    result = env.step(action)
    return result


@app.get("/state", response_model=EnvState)
def state():
    if env.task_id is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.state()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
