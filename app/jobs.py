"""In-memory job tracking for the async analysis endpoint.

This is intentionally not the PRD's eventual Redis-backed queue (see PRD
section 30) — it's the smallest thing that lets the UI show real per-step
progress instead of a black-box spinner. Progress comes directly from
LangGraph's own execution (graph.stream(..., stream_mode="updates") yields
one event per node as it actually finishes), not from a hardcoded timer, so
the checklist reflects what the pipeline is truly doing. A single process's
worth of jobs living in a dict is a fine trade-off for a local dev tool;
it does not survive a server restart and does not scale past one process.
"""

import threading
import uuid
from dataclasses import dataclass, field
from typing import Literal

from app.pipeline import PipelineOutcome, run_pipeline_with_progress

# Node name -> human-readable label, in pipeline order. Used both to render
# steps that haven't run yet ("pending") and to label ones that have.
STEP_LABELS: dict[str, str] = {
    "validate_resume": "Validating resume",
    "validate_jd": "Validating job description",
    "extract_resume": "Extracting resume details",
    "extract_jd": "Extracting job requirements",
    "match_skills": "Matching skills",
    "assess_qualitative": "Assessing responsibility & qualification fit",
    "analyze_ats_and_score": "Running ATS analysis & scoring",
    "analyze_gaps": "Analyzing gaps",
    "recommend": "Generating recommendations",
}
STEP_ORDER = list(STEP_LABELS.keys())

StepState = Literal["done", "active", "pending"]
JobStatus = Literal["running", "completed", "rejected", "failed"]


@dataclass
class JobRecord:
    id: str
    status: JobStatus = "running"
    completed_steps: list[str] = field(default_factory=list)
    outcome: PipelineOutcome | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def steps_view(self) -> list[dict]:
        """"Active" is derived from position, not tracked separately: the
        first not-yet-completed step in order is the one presumably running
        right now. Tracking a separate active_step field written at the same
        time a node is marked complete meant the two updates always landed
        together, so the completed check (checked first) always won and the
        active state was never actually observable."""
        with self.lock:
            completed = set(self.completed_steps)
            still_running = self.status == "running"
        steps = []
        active_assigned = False
        for name in STEP_ORDER:
            if name in completed:
                state: StepState = "done"
            elif still_running and not active_assigned:
                state = "active"
                active_assigned = True
            else:
                state = "pending"
            steps.append({"name": name, "label": STEP_LABELS[name], "state": state})
        return steps

    def to_dict(self) -> dict:
        with self.lock:
            status = self.status
            error = self.error
            outcome = self.outcome
        payload = {"job_id": self.id, "status": status, "steps": self.steps_view()}
        if status == "failed":
            payload["error"] = error
        elif status == "rejected" and outcome is not None:
            payload["reason"] = outcome.rejection_reason
        elif status == "completed" and outcome is not None:
            payload["result"] = outcome.result.model_dump()
        return payload


_JOBS: dict[str, JobRecord] = {}
_JOBS_LOCK = threading.Lock()


def create_job() -> JobRecord:
    job = JobRecord(id=str(uuid.uuid4()))
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    return job


def get_job(job_id: str) -> JobRecord | None:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def run_job(job: JobRecord, resume_path: str, jd: str, jd_is_file: bool) -> None:
    """Runs the pipeline synchronously (meant to be called from a worker
    thread), updating `job` after every node so a concurrent poller sees
    live progress. Intentionally swallows exceptions into job.error rather
    than raising — this runs off the request/response cycle."""

    def on_step(node_name: str, _state) -> None:
        with job.lock:
            if node_name not in job.completed_steps:
                job.completed_steps.append(node_name)

    try:
        outcome = run_pipeline_with_progress(resume_path, jd, jd_is_file=jd_is_file, on_step=on_step)
        with job.lock:
            job.outcome = outcome
            if outcome.status.value.startswith("rejected"):
                job.status = "rejected"
            else:
                job.status = "completed"
    except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed silently
        with job.lock:
            job.status = "failed"
            job.error = str(exc)
