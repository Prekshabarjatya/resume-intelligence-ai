"""In-memory job tracking for HR batch analysis (see app/hr_pipeline.py).

Same trade-offs and shape as app/jobs.py — one process, in-memory, real
progress derived from position in an ordered step list rather than a
separately-tracked "current step" field (see app/jobs.py's steps_view for
why that approach was buggy: writing "this just finished" and "this is
active" at the same instant meant "active" was never actually observable).
The step list here is built per-job since it depends on how many resumes
were uploaded and what they're named, unlike the fixed pipeline in
app/jobs.py.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from app.hr_pipeline import BatchOutcome, candidate_step_name, run_batch_pipeline

StepState = Literal["done", "active", "pending"]
JobStatus = Literal["running", "completed", "rejected", "failed"]


@dataclass
class BatchJobRecord:
    id: str
    step_order: list[str]
    step_labels: dict[str, str]
    status: JobStatus = "running"
    completed_steps: list[str] = field(default_factory=list)
    outcome: BatchOutcome | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    created_at: float = field(default_factory=time.monotonic)

    def steps_view(self) -> list[dict]:
        with self.lock:
            completed = set(self.completed_steps)
            still_running = self.status == "running"
        steps = []
        active_assigned = False
        for name in self.step_order:
            if name in completed:
                state: StepState = "done"
            elif still_running and not active_assigned:
                state = "active"
                active_assigned = True
            else:
                state = "pending"
            steps.append({"name": name, "label": self.step_labels[name], "state": state})
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


_BATCH_JOBS: dict[str, BatchJobRecord] = {}
_BATCH_JOBS_LOCK = threading.Lock()


def create_batch_job(resume_filenames: list[str]) -> BatchJobRecord:
    step_order = ["validate_jd", "extract_jd"] + [candidate_step_name(f) for f in resume_filenames]
    step_labels = {
        "validate_jd": "Validating job description",
        "extract_jd": "Extracting job requirements",
        **{candidate_step_name(f): f"Scoring {f}" for f in resume_filenames},
    }
    job = BatchJobRecord(id=str(uuid.uuid4()), step_order=step_order, step_labels=step_labels)
    with _BATCH_JOBS_LOCK:
        for stale in [j for j, rec in _BATCH_JOBS.items() if job.created_at - rec.created_at > 3600]:
            del _BATCH_JOBS[stale]
        _BATCH_JOBS[job.id] = job
    return job


def get_batch_job(job_id: str) -> BatchJobRecord | None:
    with _BATCH_JOBS_LOCK:
        job = _BATCH_JOBS.get(job_id)
        if job is not None and time.monotonic() - job.created_at > 3600:
            del _BATCH_JOBS[job_id]
            return None
        return job


def run_batch_job(job: BatchJobRecord, jd: str, resumes: list[tuple[str, str]], jd_is_file: bool) -> None:
    """resumes: list of (display_filename, temp_file_path). Runs
    synchronously — call from a worker thread, same as run_job in
    app/jobs.py."""

    def on_step(step_name: str) -> None:
        with job.lock:
            if step_name not in job.completed_steps:
                job.completed_steps.append(step_name)

    try:
        outcome = run_batch_pipeline(jd, resumes, jd_is_file=jd_is_file, on_step=on_step)
        with job.lock:
            job.outcome = outcome
            job.status = "rejected" if outcome.status == "rejected_jd" else "completed"
    except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed silently
        with job.lock:
            job.status = "failed"
            job.error = str(exc)
