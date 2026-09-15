"""FastAPI surface for the core pipeline.

Two ways to run an analysis:
- POST /analysis: synchronous, returns the full result in one response.
  Simple for scripts/tests, but the caller is blocked for the whole pipeline
  (multiple LLM calls — can take under a minute).
- POST /analysis/start + GET /analysis/{job_id}: the pipeline runs in a
  background thread while the caller polls for progress. This is what the
  UI uses to show real per-step status instead of a spinner. Both paths run
  the exact same graph (app/pipeline.py) — this is not a second pipeline,
  just a second way to observe the same one running.

Still no persistent queue/worker (see PRD section 30) — jobs live in an
in-memory dict (app/jobs.py) that doesn't survive a server restart. Good
enough for one local dev process; a real deployment would swap this for
Redis + a proper worker without touching the graph itself.
"""

import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.jobs import create_job, get_job, run_job
from app.pipeline import run_pipeline
from app.schemas.analysis import PipelineStatus

app = FastAPI(title="Resume Intelligence Platform", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"

ALLOWED_RESUME_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_JD_EXTENSIONS = {".pdf", ".docx", ".txt"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "groq_configured": bool(settings.groq_api_key)}


def _save_upload(upload: UploadFile, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        return tmp.name


def _validate_and_save_inputs(
    resume_file: UploadFile, jd_text: str | None, jd_file: UploadFile | None
) -> tuple[str, str | None, str | None]:
    """Returns (resume_path, jd_path_or_none, jd_text_or_none). Raises
    HTTPException on invalid input. Caller owns cleanup of any saved paths."""
    if not settings.groq_api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured on the server.")

    resume_ext = Path(resume_file.filename or "").suffix.lower()
    if resume_ext not in ALLOWED_RESUME_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported resume file type '{resume_ext}'. Allowed: {sorted(ALLOWED_RESUME_EXTENSIONS)}",
        )

    if not jd_text and not jd_file:
        raise HTTPException(status_code=400, detail="Provide either jd_text or jd_file.")

    resume_path = _save_upload(resume_file, resume_ext)
    jd_path = None
    if jd_file is not None:
        jd_ext = Path(jd_file.filename or "").suffix.lower()
        if jd_ext not in ALLOWED_JD_EXTENSIONS:
            Path(resume_path).unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported JD file type '{jd_ext}'. Allowed: {sorted(ALLOWED_JD_EXTENSIONS)}",
            )
        jd_path = _save_upload(jd_file, jd_ext)

    return resume_path, jd_path, jd_text


@app.post("/analysis")
async def create_analysis(
    resume_file: UploadFile = File(...),
    jd_text: str | None = Form(default=None),
    jd_file: UploadFile | None = File(default=None),
):
    resume_path, jd_path, jd_text = _validate_and_save_inputs(resume_file, jd_text, jd_file)
    try:
        if jd_path is not None:
            outcome = run_pipeline(resume_path, jd_path, jd_is_file=True)
        else:
            outcome = run_pipeline(resume_path, jd_text, jd_is_file=False)
    finally:
        Path(resume_path).unlink(missing_ok=True)
        if jd_path:
            Path(jd_path).unlink(missing_ok=True)

    if outcome.status in (PipelineStatus.REJECTED_RESUME, PipelineStatus.REJECTED_JD):
        return JSONResponse(
            status_code=422,
            content={"status": outcome.status.value, "reason": outcome.rejection_reason},
        )

    return {"status": outcome.status.value, "result": outcome.result.model_dump()}


@app.post("/analysis/start")
async def start_analysis(
    resume_file: UploadFile = File(...),
    jd_text: str | None = Form(default=None),
    jd_file: UploadFile | None = File(default=None),
):
    resume_path, jd_path, jd_text = _validate_and_save_inputs(resume_file, jd_text, jd_file)
    job = create_job()

    def _run_and_cleanup() -> None:
        try:
            if jd_path is not None:
                run_job(job, resume_path, jd_path, jd_is_file=True)
            else:
                run_job(job, resume_path, jd_text, jd_is_file=False)
        finally:
            Path(resume_path).unlink(missing_ok=True)
            if jd_path:
                Path(jd_path).unlink(missing_ok=True)

    threading.Thread(target=_run_and_cleanup, daemon=True).start()
    return {"job_id": job.id}


@app.get("/analysis/{job_id}")
def get_analysis_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No job with that id (server may have restarted).")
    return job.to_dict()


# Mounted last so it never shadows the API routes above — StaticFiles at "/"
# matches any path Starlette hasn't already resolved, and route order is
# registration order.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
