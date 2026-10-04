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

import base64
import secrets
import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.hr_jobs import create_batch_job, get_batch_job, run_batch_job
from app.jobs import create_job, get_job, run_job
from app.agents.resume_writer import generate_resume, score_resume
from app.pipeline import run_pipeline
from app.schemas.analysis import PipelineStatus
from app.schemas.resume import ResumeProfile
from app.security import SECURITY_HEADERS, RateLimiter, client_key

app = FastAPI(title="Resume Intelligence Platform", version="0.1.0")

# Per client IP, per minute. Anything that calls the LLM is tight; the ATS
# re-score is deterministic and cheap, so it can be called on every edit.
_analysis_limiter = RateLimiter(limit=6, window_seconds=60)
_generate_limiter = RateLimiter(limit=4, window_seconds=60)
_ats_limiter = RateLimiter(limit=60, window_seconds=60)
# Backstop for every LLM-calling route, shared by all clients, so the Groq
# quota is protected even if per-IP keys can be dodged.
_llm_global_limiter = RateLimiter(limit=40, window_seconds=60)
MAX_RESUME_JSON_BYTES = 100_000


def _limit_llm(request: Request, per_client: RateLimiter) -> None:
    per_client.check(client_key(request))
    _llm_global_limiter.check("all")

STATIC_DIR = Path(__file__).parent / "static"

# Optional shared-password gate (HTTP Basic) for public deployments. Off by
# default for local dev (settings.app_password == ""). /health is always
# left open so a hosting platform's health check doesn't need credentials.
# Username is ignored on purpose — this is a single shared password, not a
# per-user account system.
_UNPROTECTED_PATHS = {"/health"}


@app.middleware("http")
async def require_shared_password(request: Request, call_next):
    if not settings.app_password or request.url.path in _UNPROTECTED_PATHS:
        return await call_next(request)

    header = request.headers.get("authorization", "")
    supplied = ""
    if header.startswith("Basic "):
        try:
            decoded = base64.b64decode(header[len("Basic ") :]).decode("utf-8")
            _, _, supplied = decoded.partition(":")
        except (ValueError, UnicodeDecodeError):
            supplied = ""

    if secrets.compare_digest(supplied, settings.app_password):
        return await call_next(request)

    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Resume Intelligence Platform"'},
    )

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    # Registered after the password gate so it is the outermost layer and
    # also covers that gate's 401 responses.
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


ALLOWED_RESUME_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_JD_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_BATCH_RESUMES = 20


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


def _validate_and_save_jd(jd_text: str | None, jd_file: UploadFile | None) -> tuple[str | None, str | None]:
    """Returns (jd_path_or_none, jd_text_or_none). Raises HTTPException on
    invalid input. Caller owns cleanup of any saved path."""
    if not settings.groq_api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured on the server.")

    if not jd_text and not jd_file:
        raise HTTPException(status_code=400, detail="Provide either jd_text or jd_file.")

    if jd_file is None:
        return None, jd_text

    jd_ext = Path(jd_file.filename or "").suffix.lower()
    if jd_ext not in ALLOWED_JD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported JD file type '{jd_ext}'. Allowed: {sorted(ALLOWED_JD_EXTENSIONS)}",
        )
    return _save_upload(jd_file, jd_ext), jd_text


def _validate_and_save_resumes(resume_files: list[UploadFile]) -> list[tuple[str, str]]:
    """Returns [(display_filename, temp_path), ...]. Raises HTTPException on
    invalid input. Caller owns cleanup of any saved paths."""
    if not resume_files:
        raise HTTPException(status_code=400, detail="Provide at least one resume file.")
    if len(resume_files) > MAX_BATCH_RESUMES:
        raise HTTPException(
            status_code=400, detail=f"At most {MAX_BATCH_RESUMES} resumes per batch (got {len(resume_files)})."
        )

    saved: list[tuple[str, str]] = []
    for upload in resume_files:
        filename = upload.filename or "resume"
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_RESUME_EXTENSIONS:
            for _, path in saved:
                Path(path).unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported resume file type '{ext}' in '{filename}'. Allowed: {sorted(ALLOWED_RESUME_EXTENSIONS)}",
            )
        saved.append((filename, _save_upload(upload, ext)))
    return saved


@app.post("/analysis")
async def create_analysis(
    request: Request,
    resume_file: UploadFile = File(...),
    jd_text: str | None = Form(default=None),
    jd_file: UploadFile | None = File(default=None),
):
    _limit_llm(request, _analysis_limiter)
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
            content={
                "status": outcome.status.value,
                "reason": outcome.rejection_reason,
                "metrics": outcome.metrics.to_dict() if outcome.metrics else None,
            },
        )

    return {
        "status": outcome.status.value,
        "result": outcome.result.model_dump(),
        "metrics": outcome.metrics.to_dict() if outcome.metrics else None,
    }


@app.post("/analysis/start")
async def start_analysis(
    request: Request,
    resume_file: UploadFile = File(...),
    jd_text: str | None = Form(default=None),
    jd_file: UploadFile | None = File(default=None),
):
    _limit_llm(request, _analysis_limiter)
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


def _completed_result(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No job with that id (it may have expired).")
    with job.lock:
        outcome = job.outcome if job.status == "completed" else None
    if outcome is None or outcome.result is None:
        raise HTTPException(status_code=409, detail="That analysis has not completed.")
    return outcome.result


@app.post("/analysis/{job_id}/resume")
def generate_improved_resume(job_id: str, request: Request):
    """Rewrites the analyzed resume for the analyzed job. The server uses its
    own stored analysis, never client-supplied resume text, so this can't be
    used to push arbitrary content at the LLM."""
    _limit_llm(request, _generate_limiter)
    result = _completed_result(job_id)
    resume = generate_resume(result.resume_profile, result.job_profile, result.gaps)
    return {
        "resume": resume.model_dump(),
        "ats": score_resume(resume, result.job_profile).model_dump(),
        # Scored the same way as "ats", so the before/after comparison is fair.
        "baseline_ats": score_resume(result.resume_profile, result.job_profile).model_dump(),
    }


@app.post("/analysis/{job_id}/ats")
async def score_edited_resume(job_id: str, request: Request):
    """Deterministic ATS score for an edited resume. No LLM call."""
    _ats_limiter.check(client_key(request))
    result = _completed_result(job_id)
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_RESUME_JSON_BYTES:
        raise HTTPException(status_code=413, detail="Resume data is too large.")
    body = await request.body()
    if len(body) > MAX_RESUME_JSON_BYTES:
        raise HTTPException(status_code=413, detail="Resume data is too large.")
    try:
        resume = ResumeProfile.model_validate_json(body)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid resume data.")
    return {"ats": score_resume(resume, result.job_profile).model_dump()}


@app.post("/hr/batch-analysis/start")
async def start_batch_analysis(
    request: Request,
    resume_files: list[UploadFile] = File(...),
    jd_text: str | None = Form(default=None),
    jd_file: UploadFile | None = File(default=None),
):
    """HR mode: one JD scored against many resumes, ranked for screening.
    Distinct from /analysis/start (candidate mode) — see app/hr_pipeline.py
    for why this isn't just a loop over the candidate pipeline."""
    _limit_llm(request, _analysis_limiter)
    jd_path, jd_text = _validate_and_save_jd(jd_text, jd_file)
    try:
        resumes = _validate_and_save_resumes(resume_files)
    except HTTPException:
        if jd_path:
            Path(jd_path).unlink(missing_ok=True)
        raise

    job = create_batch_job([filename for filename, _ in resumes])

    def _run_and_cleanup() -> None:
        try:
            run_batch_job(job, jd_path if jd_path is not None else jd_text, resumes, jd_is_file=jd_path is not None)
        finally:
            if jd_path:
                Path(jd_path).unlink(missing_ok=True)
            for _, path in resumes:
                Path(path).unlink(missing_ok=True)

    threading.Thread(target=_run_and_cleanup, daemon=True).start()
    return {"job_id": job.id}


@app.get("/hr/batch-analysis/{job_id}")
def get_batch_analysis_status(job_id: str):
    job = get_batch_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No job with that id (server may have restarted).")
    return job.to_dict()


# Mounted last so it never shadows the API routes above — StaticFiles at "/"
# matches any path Starlette hasn't already resolved, and route order is
# registration order.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
