"""FastAPI surface for the core pipeline. Synchronous for this slice — no
queue/worker yet (see PRD section 30 for the eventual async design). A real
deployment would move `run_pipeline` behind a background job as soon as
latency becomes a problem.
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
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


@app.post("/analysis")
async def create_analysis(
    resume_file: UploadFile = File(...),
    jd_text: str | None = Form(default=None),
    jd_file: UploadFile | None = File(default=None),
):
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
    try:
        if jd_file is not None:
            jd_ext = Path(jd_file.filename or "").suffix.lower()
            if jd_ext not in ALLOWED_JD_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported JD file type '{jd_ext}'. Allowed: {sorted(ALLOWED_JD_EXTENSIONS)}",
                )
            jd_path = _save_upload(jd_file, jd_ext)
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


# Mounted last so it never shadows the API routes above — StaticFiles at "/"
# matches any path Starlette hasn't already resolved, and route order is
# registration order.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
