"""Tests for HR/batch mode (app/hr_pipeline.py, app/hr_jobs.py). Every LLM
call and file read is mocked, so this runs offline without a GROQ_API_KEY."""

import app.hr_pipeline as hr_pipeline_module
from app.hr_jobs import create_batch_job, run_batch_job
from app.hr_pipeline import run_batch_pipeline
from app.schemas.analysis import (
    CandidateStatus,
    GapList,
    QualitativeMatch,
)
from app.schemas.document import DocumentClassification, ExtractedDocument, SourceFormat
from app.schemas.job import Importance, JobProfile, SkillRequirement
from app.schemas.resume import PersonalInfo, ResumeProfile

VALID_JD_TEXT = "Backend Engineer at Acme.\nRequired: Python. Responsibilities: build APIs."


def _resume_doc(text="Jane Doe\nSoftware Engineer\nSkills: Python, FastAPI") -> ExtractedDocument:
    return ExtractedDocument(source_format=SourceFormat.TEXT, raw_text=text)


def _resume_profile(skills=None) -> ResumeProfile:
    return ResumeProfile(
        personal_info=PersonalInfo(name="Jane Doe", email="jane@example.com", phone="555-1234"),
        summary="Backend engineer.",
        skills=skills or ["Python", "FastAPI"],
    )


def _job_profile() -> JobProfile:
    return JobProfile(
        role="Backend Engineer",
        company="Acme",
        required_skills=[
            SkillRequirement(skill="Python", importance=Importance.REQUIRED, evidence="Required: Python")
        ],
        responsibilities=["build APIs"],
        keywords=["python"],
    )


def _qualitative(score=80.0) -> QualitativeMatch:
    return QualitativeMatch(
        responsibility_match=score,
        responsibility_reasoning="reasoning",
        qualification_match=score,
        qualification_reasoning="reasoning",
    )


def _patch_common(monkeypatch, *, resume_docs=None, resume_profiles=None, qualitative_scores=None):
    """resume_docs / resume_profiles / qualitative_scores: dicts keyed by
    resume_path, or plain values reused for every candidate."""

    monkeypatch.setattr(
        hr_pipeline_module,
        "classify_job_description",
        lambda text: DocumentClassification(document_type="job_description", confidence=0.95, reason="looks like a JD"),
    )
    monkeypatch.setattr(hr_pipeline_module, "extract_job_profile", lambda text: _job_profile())

    def fake_extract_from_file(path):
        if resume_docs and path in resume_docs:
            return resume_docs[path]
        return _resume_doc()

    monkeypatch.setattr(hr_pipeline_module, "extract_from_file", fake_extract_from_file)

    def fake_classify_resume(text):
        return DocumentClassification(document_type="resume", confidence=0.95, reason="looks like a resume")

    monkeypatch.setattr(hr_pipeline_module, "classify_resume", fake_classify_resume)

    def fake_extract_resume_profile(text):
        return _resume_profile()

    monkeypatch.setattr(hr_pipeline_module, "extract_resume_profile", fake_extract_resume_profile)
    monkeypatch.setattr(hr_pipeline_module, "assess_qualitative_match", lambda resume, job: _qualitative())
    monkeypatch.setattr(
        hr_pipeline_module, "analyze_gaps", lambda resume, job, matches, qualitative: GapList(gaps=[])
    )


def test_batch_scores_multiple_candidates_and_ranks_by_score(monkeypatch):
    _patch_common(monkeypatch)

    # Give /path/b a stronger qualitative match so it should rank first.
    def fake_qualitative(resume, job):
        return _qualitative(score=95.0) if resume.personal_info.name == "Strong Candidate" else _qualitative(score=50.0)

    def fake_extract_resume_profile(text):
        if "Strong" in text:
            return ResumeProfile(personal_info=PersonalInfo(name="Strong Candidate"), skills=["Python"])
        return ResumeProfile(personal_info=PersonalInfo(name="Weak Candidate"), skills=["Python"])

    monkeypatch.setattr(hr_pipeline_module, "assess_qualitative_match", fake_qualitative)
    monkeypatch.setattr(hr_pipeline_module, "extract_resume_profile", fake_extract_resume_profile)
    monkeypatch.setattr(
        hr_pipeline_module,
        "extract_from_file",
        lambda path: _resume_doc("Strong candidate resume") if path == "/path/strong.pdf" else _resume_doc("Weak resume"),
    )

    outcome = run_batch_pipeline(
        VALID_JD_TEXT,
        resumes=[("weak.pdf", "/path/weak.pdf"), ("strong.pdf", "/path/strong.pdf")],
        jd_is_file=False,
    )

    assert outcome.status == "completed"
    names = [c.resume_profile.personal_info.name for c in outcome.result.candidates]
    assert names == ["Strong Candidate", "Weak Candidate"]
    assert all(c.status == CandidateStatus.SCORED for c in outcome.result.candidates)


def test_batch_rejects_invalid_jd_before_touching_resumes(monkeypatch):
    monkeypatch.setattr(
        hr_pipeline_module,
        "classify_job_description",
        lambda text: DocumentClassification(document_type="article", confidence=0.9, reason="not a JD"),
    )

    def fail(*args, **kwargs):
        raise AssertionError("should not process resumes when JD is rejected")

    monkeypatch.setattr(hr_pipeline_module, "extract_job_profile", fail)
    monkeypatch.setattr(hr_pipeline_module, "extract_from_file", fail)

    outcome = run_batch_pipeline(VALID_JD_TEXT, resumes=[("a.pdf", "/path/a.pdf")], jd_is_file=False)

    assert outcome.status == "rejected_jd"
    assert outcome.result is None


def test_batch_skips_rejected_resume_but_keeps_going(monkeypatch):
    _patch_common(monkeypatch)

    def fake_classify_resume(text):
        if "timetable" in text:
            return DocumentClassification(document_type="timetable", confidence=0.9, reason="looks like a schedule")
        return DocumentClassification(document_type="resume", confidence=0.95, reason="looks like a resume")

    monkeypatch.setattr(hr_pipeline_module, "classify_resume", fake_classify_resume)
    monkeypatch.setattr(
        hr_pipeline_module,
        "extract_from_file",
        lambda path: _resume_doc("timetable content") if path == "/path/bad.pdf" else _resume_doc(),
    )

    outcome = run_batch_pipeline(
        VALID_JD_TEXT,
        resumes=[("bad.pdf", "/path/bad.pdf"), ("good.pdf", "/path/good.pdf")],
        jd_is_file=False,
    )

    assert outcome.status == "completed"
    by_filename = {c.filename: c for c in outcome.result.candidates}
    assert by_filename["bad.pdf"].status == CandidateStatus.REJECTED_RESUME
    assert by_filename["good.pdf"].status == CandidateStatus.SCORED
    # Scored candidates rank ahead of unscored ones regardless of upload order.
    assert outcome.result.candidates[0].filename == "good.pdf"


def test_batch_captures_per_candidate_failure_without_sinking_batch(monkeypatch):
    _patch_common(monkeypatch)

    def flaky_extract(path):
        if path == "/path/broken.pdf":
            raise RuntimeError("corrupt file")
        return _resume_doc()

    monkeypatch.setattr(hr_pipeline_module, "extract_from_file", flaky_extract)

    outcome = run_batch_pipeline(
        VALID_JD_TEXT,
        resumes=[("broken.pdf", "/path/broken.pdf"), ("fine.pdf", "/path/fine.pdf")],
        jd_is_file=False,
    )

    assert outcome.status == "completed"
    by_filename = {c.filename: c for c in outcome.result.candidates}
    assert by_filename["broken.pdf"].status == CandidateStatus.FAILED
    assert "corrupt file" in by_filename["broken.pdf"].error
    assert by_filename["fine.pdf"].status == CandidateStatus.SCORED


def test_batch_job_tracks_steps_and_completes(monkeypatch):
    _patch_common(monkeypatch)
    job = create_batch_job(["a.pdf", "b.pdf"])

    run_batch_job(job, VALID_JD_TEXT, resumes=[("a.pdf", "/path/a.pdf"), ("b.pdf", "/path/b.pdf")], jd_is_file=False)

    view = job.to_dict()
    assert view["status"] == "completed"
    assert all(step["state"] == "done" for step in view["steps"])
    assert len(view["result"]["candidates"]) == 2


def test_batch_job_rejection_leaves_candidate_steps_pending(monkeypatch):
    monkeypatch.setattr(
        hr_pipeline_module,
        "classify_job_description",
        lambda text: DocumentClassification(document_type="article", confidence=0.9, reason="not a JD"),
    )
    job = create_batch_job(["a.pdf"])

    run_batch_job(job, VALID_JD_TEXT, resumes=[("a.pdf", "/path/a.pdf")], jd_is_file=False)

    view = job.to_dict()
    assert view["status"] == "rejected"
    steps_by_name = {s["name"]: s["state"] for s in view["steps"]}
    assert steps_by_name["validate_jd"] == "done"
    assert steps_by_name["candidate:a.pdf"] == "pending"
