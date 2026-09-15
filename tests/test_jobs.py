"""Tests for the async job runner (app/jobs.py) and the streaming pipeline
entry point (run_pipeline_with_progress). Every LLM call and the resume file
read are mocked, so this runs offline without a GROQ_API_KEY."""

import app.graph as graph_module
import app.pipeline as pipeline_module
from app.jobs import STEP_ORDER, create_job, run_job
from app.pipeline import run_pipeline_with_progress
from app.schemas.analysis import (
    GapList,
    PipelineStatus,
    QualitativeMatch,
    RecommendationList,
)
from app.schemas.document import DocumentClassification, ExtractedDocument, SourceFormat
from app.schemas.job import Importance, JobProfile, SkillRequirement
from app.schemas.resume import PersonalInfo, ResumeProfile

VALID_RESUME_TEXT = "Jane Doe\nSoftware Engineer\nExperience: Acme Corp\nSkills: Python, FastAPI"
VALID_JD_TEXT = "Backend Engineer at Acme.\nRequired: Python. Responsibilities: build APIs."


def _resume_doc() -> ExtractedDocument:
    return ExtractedDocument(source_format=SourceFormat.TEXT, raw_text=VALID_RESUME_TEXT)


def _resume_profile() -> ResumeProfile:
    return ResumeProfile(
        personal_info=PersonalInfo(name="Jane Doe", email="jane@example.com", phone="555-1234"),
        summary="Backend engineer.",
        skills=["Python", "FastAPI"],
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


def _patch_happy_path(monkeypatch):
    monkeypatch.setattr(pipeline_module, "extract_from_file", lambda path: _resume_doc())
    monkeypatch.setattr(
        graph_module,
        "classify_resume",
        lambda text: DocumentClassification(document_type="resume", confidence=0.95, reason="looks like a resume"),
    )
    monkeypatch.setattr(
        graph_module,
        "classify_job_description",
        lambda text: DocumentClassification(
            document_type="job_description", confidence=0.95, reason="looks like a JD"
        ),
    )
    monkeypatch.setattr(graph_module, "extract_resume_profile", lambda text: _resume_profile())
    monkeypatch.setattr(graph_module, "extract_job_profile", lambda text: _job_profile())
    monkeypatch.setattr(
        graph_module,
        "assess_qualitative_match",
        lambda resume, job: QualitativeMatch(
            responsibility_match=80.0,
            responsibility_reasoning="Built APIs before.",
            qualification_match=90.0,
            qualification_reasoning="Meets qualifications.",
        ),
    )
    monkeypatch.setattr(graph_module, "analyze_gaps", lambda resume, job, matches, qualitative: GapList(gaps=[]))
    monkeypatch.setattr(
        graph_module, "generate_recommendations", lambda gaps, breakdown, ats: RecommendationList(recommendations=[])
    )


def test_run_pipeline_with_progress_reports_every_node_in_order(monkeypatch):
    _patch_happy_path(monkeypatch)
    seen_nodes = []

    outcome = run_pipeline_with_progress(
        "unused-resume-path.pdf",
        VALID_JD_TEXT,
        jd_is_file=False,
        on_step=lambda node_name, state: seen_nodes.append(node_name),
    )

    assert seen_nodes == STEP_ORDER
    assert outcome.status == PipelineStatus.COMPLETED
    assert outcome.result is not None
    assert outcome.result.qualitative_match.responsibility_match == 80.0


def test_run_job_marks_all_steps_done_on_completion(monkeypatch):
    _patch_happy_path(monkeypatch)
    job = create_job()

    run_job(job, "unused-resume-path.pdf", VALID_JD_TEXT, jd_is_file=False)

    view = job.to_dict()
    assert view["status"] == "completed"
    assert all(step["state"] == "done" for step in view["steps"])
    assert view["result"]["match_breakdown"]["overall_score"] > 0


def test_run_job_rejection_stops_partway_through_steps(monkeypatch):
    monkeypatch.setattr(pipeline_module, "extract_from_file", lambda path: _resume_doc())
    monkeypatch.setattr(
        graph_module,
        "classify_resume",
        lambda text: DocumentClassification(document_type="timetable", confidence=0.9, reason="looks like a schedule"),
    )

    def fail(*args, **kwargs):
        raise AssertionError("should not run past a rejected resume")

    monkeypatch.setattr(graph_module, "classify_job_description", fail)

    job = create_job()
    run_job(job, "unused-resume-path.pdf", VALID_JD_TEXT, jd_is_file=False)

    view = job.to_dict()
    assert view["status"] == "rejected"
    assert "reason" in view
    steps_by_name = {s["name"]: s["state"] for s in view["steps"]}
    assert steps_by_name["validate_resume"] == "done"
    assert steps_by_name["validate_jd"] == "pending"


def test_run_job_failure_is_captured_not_raised(monkeypatch):
    def boom(path):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(pipeline_module, "extract_from_file", boom)

    job = create_job()
    run_job(job, "unused-resume-path.pdf", VALID_JD_TEXT, jd_is_file=False)

    view = job.to_dict()
    assert view["status"] == "failed"
    assert "disk exploded" in view["error"]
