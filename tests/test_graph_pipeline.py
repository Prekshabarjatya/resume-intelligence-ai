"""End-to-end graph tests with every LLM call mocked out, so this suite runs
without a GROQ_API_KEY or network access. Verifies routing (rejection
short-circuits) and that a happy path produces a complete AnalysisResult."""

import app.graph as graph_module
from app.schemas.analysis import (
    ATSAnalysis,
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


def _doc(text: str) -> ExtractedDocument:
    return ExtractedDocument(source_format=SourceFormat.TEXT, raw_text=text)


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
        required_skills=[SkillRequirement(skill="Python", importance=Importance.REQUIRED, evidence="Required: Python")],
        responsibilities=["build APIs"],
        keywords=["python"],
    )


def _patch_happy_path(monkeypatch):
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
    monkeypatch.setattr(
        graph_module,
        "analyze_gaps",
        lambda resume, job, matches, qualitative: GapList(gaps=[]),
    )
    monkeypatch.setattr(
        graph_module,
        "generate_recommendations",
        lambda gaps, breakdown, ats: RecommendationList(recommendations=[]),
    )


def test_happy_path_completes(monkeypatch):
    _patch_happy_path(monkeypatch)
    graph = graph_module.build_graph()

    from app.schemas.analysis import AgentState

    initial_state = AgentState(resume_document=_doc(VALID_RESUME_TEXT), jd_document=_doc(VALID_JD_TEXT))
    final = graph.invoke(initial_state)
    final_state = AgentState.model_validate(final)

    assert final_state.status == PipelineStatus.COMPLETED
    assert final_state.resume_profile is not None
    assert final_state.job_profile is not None
    assert len(final_state.skill_matches) == 1
    assert final_state.skill_matches[0].matched_resume_skill == "Python"
    assert final_state.match_breakdown is not None
    assert final_state.match_breakdown.overall_score > 0


def test_invalid_resume_short_circuits(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "classify_resume",
        lambda text: DocumentClassification(document_type="timetable", confidence=0.9, reason="looks like a class schedule"),
    )

    def fail(*args, **kwargs):
        raise AssertionError("should not be called when resume is rejected")

    monkeypatch.setattr(graph_module, "classify_job_description", fail)
    monkeypatch.setattr(graph_module, "extract_resume_profile", fail)

    from app.schemas.analysis import AgentState

    graph = graph_module.build_graph()
    initial_state = AgentState(resume_document=_doc("Monday 9am Math, Tuesday 10am Physics"), jd_document=_doc(VALID_JD_TEXT))
    final = graph.invoke(initial_state)
    final_state = AgentState.model_validate(final)

    assert final_state.status == PipelineStatus.REJECTED_RESUME
    assert final_state.job_profile is None


def test_invalid_jd_short_circuits(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "classify_resume",
        lambda text: DocumentClassification(document_type="resume", confidence=0.95, reason="looks like a resume"),
    )
    monkeypatch.setattr(
        graph_module,
        "classify_job_description",
        lambda text: DocumentClassification(document_type="article", confidence=0.9, reason="reads like a blog post"),
    )

    def fail(*args, **kwargs):
        raise AssertionError("should not be called when JD is rejected")

    monkeypatch.setattr(graph_module, "extract_resume_profile", fail)
    monkeypatch.setattr(graph_module, "extract_job_profile", fail)

    from app.schemas.analysis import AgentState

    graph = graph_module.build_graph()
    initial_state = AgentState(resume_document=_doc(VALID_RESUME_TEXT), jd_document=_doc("Some blog post about backend engineering trends."))
    final = graph.invoke(initial_state)
    final_state = AgentState.model_validate(final)

    assert final_state.status == PipelineStatus.REJECTED_JD
    assert final_state.resume_profile is None
