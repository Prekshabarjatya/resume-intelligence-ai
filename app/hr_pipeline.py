"""HR / batch mode: one job description scored against many resumes, ranked
for screening. This is a different workflow from app/pipeline.py, not a
variant of it — the candidate pipeline produces coaching addressed to the
resume's owner ("add AWS to your resume"), which makes no sense here. HR
mode drops recommendations entirely and keeps only screening-relevant
output: score, skill matches, and gaps (framed as facts, not advice).

Reuses every deterministic tool and LLM agent from the single-candidate
path (app/graph.py, app/tools/scoring.py) unchanged — this is not a second
implementation of the scoring logic, just a different orchestration shape.
A LangGraph StateGraph doesn't fit well here: there's no per-candidate
branching to model, since a rejected resume just gets skipped rather than
aborting the whole run. A plain loop is the honest shape for that.
"""

from dataclasses import dataclass
from typing import Callable

from app.agents.extraction import extract_job_profile, extract_resume_profile
from app.agents.gap_analysis import analyze_gaps
from app.agents.matching import assess_qualitative_match
from app.agents.skill_matching import resolve_unmatched_skills
from app.agents.validation import classify_job_description, classify_resume
from app.graph import JD_CONFIDENCE_THRESHOLD, RESUME_CONFIDENCE_THRESHOLD
from app.ingestion.extract import extract_from_file, extract_plain_text
from app.schemas.analysis import BatchAnalysisResult, CandidateResult, CandidateStatus
from app.schemas.job import JobProfile
from app.tools.scoring import (
    analyze_ats,
    compute_match_breakdown,
    compute_total_years_experience,
    exact_match_skills,
    experience_match_score,
    keyword_coverage_score,
    skill_match_score,
)

OnBatchStepCallback = Callable[[str], None]


@dataclass
class BatchOutcome:
    status: str  # "completed" | "rejected_jd"
    result: BatchAnalysisResult | None = None
    rejection_reason: str | None = None


def candidate_step_name(filename: str) -> str:
    return f"candidate:{filename}"


def score_one_candidate(filename: str, resume_path: str, job_profile: JobProfile) -> CandidateResult:
    resume_document = extract_from_file(resume_path)
    classification = classify_resume(resume_document.raw_text)
    is_valid = (
        classification.is_resume() and classification.confidence >= RESUME_CONFIDENCE_THRESHOLD
    )
    if not is_valid:
        return CandidateResult(
            filename=filename,
            status=CandidateStatus.REJECTED_RESUME,
            resume_classification=classification,
            rejection_reason=classification.reason,
        )

    resume_profile = extract_resume_profile(resume_document.raw_text)
    resume_profile.total_years_experience = compute_total_years_experience(resume_profile)

    all_requirements = job_profile.required_skills + job_profile.preferred_skills
    exact_matches, unmatched = exact_match_skills(all_requirements, resume_profile.skills)
    semantic_matches = resolve_unmatched_skills(unmatched, resume_profile.skills)
    skill_matches = exact_matches + semantic_matches

    qualitative = assess_qualitative_match(resume_profile, job_profile)

    kw_score = keyword_coverage_score(resume_document.raw_text, job_profile.keywords)
    ats = analyze_ats(resume_profile, resume_document.raw_text, kw_score)
    sk_score = skill_match_score(skill_matches)
    exp_score = experience_match_score(job_profile, resume_profile.total_years_experience)
    breakdown = compute_match_breakdown(
        skill_match=sk_score,
        experience_match=exp_score,
        responsibility_match=qualitative.responsibility_match,
        keyword_coverage=kw_score,
        qualification_match=qualitative.qualification_match,
        ats_compatibility=ats.score,
    )

    gap_list = analyze_gaps(resume_profile, job_profile, skill_matches, qualitative)

    return CandidateResult(
        filename=filename,
        status=CandidateStatus.SCORED,
        resume_classification=classification,
        resume_profile=resume_profile,
        skill_matches=skill_matches,
        qualitative_match=qualitative,
        match_breakdown=breakdown,
        ats_analysis=ats,
        gaps=gap_list.gaps,
    )


def run_batch_pipeline(
    jd: str,
    resumes: list[tuple[str, str]],
    jd_is_file: bool = False,
    on_step: OnBatchStepCallback | None = None,
) -> BatchOutcome:
    """resumes: list of (display_filename, temp_file_path)."""

    def step(name: str) -> None:
        if on_step is not None:
            on_step(name)

    jd_document = extract_from_file(jd) if jd_is_file else extract_plain_text(jd)
    jd_classification = classify_job_description(jd_document.raw_text)
    step("validate_jd")

    is_jd_valid = (
        jd_classification.is_job_description()
        and jd_classification.confidence >= JD_CONFIDENCE_THRESHOLD
    )
    if not is_jd_valid:
        return BatchOutcome(
            status="rejected_jd",
            rejection_reason=f"Uploaded job description was rejected: {jd_classification.reason}",
        )

    job_profile = extract_job_profile(jd_document.raw_text)
    step("extract_jd")

    candidates: list[CandidateResult] = []
    for filename, path in resumes:
        try:
            candidates.append(score_one_candidate(filename, path, job_profile))
        except Exception as exc:  # noqa: BLE001 - one bad resume shouldn't sink the batch
            candidates.append(CandidateResult(filename=filename, status=CandidateStatus.FAILED, error=str(exc)))
        step(candidate_step_name(filename))

    scored = sorted(
        (c for c in candidates if c.status == CandidateStatus.SCORED),
        key=lambda c: c.match_breakdown.overall_score,
        reverse=True,
    )
    unscored = [c for c in candidates if c.status != CandidateStatus.SCORED]

    result = BatchAnalysisResult(
        jd_classification=jd_classification, job_profile=job_profile, candidates=scored + unscored
    )
    return BatchOutcome(status="completed", result=result)
