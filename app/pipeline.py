"""Entry point that runs the compiled graph end-to-end and shapes the final
AnalysisResult (or a rejection) from the resulting state. Used by both the
CLI and the FastAPI endpoint so they can't drift apart.
"""

from dataclasses import dataclass

from app.graph import get_compiled_graph
from app.ingestion.extract import extract_from_file, extract_plain_text
from app.schemas.analysis import AgentState, AnalysisResult, PipelineStatus


@dataclass
class PipelineOutcome:
    status: PipelineStatus
    result: AnalysisResult | None = None
    rejection_reason: str | None = None


def _load_document(path_or_text: str, is_file: bool, filename: str | None = None):
    if is_file:
        return extract_from_file(path_or_text)
    return extract_plain_text(path_or_text, filename=filename)


def run_pipeline(
    resume_path: str,
    jd: str,
    jd_is_file: bool = False,
) -> PipelineOutcome:
    resume_document = extract_from_file(resume_path)
    jd_document = _load_document(jd, is_file=jd_is_file)

    initial_state = AgentState(resume_document=resume_document, jd_document=jd_document)

    graph = get_compiled_graph()
    final_state_dict = graph.invoke(initial_state)
    final_state = AgentState.model_validate(final_state_dict)

    if final_state.status == PipelineStatus.REJECTED_RESUME:
        return PipelineOutcome(
            status=final_state.status,
            rejection_reason=(
                f"Uploaded resume was rejected: {final_state.resume_classification.reason}"
            ),
        )

    if final_state.status == PipelineStatus.REJECTED_JD:
        return PipelineOutcome(
            status=final_state.status,
            rejection_reason=(
                f"Uploaded job description was rejected: {final_state.jd_classification.reason}"
            ),
        )

    result = AnalysisResult(
        resume_classification=final_state.resume_classification,
        jd_classification=final_state.jd_classification,
        resume_profile=final_state.resume_profile,
        job_profile=final_state.job_profile,
        skill_matches=final_state.skill_matches,
        match_breakdown=final_state.match_breakdown,
        ats_analysis=final_state.ats_analysis,
        gaps=final_state.gaps,
        recommendations=final_state.recommendations,
    )
    return PipelineOutcome(status=final_state.status, result=result)
