"""Entry point that runs the compiled graph end-to-end and shapes the final
AnalysisResult (or a rejection) from the resulting state. Used by the CLI,
the synchronous API endpoint, and the async job runner so none of them can
drift apart.
"""

from dataclasses import dataclass
from typing import Callable

from app.graph import get_compiled_graph
from app.ingestion.extract import extract_from_file, extract_plain_text
from app.schemas.analysis import AgentState, AnalysisResult, PipelineStatus

OnStepCallback = Callable[[str, AgentState], None]


@dataclass
class PipelineOutcome:
    status: PipelineStatus
    result: AnalysisResult | None = None
    rejection_reason: str | None = None


def _load_document(path_or_text: str, is_file: bool, filename: str | None = None):
    if is_file:
        return extract_from_file(path_or_text)
    return extract_plain_text(path_or_text, filename=filename)


def _outcome_from_state(final_state: AgentState) -> PipelineOutcome:
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
        qualitative_match=final_state.qualitative_match,
        match_breakdown=final_state.match_breakdown,
        ats_analysis=final_state.ats_analysis,
        gaps=final_state.gaps,
        recommendations=final_state.recommendations,
    )
    return PipelineOutcome(status=final_state.status, result=result)


def run_pipeline_with_progress(
    resume_path: str,
    jd: str,
    jd_is_file: bool = False,
    on_step: OnStepCallback | None = None,
) -> PipelineOutcome:
    """Same as run_pipeline, but calls on_step(node_name, state_so_far) after
    every node LangGraph actually finishes — real progress, not a fake timer."""
    resume_document = extract_from_file(resume_path)
    jd_document = _load_document(jd, is_file=jd_is_file)
    initial_state = AgentState(resume_document=resume_document, jd_document=jd_document)

    graph = get_compiled_graph()
    accumulated = initial_state.model_dump()
    for update in graph.stream(initial_state, stream_mode="updates"):
        for node_name, partial in update.items():
            accumulated.update(partial)
            if on_step is not None:
                on_step(node_name, AgentState.model_validate(accumulated))

    final_state = AgentState.model_validate(accumulated)
    return _outcome_from_state(final_state)


def run_pipeline(
    resume_path: str,
    jd: str,
    jd_is_file: bool = False,
) -> PipelineOutcome:
    return run_pipeline_with_progress(resume_path, jd, jd_is_file=jd_is_file, on_step=None)
