"""LangGraph orchestration for the core pipeline:

validate_resume -> validate_jd -> extract_resume -> extract_jd
    -> match_skills -> assess_qualitative -> analyze_ats -> score
    -> analyze_gaps -> recommend -> END

Either validation step can short-circuit straight to END with a rejection
recorded in state.status. Deterministic steps (skill matching, ATS, scoring)
never call the LLM; the rest are real Groq calls (see app/agents/*).
"""

from langgraph.graph import END, StateGraph

from app.agents.extraction import extract_job_profile, extract_resume_profile
from app.agents.gap_analysis import analyze_gaps
from app.agents.matching import assess_qualitative_match
from app.agents.recommendations import generate_recommendations
from app.agents.skill_matching import resolve_unmatched_skills
from app.agents.validation import classify_job_description, classify_resume
from app.schemas.analysis import AgentState, PipelineStatus
from app.tools.scoring import (
    analyze_ats,
    compute_match_breakdown,
    compute_total_years_experience,
    exact_match_skills,
    experience_match_score,
    keyword_coverage_score,
    skill_match_score,
)

RESUME_CONFIDENCE_THRESHOLD = 0.6
JD_CONFIDENCE_THRESHOLD = 0.6


def node_validate_resume(state: AgentState) -> dict:
    classification = classify_resume(state.resume_document.raw_text)
    is_valid = (
        classification.document_type == "resume" and classification.confidence >= RESUME_CONFIDENCE_THRESHOLD
    )
    return {
        "resume_classification": classification,
        "status": PipelineStatus.VALIDATING_RESUME if is_valid else PipelineStatus.REJECTED_RESUME,
    }


def route_after_resume_validation(state: AgentState) -> str:
    return "rejected" if state.status == PipelineStatus.REJECTED_RESUME else "continue"


def node_validate_jd(state: AgentState) -> dict:
    classification = classify_job_description(state.jd_document.raw_text)
    is_valid = (
        classification.document_type == "job_description"
        and classification.confidence >= JD_CONFIDENCE_THRESHOLD
    )
    return {
        "jd_classification": classification,
        "status": PipelineStatus.VALIDATING_JD if is_valid else PipelineStatus.REJECTED_JD,
    }


def route_after_jd_validation(state: AgentState) -> str:
    return "rejected" if state.status == PipelineStatus.REJECTED_JD else "continue"


def node_extract_resume(state: AgentState) -> dict:
    profile = extract_resume_profile(state.resume_document.raw_text)
    profile.total_years_experience = compute_total_years_experience(profile)
    return {"resume_profile": profile, "status": PipelineStatus.EXTRACTING}


def node_extract_jd(state: AgentState) -> dict:
    profile = extract_job_profile(state.jd_document.raw_text)
    return {"job_profile": profile}


def node_match_skills(state: AgentState) -> dict:
    all_requirements = state.job_profile.required_skills + state.job_profile.preferred_skills
    exact_matches, unmatched = exact_match_skills(all_requirements, state.resume_profile.skills)
    semantic_matches = resolve_unmatched_skills(unmatched, state.resume_profile.skills)
    return {"skill_matches": exact_matches + semantic_matches, "status": PipelineStatus.ANALYZING}


def node_assess_qualitative(state: AgentState) -> dict:
    qualitative = assess_qualitative_match(state.resume_profile, state.job_profile)
    return {"qualitative_match": qualitative}


def node_analyze_ats_and_score(state: AgentState) -> dict:
    qualitative = state.qualitative_match
    resume_text = state.resume_document.raw_text
    kw_score = keyword_coverage_score(resume_text, state.job_profile.keywords)
    ats = analyze_ats(state.resume_profile, resume_text, kw_score)
    sk_score = skill_match_score(state.skill_matches)
    exp_score = experience_match_score(state.job_profile, state.resume_profile.total_years_experience)

    breakdown = compute_match_breakdown(
        skill_match=sk_score,
        experience_match=exp_score,
        responsibility_match=qualitative.responsibility_match if qualitative else 0.0,
        keyword_coverage=kw_score,
        qualification_match=qualitative.qualification_match if qualitative else 0.0,
        ats_compatibility=ats.score,
    )
    return {"ats_analysis": ats, "match_breakdown": breakdown}


def node_analyze_gaps(state: AgentState) -> dict:
    gap_list = analyze_gaps(
        state.resume_profile, state.job_profile, state.skill_matches, state.qualitative_match
    )
    return {"gaps": gap_list.gaps}


def node_recommend(state: AgentState) -> dict:
    rec_list = generate_recommendations(state.gaps, state.match_breakdown, state.ats_analysis)
    return {"recommendations": rec_list.recommendations, "status": PipelineStatus.COMPLETED}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("validate_resume", node_validate_resume)
    graph.add_node("validate_jd", node_validate_jd)
    graph.add_node("extract_resume", node_extract_resume)
    graph.add_node("extract_jd", node_extract_jd)
    graph.add_node("match_skills", node_match_skills)
    graph.add_node("assess_qualitative", node_assess_qualitative)
    graph.add_node("analyze_ats_and_score", node_analyze_ats_and_score)
    graph.add_node("analyze_gaps", node_analyze_gaps)
    graph.add_node("recommend", node_recommend)

    graph.set_entry_point("validate_resume")
    graph.add_conditional_edges(
        "validate_resume", route_after_resume_validation, {"rejected": END, "continue": "validate_jd"}
    )
    graph.add_conditional_edges(
        "validate_jd", route_after_jd_validation, {"rejected": END, "continue": "extract_resume"}
    )
    graph.add_edge("extract_resume", "extract_jd")
    graph.add_edge("extract_jd", "match_skills")
    graph.add_edge("match_skills", "assess_qualitative")
    graph.add_edge("assess_qualitative", "analyze_ats_and_score")
    graph.add_edge("analyze_ats_and_score", "analyze_gaps")
    graph.add_edge("analyze_gaps", "recommend")
    graph.add_edge("recommend", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
