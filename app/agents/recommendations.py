"""Recommendation agent — real LLM call.

Turns gaps + scores into prioritized, actionable recommendations. This is
the last LLM step in the core pipeline (optimization/generation is out of
scope for this slice).
"""

from app.schemas.analysis import ATSAnalysis, GapItem, MatchBreakdown, RecommendationList
from app.tools.llm import get_reasoning_llm

_SYSTEM_PROMPT = """You are a career coach producing prioritized recommendations
for a candidate applying to a specific role, based on gap analysis, ATS
findings, and match scores already computed.

Prioritize: "critical" (blocks the application), "high", "medium", "low".
Do not suggest fabricating any experience, credential, or metric — only
suggest surfacing/emphasizing real content, restructuring, or rewording.
Produce at most 8 recommendations, most important first."""


def generate_recommendations(
    gaps: list[GapItem],
    match_breakdown: MatchBreakdown,
    ats_analysis: ATSAnalysis,
) -> RecommendationList:
    llm = get_reasoning_llm().with_structured_output(RecommendationList)
    return llm.invoke(
        [
            ("system", _SYSTEM_PROMPT),
            (
                "human",
                "GAPS (JSON):\n"
                f"{[g.model_dump() for g in gaps]}\n\n"
                "MATCH BREAKDOWN (JSON):\n"
                f"{match_breakdown.model_dump_json(indent=2)}\n\n"
                "ATS ANALYSIS (JSON):\n"
                f"{ats_analysis.model_dump_json(indent=2)}",
            ),
        ]
    )
