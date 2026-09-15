"""Semantic skill-synonym judgment — real LLM call.

Exact/substring skill matching is deterministic (app/tools/scoring.py).
For everything left unmatched, a generic sentence-embedding model turned out
to be unreliable on short skill phrases (e.g. "Large Language Model
applications" vs "LLM applications using GPT APIs" scored *lower* than
"AWS" vs "Java" in a quick check) — not something to threshold a real score
on. An LLM judging synonymy is more reliable here, and it's grounded
strictly in the candidate's own stated skill list: it can only point at a
skill that's actually listed, or say none matches — it can never invent one.
"""

from pydantic import Field

from app.config import settings
from app.schemas.analysis import MatchType, SkillMatch
from app.schemas.base import LLMSchema
from app.schemas.job import SkillRequirement
from app.tools.llm import get_classifier_llm

_SYSTEM_PROMPT = """You judge whether a job's required/preferred skill is
genuinely satisfied by any skill the candidate has actually listed.

For each target skill, choose the single best candidate skill from the
provided list that is a true synonym, alternate name, or direct subset/superset
of the same technology (e.g. "LLM applications" and "GPT API development" are
equivalent; "Postgres" and "PostgreSQL" are equivalent). Do NOT match skills
that are merely in the same general domain but functionally different (e.g.
AWS and Python are NOT a match; React and Vue are NOT a match).

If nothing in the candidate's list is a genuine equivalent, set
matched_resume_skill to null and is_match to false. Only ever choose
matched_resume_skill from the exact strings given in the candidate's skill
list — never invent a skill."""


class SkillMatchJudgment(LLMSchema):
    required_skill: str
    matched_resume_skill: str | None
    is_match: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class SkillMatchJudgmentList(LLMSchema):
    judgments: list[SkillMatchJudgment] | None = Field(default_factory=list)


def judge_semantic_skill_matches(
    unmatched_required_skills: list[str], resume_skills: list[str]
) -> SkillMatchJudgmentList:
    if not unmatched_required_skills or not resume_skills:
        return SkillMatchJudgmentList(
            judgments=[
                SkillMatchJudgment(
                    required_skill=s,
                    matched_resume_skill=None,
                    is_match=False,
                    confidence=0.0,
                    reasoning="Candidate has no listed skills to compare against.",
                )
                for s in unmatched_required_skills
            ]
        )

    llm = get_classifier_llm().with_structured_output(SkillMatchJudgmentList)
    return llm.invoke(
        [
            ("system", _SYSTEM_PROMPT),
            (
                "human",
                f"TARGET SKILLS TO CHECK:\n{unmatched_required_skills}\n\n"
                f"CANDIDATE'S ACTUAL LISTED SKILLS:\n{resume_skills}",
            ),
        ]
    )


def resolve_unmatched_skills(
    unmatched_requirements: list[SkillRequirement], resume_skills: list[str]
) -> list[SkillMatch]:
    """Runs the LLM synonym judgment for every requirement exact-matching
    missed, and converts the result into SkillMatch records. A judgment is
    only accepted as SEMANTIC if is_match is true AND confidence clears the
    configured threshold — a low-confidence "yes" from the model still
    counts as no match, per the "never claim a skill exists just because
    it's adjacent" rule."""
    if not unmatched_requirements:
        return []

    judgment_list = judge_semantic_skill_matches(
        [r.skill for r in unmatched_requirements], resume_skills
    )
    judgments_by_skill = {j.required_skill: j for j in judgment_list.judgments}

    results: list[SkillMatch] = []
    for req in unmatched_requirements:
        judgment = judgments_by_skill.get(req.skill)
        accepted = (
            judgment is not None
            and judgment.is_match
            and judgment.matched_resume_skill in resume_skills
            and judgment.confidence >= settings.semantic_match_threshold
        )
        if accepted:
            results.append(
                SkillMatch(
                    required_skill=req.skill,
                    importance=req.importance.value,
                    match_type=MatchType.SEMANTIC,
                    matched_resume_skill=judgment.matched_resume_skill,
                    similarity=judgment.confidence,
                )
            )
        else:
            results.append(
                SkillMatch(
                    required_skill=req.skill,
                    importance=req.importance.value,
                    match_type=MatchType.NONE,
                    matched_resume_skill=None,
                    similarity=judgment.confidence if judgment else 0.0,
                )
            )
    return results
