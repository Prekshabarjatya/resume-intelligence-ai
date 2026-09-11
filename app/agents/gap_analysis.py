"""Gap analysis agent — real LLM call.

Turns the deterministic skill-match results and the qualitative match
judgment into concrete, evidence-based gaps. Explicitly instructed not to
suggest fabricating experience — only to flag what's missing and, if genuine,
to surface it.
"""

from app.schemas.analysis import GapList, MatchType, QualitativeMatch, SkillMatch
from app.schemas.job import JobProfile
from app.schemas.resume import ResumeProfile
from app.tools.llm import get_reasoning_llm

_SYSTEM_PROMPT = """You are a gap analysis agent for a resume/job-fit tool.
Given the candidate profile, job profile, computed skill matches, and a
qualitative match assessment, identify concrete gaps.

Categories: "missing_skill", "experience_gap", "qualification_gap", "weak_evidence".

Rules:
- Only report gaps clearly supported by the data given — do not speculate.
- For missing_skill gaps, phrase the recommendation conditionally: e.g.
  "If you genuinely have experience with X, add it — do not fabricate it."
- Never recommend inventing companies, dates, metrics, or credentials.
- Keep each gap description to one sentence and each recommendation to one
  actionable sentence."""


def analyze_gaps(
    resume: ResumeProfile,
    job: JobProfile,
    skill_matches: list[SkillMatch],
    qualitative: QualitativeMatch,
) -> GapList:
    missing = [m for m in skill_matches if m.match_type == MatchType.NONE]
    weak = [m for m in skill_matches if m.match_type == MatchType.SEMANTIC and m.similarity < 0.75]

    llm = get_reasoning_llm().with_structured_output(GapList)
    return llm.invoke(
        [
            ("system", _SYSTEM_PROMPT),
            (
                "human",
                "CANDIDATE PROFILE (JSON):\n"
                f"{resume.model_dump_json(indent=2)}\n\n"
                "JOB PROFILE (JSON):\n"
                f"{job.model_dump_json(indent=2)}\n\n"
                "MISSING SKILLS:\n"
                f"{[m.required_skill for m in missing]}\n\n"
                "WEAKLY-MATCHED SKILLS (semantic, low confidence):\n"
                f"{[(m.required_skill, m.matched_resume_skill, m.similarity) for m in weak]}\n\n"
                "QUALITATIVE MATCH ASSESSMENT (JSON):\n"
                f"{qualitative.model_dump_json(indent=2)}",
            ),
        ]
    )
