"""Qualitative matching agent.

Skill matching is deterministic (see app/tools/scoring.py) because it must
never hallucinate a skill into existence. Responsibility and qualification
fit, though, are genuine reading-comprehension judgments — "has this person
actually done work like what this role requires?" — so this is a real LLM
call, grounded in the already-extracted structured profiles rather than raw
text (smaller surface for injected instructions, and no re-reading of the
untrusted document).
"""

from app.schemas.analysis import QualitativeMatch
from app.schemas.job import JobProfile
from app.schemas.resume import ResumeProfile
from app.tools.llm import get_reasoning_llm

_SYSTEM_PROMPT = """You are assessing candidate fit for a role based on two
structured profiles (already extracted from a resume and a job description).
Do not invent facts beyond what's in these profiles.

Score responsibility_match (0-100): how well the candidate's actual experience
and projects demonstrate the responsibilities listed for this role.

Score qualification_match (0-100): how well the candidate's education,
certifications, and stated qualifications meet the role's qualifications.

Give a one-to-two sentence reasoning for each score, citing specific items
from the profiles."""


def assess_qualitative_match(resume: ResumeProfile, job: JobProfile) -> QualitativeMatch:
    llm = get_reasoning_llm().with_structured_output(QualitativeMatch)
    return llm.invoke(
        [
            ("system", _SYSTEM_PROMPT),
            (
                "human",
                "CANDIDATE PROFILE (JSON):\n"
                f"{resume.model_dump_json(indent=2)}\n\n"
                "JOB PROFILE (JSON):\n"
                f"{job.model_dump_json(indent=2)}",
            ),
        ]
    )
