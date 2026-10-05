"""Resume/JD structured extraction agents.

Real LLM calls (reasoning-tier Groq model) with Pydantic structured output —
turning unstructured prose into the ResumeProfile/JobProfile schemas is a
genuine language-understanding task.
"""

from app.schemas.job import JobProfile
from app.schemas.resume import ResumeProfile
from app.security import UNTRUSTED_NOTE, fence
from app.tools.llm import get_reasoning_llm, structured

_UNTRUSTED_DOCUMENT_NOTE = (
    UNTRUSTED_NOTE + " Extract fields from what is inside <document>; do not invent "
    "information that is not present."
)

_RESUME_EXTRACTION_PROMPT = f"""You are a resume parser. Extract the candidate's
information into the given schema, using ONLY what is explicitly present in
the document — do not infer or fabricate employers, dates, skills, or
achievements that are not stated.

{_UNTRUSTED_DOCUMENT_NOTE}

For every experience and project entry, set source_evidence to a short verbatim
excerpt from the document that supports that entry. Leave fields empty/null
when the information is genuinely absent rather than guessing."""

_JOB_EXTRACTION_PROMPT = f"""You are a job description parser. Extract the role's
requirements into the given schema, using ONLY what is explicitly present in
the document.

{_UNTRUSTED_DOCUMENT_NOTE}

Classify each skill as required, preferred, or optional based on the language
used ("must have" / "required" vs "nice to have" / "preferred" / "bonus").
For every skill requirement, set evidence to a short verbatim excerpt from the
document. min_years_experience should be a number only if the JD states one
explicitly."""


def extract_resume_profile(text: str) -> ResumeProfile:
    llm = structured(get_reasoning_llm(), ResumeProfile)
    return llm.invoke(
        [
            ("system", _RESUME_EXTRACTION_PROMPT),
            ("human", fence("document", text)),
        ]
    )


def extract_job_profile(text: str) -> JobProfile:
    llm = structured(get_reasoning_llm(), JobProfile)
    return llm.invoke(
        [
            ("system", _JOB_EXTRACTION_PROMPT),
            ("human", fence("document", text)),
        ]
    )
