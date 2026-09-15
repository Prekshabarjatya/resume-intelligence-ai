from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.base import LLMSchema
from app.schemas.document import DocumentClassification, ExtractedDocument
from app.schemas.job import JobProfile
from app.schemas.resume import ResumeProfile


class MatchType(str, Enum):
    EXACT = "exact"
    SEMANTIC = "semantic"
    NONE = "none"


class SkillMatch(BaseModel):
    required_skill: str
    importance: str
    match_type: MatchType
    matched_resume_skill: str | None = None
    similarity: float = 0.0
    reasoning: str = Field(
        default="", description="Why this was (or wasn't) judged a match — empty for trivial exact matches."
    )


class ATSIssue(BaseModel):
    issue: str
    severity: str  # "low" | "medium" | "high"


class ATSAnalysis(BaseModel):
    score: float = Field(ge=0.0, le=100.0)
    keyword_coverage: float = Field(ge=0.0, le=100.0)
    issues: list[ATSIssue] = Field(default_factory=list)


class MatchBreakdown(BaseModel):
    overall_score: float = Field(ge=0.0, le=100.0)
    skill_match: float
    experience_match: float
    responsibility_match: float
    keyword_coverage: float
    qualification_match: float
    ats_compatibility: float


class QualitativeMatch(BaseModel):
    responsibility_match: float = Field(ge=0.0, le=100.0)
    responsibility_reasoning: str
    qualification_match: float = Field(ge=0.0, le=100.0)
    qualification_reasoning: str


class GapItem(BaseModel):
    category: str  # "missing_skill" | "experience_gap" | "qualification_gap" | "weak_evidence"
    description: str
    recommendation: str


class Recommendation(BaseModel):
    title: str
    detail: str
    priority: str  # "critical" | "high" | "medium" | "low"


class GapList(LLMSchema):
    gaps: list[GapItem] | None = Field(default_factory=list)


class RecommendationList(LLMSchema):
    recommendations: list[Recommendation] | None = Field(default_factory=list)


class AnalysisResult(BaseModel):
    resume_classification: DocumentClassification
    jd_classification: DocumentClassification
    resume_profile: ResumeProfile
    job_profile: JobProfile
    skill_matches: list[SkillMatch]
    qualitative_match: QualitativeMatch
    match_breakdown: MatchBreakdown
    ats_analysis: ATSAnalysis
    gaps: list[GapItem]
    recommendations: list[Recommendation]


class CandidateStatus(str, Enum):
    SCORED = "scored"
    REJECTED_RESUME = "rejected_resume"
    FAILED = "failed"


class CandidateResult(BaseModel):
    """One resume scored against the shared job profile in a batch/HR run.
    Unlike AnalysisResult, there are no candidate-facing recommendations —
    those are advice addressed to the resume's owner ("add X to your
    resume"), which makes no sense in a screening context. Gaps are kept
    since "what's missing" is neutral, useful screening information."""

    filename: str
    status: CandidateStatus
    rejection_reason: str | None = None
    error: str | None = None
    resume_classification: DocumentClassification | None = None
    resume_profile: ResumeProfile | None = None
    skill_matches: list[SkillMatch] = Field(default_factory=list)
    qualitative_match: QualitativeMatch | None = None
    match_breakdown: MatchBreakdown | None = None
    ats_analysis: ATSAnalysis | None = None
    gaps: list[GapItem] = Field(default_factory=list)


class BatchAnalysisResult(BaseModel):
    jd_classification: DocumentClassification
    job_profile: JobProfile
    candidates: list[CandidateResult]


class PipelineStatus(str, Enum):
    UPLOADED = "uploaded"
    VALIDATING_RESUME = "validating_resume"
    REJECTED_RESUME = "rejected_resume"
    VALIDATING_JD = "validating_jd"
    REJECTED_JD = "rejected_jd"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentState(BaseModel):
    """Shared state threaded through the LangGraph pipeline. Agents read and
    write structured fields here rather than passing free-text between
    each other.
    """

    resume_document: ExtractedDocument | None = None
    jd_document: ExtractedDocument | None = None

    resume_classification: DocumentClassification | None = None
    jd_classification: DocumentClassification | None = None

    resume_profile: ResumeProfile | None = None
    job_profile: JobProfile | None = None

    skill_matches: list[SkillMatch] = Field(default_factory=list)
    qualitative_match: QualitativeMatch | None = None
    ats_analysis: ATSAnalysis | None = None
    match_breakdown: MatchBreakdown | None = None
    gaps: list[GapItem] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)

    status: PipelineStatus = PipelineStatus.UPLOADED
    errors: list[str] = Field(default_factory=list)
