from enum import Enum

from pydantic import Field

from app.schemas.base import LLMSchema


class Importance(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    OPTIONAL = "optional"


class SkillRequirement(LLMSchema):
    skill: str
    importance: Importance
    evidence: str = Field(description="Verbatim excerpt from the JD naming this requirement.")


class JobProfile(LLMSchema):
    role: str | None = None
    company: str | None = None
    seniority: str | None = None
    location: str | None = None
    employment_type: str | None = None
    required_skills: list[SkillRequirement] | None = Field(default_factory=list)
    preferred_skills: list[SkillRequirement] | None = Field(default_factory=list)
    responsibilities: list[str] | None = Field(default_factory=list)
    qualifications: list[str] | None = Field(default_factory=list)
    min_years_experience: float | None = None
    keywords: list[str] | None = Field(default_factory=list)
    soft_skills: list[str] | None = Field(default_factory=list)
