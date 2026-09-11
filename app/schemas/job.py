from enum import Enum

from pydantic import BaseModel, Field


class Importance(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    OPTIONAL = "optional"


class SkillRequirement(BaseModel):
    skill: str
    importance: Importance
    evidence: str = Field(description="Verbatim excerpt from the JD naming this requirement.")


class JobProfile(BaseModel):
    role: str | None = None
    company: str | None = None
    seniority: str | None = None
    location: str | None = None
    employment_type: str | None = None
    required_skills: list[SkillRequirement] = Field(default_factory=list)
    preferred_skills: list[SkillRequirement] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    qualifications: list[str] = Field(default_factory=list)
    min_years_experience: float | None = None
    keywords: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
