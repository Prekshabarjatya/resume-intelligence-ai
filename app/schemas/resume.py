from pydantic import BaseModel, Field

from app.schemas.base import LLMSchema


class PersonalInfo(LLMSchema):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    links: list[str] | None = Field(default_factory=list)


class ExperienceEntry(LLMSchema):
    title: str
    company: str
    start_date: str | None = None
    end_date: str | None = None
    is_current: bool = False
    bullets: list[str] | None = Field(default_factory=list)
    source_evidence: str = Field(
        description="Verbatim excerpt from the original resume this entry was derived from."
    )


class EducationEntry(BaseModel):
    degree: str
    institution: str
    start_date: str | None = None
    end_date: str | None = None
    details: str | None = None


class ProjectEntry(LLMSchema):
    name: str
    description: str
    technologies: list[str] | None = Field(default_factory=list)
    source_evidence: str = ""


class ResumeProfile(LLMSchema):
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    summary: str | None = None
    skills: list[str] | None = Field(default_factory=list)
    experience: list[ExperienceEntry] | None = Field(default_factory=list)
    education: list[EducationEntry] | None = Field(default_factory=list)
    projects: list[ProjectEntry] | None = Field(default_factory=list)
    certifications: list[str] | None = Field(default_factory=list)
    achievements: list[str] | None = Field(default_factory=list)
    publications: list[str] | None = Field(default_factory=list)
    awards: list[str] | None = Field(default_factory=list)
    total_years_experience: float | None = Field(
        default=None, description="Deterministically computed from experience dates, not LLM-estimated."
    )
