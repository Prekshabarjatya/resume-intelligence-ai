from pydantic import BaseModel, Field


class PersonalInfo(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    links: list[str] = Field(default_factory=list)


class ExperienceEntry(BaseModel):
    title: str
    company: str
    start_date: str | None = None
    end_date: str | None = None
    is_current: bool = False
    bullets: list[str] = Field(default_factory=list)
    source_evidence: str = Field(
        description="Verbatim excerpt from the original resume this entry was derived from."
    )


class EducationEntry(BaseModel):
    degree: str
    institution: str
    start_date: str | None = None
    end_date: str | None = None
    details: str | None = None


class ProjectEntry(BaseModel):
    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    source_evidence: str = ""


class ResumeProfile(BaseModel):
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)
    awards: list[str] = Field(default_factory=list)
    total_years_experience: float | None = Field(
        default=None, description="Deterministically computed from experience dates, not LLM-estimated."
    )
