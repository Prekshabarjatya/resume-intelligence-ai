"""Deterministic scoring: skill matching, keyword coverage, experience
duration, ATS heuristics, and the final weighted match score. None of this
is delegated to the LLM — every number here is reproducible and explainable,
per the PRD's "scores should be explainable" principle.
"""

import re
from datetime import date, datetime

from dateutil import parser as date_parser

from app.config import settings
from app.schemas.analysis import ATSAnalysis, ATSIssue, MatchBreakdown, MatchType, SkillMatch
from app.schemas.job import JobProfile, SkillRequirement
from app.schemas.resume import ResumeProfile


def _normalize(text: str) -> str:
    """Lowercases and reduces text to words. Anything that isn't a letter,
    digit, '+', '.', '#' or space becomes a space (so newlines and hyphens
    separate words instead of gluing them together)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+.# ]", " ", text.lower())).strip()


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Whole-word containment on normalized text. "java" is not found in
    "javascript", "go" not in "google", "c" not in "c++" or "docker"; a
    sentence-ending period after the phrase is allowed ("... using python.")."""
    if not needle:
        return False
    pattern = rf"(?<![a-z0-9+.#]){re.escape(needle)}(?![a-z0-9+#])(?!\.[a-z0-9])"
    return re.search(pattern, haystack) is not None


def _exact_match_one(requirement: SkillRequirement, resume_skills: list[str]) -> SkillMatch | None:
    """Case/punctuation-insensitive exact or substring match. Returns None
    (not a MatchType.NONE SkillMatch) when nothing matches, so the caller
    knows this requirement still needs the semantic judgment call."""
    norm_req = _normalize(requirement.skill)
    if not norm_req:
        return None
    normalized_resume = {_normalize(s): s for s in resume_skills}

    if norm_req in normalized_resume:
        return SkillMatch(
            required_skill=requirement.skill,
            importance=requirement.importance.value,
            match_type=MatchType.EXACT,
            matched_resume_skill=normalized_resume[norm_req],
            similarity=1.0,
        )

    for norm_skill, original in normalized_resume.items():
        if norm_skill and (_contains_phrase(norm_skill, norm_req) or _contains_phrase(norm_req, norm_skill)):
            return SkillMatch(
                required_skill=requirement.skill,
                importance=requirement.importance.value,
                match_type=MatchType.EXACT,
                matched_resume_skill=original,
                similarity=1.0,
            )
    return None


def exact_match_skills(
    requirements: list[SkillRequirement], resume_skills: list[str]
) -> tuple[list[SkillMatch], list[SkillRequirement]]:
    """Deterministic first pass. Returns (matches, still_unmatched) — the
    unmatched requirements are what the LLM synonym-judgment agent
    (app/agents/skill_matching.py) needs to look at next."""
    matches: list[SkillMatch] = []
    unmatched: list[SkillRequirement] = []
    for req in requirements:
        match = _exact_match_one(req, resume_skills)
        if match is not None:
            matches.append(match)
        else:
            unmatched.append(req)
    return matches, unmatched


def skill_match_score(skill_matches: list[SkillMatch]) -> float:
    """Weighted so required skills matter far more than preferred ones."""
    if not skill_matches:
        return 100.0

    weight_by_importance = {"required": 2.0, "preferred": 1.0, "optional": 0.5}
    total_weight = 0.0
    earned_weight = 0.0
    for match in skill_matches:
        w = weight_by_importance.get(match.importance, 1.0)
        total_weight += w
        if match.match_type == MatchType.EXACT:
            earned_weight += w
        elif match.match_type == MatchType.SEMANTIC:
            earned_weight += w * match.similarity

    if total_weight == 0:
        return 100.0
    return round(100 * earned_weight / total_weight, 1)


def keyword_coverage_score(resume_text: str, keywords: list[str]) -> float:
    keywords = [kw for kw in keywords if _normalize(kw)]
    if not keywords:
        return 100.0
    normalized_resume = _normalize(resume_text)
    hits = sum(1 for kw in keywords if _contains_phrase(normalized_resume, _normalize(kw)))
    return round(100 * hits / len(keywords), 1)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    if value.strip().lower() in {"present", "current", "now", "ongoing"}:
        return date.today()
    try:
        return date_parser.parse(value, default=datetime(2000, 1, 1)).date()
    except (ValueError, OverflowError):
        return None


def compute_total_years_experience(resume: ResumeProfile) -> float | None:
    """Sums non-overlapping-ish experience by just summing each entry's
    duration. Simple and deterministic; does not attempt overlap merging."""
    total_days = 0
    found_any = False
    for entry in resume.experience:
        start = _parse_date(entry.start_date)
        end = _parse_date(entry.end_date) if not entry.is_current else date.today()
        if start and end and end >= start:
            total_days += (end - start).days
            found_any = True
    if not found_any:
        return None
    return round(total_days / 365.25, 1)


def experience_match_score(job_profile: JobProfile, total_years: float | None) -> float:
    if job_profile.min_years_experience is None:
        return 100.0
    if total_years is None:
        return 0.0
    if total_years >= job_profile.min_years_experience:
        return 100.0
    return round(100 * total_years / job_profile.min_years_experience, 1)


STANDARD_SECTION_HEADINGS = [
    "experience",
    "education",
    "skills",
]


def analyze_ats(resume: ResumeProfile, resume_text: str, keyword_coverage: float) -> ATSAnalysis:
    issues: list[ATSIssue] = []

    if not resume.personal_info.email:
        issues.append(ATSIssue(issue="No email address detected.", severity="high"))
    if not resume.personal_info.phone:
        issues.append(ATSIssue(issue="No phone number detected.", severity="medium"))

    normalized = _normalize(resume_text)
    for heading in STANDARD_SECTION_HEADINGS:
        if heading not in normalized:
            issues.append(
                ATSIssue(issue=f"Missing a standard '{heading}' section heading.", severity="medium")
            )

    if not resume.skills:
        issues.append(ATSIssue(issue="No skills section detected.", severity="high"))

    if len(resume_text) < 200:
        issues.append(
            ATSIssue(issue="Very little extractable text — check for image-based content.", severity="high")
        )

    severity_penalty = {"high": 15, "medium": 8, "low": 3}
    penalty = sum(severity_penalty.get(i.severity, 5) for i in issues)
    structure_score = max(0.0, 100.0 - penalty)

    score = round(0.5 * structure_score + 0.5 * keyword_coverage, 1)

    return ATSAnalysis(score=score, keyword_coverage=keyword_coverage, issues=issues)


def compute_match_breakdown(
    skill_match: float,
    experience_match: float,
    responsibility_match: float,
    keyword_coverage: float,
    qualification_match: float,
    ats_compatibility: float,
) -> MatchBreakdown:
    overall = (
        settings.weight_skill_match * skill_match
        + settings.weight_experience_match * experience_match
        + settings.weight_responsibility_match * responsibility_match
        + settings.weight_keyword_coverage * keyword_coverage
        + settings.weight_qualification_match * qualification_match
        + settings.weight_ats_compatibility * ats_compatibility
    )
    return MatchBreakdown(
        overall_score=round(overall, 1),
        skill_match=skill_match,
        experience_match=experience_match,
        responsibility_match=responsibility_match,
        keyword_coverage=keyword_coverage,
        qualification_match=qualification_match,
        ats_compatibility=ats_compatibility,
    )
