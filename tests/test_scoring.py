from app.schemas.job import Importance, SkillRequirement
from app.schemas.resume import EducationEntry, ExperienceEntry, PersonalInfo, ResumeProfile
from app.tools.scoring import (
    analyze_ats,
    compute_match_breakdown,
    compute_total_years_experience,
    exact_match_skills,
    experience_match_score,
    keyword_coverage_score,
    skill_match_score,
)


def make_resume(**overrides) -> ResumeProfile:
    defaults = dict(
        personal_info=PersonalInfo(name="Jane Doe", email="jane@example.com", phone="555-1234"),
        summary="Backend engineer.",
        skills=["Python", "FastAPI", "PostgreSQL"],
        experience=[
            ExperienceEntry(
                title="Software Engineer",
                company="Acme Corp",
                start_date="2021-01-01",
                end_date="2023-01-01",
                bullets=["Built things."],
                source_evidence="Software Engineer, Acme Corp, 2021-2023",
            )
        ],
        education=[EducationEntry(degree="B.S. Computer Science", institution="State University")],
    )
    defaults.update(overrides)
    return ResumeProfile(**defaults)


def test_exact_match_case_and_punctuation_insensitive():
    reqs = [SkillRequirement(skill="python", importance=Importance.REQUIRED, evidence="must know Python")]
    matches, unmatched = exact_match_skills(reqs, ["Python", "FastAPI"])
    assert len(matches) == 1
    assert unmatched == []
    assert matches[0].matched_resume_skill == "Python"
    assert matches[0].similarity == 1.0


def test_exact_match_substring():
    reqs = [
        SkillRequirement(skill="PostgreSQL", importance=Importance.REQUIRED, evidence="needs Postgres")
    ]
    matches, unmatched = exact_match_skills(reqs, ["Postgres/PostgreSQL administration"])
    assert len(matches) == 1
    assert unmatched == []


def test_unmatched_skill_falls_through_for_llm_judgment():
    reqs = [SkillRequirement(skill="AWS", importance=Importance.REQUIRED, evidence="needs AWS")]
    matches, unmatched = exact_match_skills(reqs, ["Python", "FastAPI"])
    assert matches == []
    assert len(unmatched) == 1
    assert unmatched[0].skill == "AWS"


def test_skill_match_score_weights_required_over_preferred():
    from app.schemas.analysis import MatchType, SkillMatch

    matches = [
        SkillMatch(
            required_skill="Python",
            importance="required",
            match_type=MatchType.EXACT,
            matched_resume_skill="Python",
            similarity=1.0,
        ),
        SkillMatch(
            required_skill="AWS",
            importance="required",
            match_type=MatchType.NONE,
            matched_resume_skill=None,
            similarity=0.0,
        ),
        SkillMatch(
            required_skill="Docker",
            importance="preferred",
            match_type=MatchType.NONE,
            matched_resume_skill=None,
            similarity=0.0,
        ),
    ]
    score = skill_match_score(matches)
    # required weight 2.0 each, preferred weight 1.0: earned 2.0 / total 5.0 = 40%
    assert score == 40.0


def test_keyword_coverage_score():
    score = keyword_coverage_score("I build APIs with Python and FastAPI.", ["python", "docker"])
    assert score == 50.0


def test_compute_total_years_experience():
    resume = make_resume()
    years = compute_total_years_experience(resume)
    assert years == 2.0


def test_compute_total_years_experience_handles_current_role():
    resume = make_resume(
        experience=[
            ExperienceEntry(
                title="Engineer",
                company="Acme",
                start_date="2020-01-01",
                is_current=True,
                source_evidence="Engineer, Acme, 2020-present",
            )
        ]
    )
    years = compute_total_years_experience(resume)
    assert years is not None and years > 0


def test_compute_total_years_experience_none_when_no_dates():
    resume = make_resume(experience=[])
    assert compute_total_years_experience(resume) is None


def test_experience_match_score_no_requirement_is_full_score():
    from app.schemas.job import JobProfile

    job = JobProfile(min_years_experience=None)
    assert experience_match_score(job, None) == 100.0


def test_experience_match_score_partial():
    from app.schemas.job import JobProfile

    job = JobProfile(min_years_experience=4.0)
    assert experience_match_score(job, 2.0) == 50.0


def test_experience_match_score_missing_experience_data_scores_zero():
    from app.schemas.job import JobProfile

    job = JobProfile(min_years_experience=4.0)
    assert experience_match_score(job, None) == 0.0


def test_analyze_ats_flags_missing_contact_and_sections():
    resume = make_resume(personal_info=PersonalInfo(name="Jane"), skills=[])
    ats = analyze_ats(resume, "Just some text with experience and education mentioned.", keyword_coverage=0.0)
    issue_texts = " ".join(i.issue for i in ats.issues)
    assert "email" in issue_texts.lower()
    assert "skills" in issue_texts.lower()


def test_compute_match_breakdown_is_weighted_average():
    breakdown = compute_match_breakdown(
        skill_match=100,
        experience_match=100,
        responsibility_match=100,
        keyword_coverage=100,
        qualification_match=100,
        ats_compatibility=100,
    )
    assert breakdown.overall_score == 100.0
