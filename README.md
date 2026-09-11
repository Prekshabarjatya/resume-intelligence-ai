# Resume Intelligence Platform — Core Agent Pipeline

The core multi-agent pipeline slice of the Agentic Resume Intelligence &
Optimization Platform: upload a resume and a job description, validate both,
extract structured profiles, match skills/experience, and get an explainable
gap analysis and recommendations.

Out of scope for this slice (see the full PRD for the eventual system):
auth, the Next.js frontend, resume optimization/generation, fact validation
of generated content, Postgres/Redis/Docker, async job queue. This runs
synchronously via a CLI or a single FastAPI endpoint.

## Architecture

```
resume + JD (PDF/DOCX/text)
        │
        ▼
  ingestion (deterministic: PyMuPDF / python-docx)
        │
        ▼
  LangGraph pipeline (app/graph.py)
        │
  validate_resume ──(rejected)──► END
        │
  validate_jd ──(rejected)──► END
        │
  extract_resume / extract_jd
        │
  match_skills (deterministic exact-match, then an LLM synonym-judgment
                call for anything not matched exactly)
        │
  assess_qualitative (LLM: responsibility & qualification fit)
        │
  analyze_ats_and_score (deterministic: ATS heuristics + weighted score)
        │
  analyze_gaps (LLM, grounded in the computed matches)
        │
  recommend (LLM, grounded in gaps + scores)
        │
        ▼
   AnalysisResult
```

Every LLM call is a real Groq call (`app/agents/*.py`) with Pydantic
structured output — no free-text parsing. Skill exact-matching, keyword
coverage, ATS structural checks, experience-duration math, and the final
weighted score are all deterministic code (`app/tools/scoring.py`) — nothing
there is left to the LLM to "calculate," per the platform's core principle
that LLMs reason and code computes.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install -e ".[dev]" --python .venv/bin/python
cp .env.example .env
# then put your Groq API key in .env — get one at https://console.groq.com
```

## Run

CLI (no server needed):

```bash
.venv/bin/python cli.py sample_data/resume.docx "$(cat sample_data/jd.txt)"
# or, JD as a file:
.venv/bin/python cli.py sample_data/resume.docx --jd-file sample_data/jd.txt
```

API server:

```bash
.venv/bin/uvicorn app.main:app --reload
# POST /analysis (multipart/form-data): resume_file + (jd_text or jd_file)
# GET  /health
```

## Tests

```bash
.venv/bin/python -m pytest
```

23 tests, all offline — every LLM call in the graph tests is mocked, so no
`GROQ_API_KEY` or network access is needed to run the suite. `sample_data/`
has a sample resume/JD for a real end-to-end run once you've set your key.

## Project layout

```
app/
  schemas/       Pydantic models: Document, ResumeProfile, JobProfile, AnalysisResult, AgentState
  ingestion/      Deterministic PDF/DOCX/text extraction
  tools/          Deterministic scoring: skill exact-match, keyword coverage,
                  ATS heuristics, experience-duration math, weighted score
  agents/         Real Groq LLM calls: validation, extraction, skill-synonym
                  judgment, qualitative matching, gap analysis, recommendations
  graph.py        LangGraph orchestration wiring it all together
  pipeline.py     Shared entry point used by both the CLI and the API
  main.py         FastAPI app
cli.py            Local runner
tests/            Unit tests (deterministic tools) + fully-mocked graph tests
sample_data/      Example resume.docx + jd.txt for a manual end-to-end run
```

## Notable design decisions

- **Two-tier LLM usage** (`GROQ_CLASSIFIER_MODEL` / `GROQ_REASONING_MODEL`):
  cheap model for classification/skill-synonym judgment, stronger model for
  extraction and reasoning-heavy steps.
- **Skill matching is exact-match-first, LLM-second.** A generic
  sentence-embedding model was tried for the semantic tier and produced
  unreliable similarity scores on short skill phrases (a true synonym pair
  scored *lower* than an unrelated pair in testing). An LLM judgment call,
  grounded strictly in the candidate's own listed skills so it can only
  point at a real skill or say "no match," replaced it — see
  `app/agents/skill_matching.py`.
- **Prompt injection**: every agent prompt that sees raw uploaded document
  text explicitly instructs the model to treat that text as data, not
  instructions.
- **Hallucination guardrails**: extraction requires verbatim source evidence
  for experience/project entries; gap analysis is explicitly instructed
  never to suggest fabricating experience, only to flag genuine gaps.
