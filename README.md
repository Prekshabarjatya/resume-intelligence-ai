# Resume Intelligence Platform — Core Agent Pipeline

The core multi-agent pipeline slice of the Agentic Resume Intelligence &
Optimization Platform: upload a resume and a job description, validate both,
extract structured profiles, match skills/experience, and get an explainable
gap analysis and recommendations.

Includes a single-page UI (`app/static/index.html`, served by the FastAPI
app itself) that shows live per-step pipeline progress and the reasoning
behind every score — see "UI" below.

Out of scope for this slice (see the full PRD for the eventual system):
auth, the Next.js frontend, resume optimization/generation, fact validation
of generated content, Postgres/Redis/Docker, a real message queue. Jobs run
in a background thread with in-memory status tracking (`app/jobs.py`) — good
enough for one local dev process, not a substitute for the PRD's eventual
Redis-backed worker.

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

UI (recommended — shows live pipeline progress and reasoning):

```bash
.venv/bin/uvicorn app.main:app --reload
# open http://localhost:8000
```

CLI (no server needed):

```bash
.venv/bin/python cli.py sample_data/resume.docx "$(cat sample_data/jd.txt)"
# or, JD as a file:
.venv/bin/python cli.py sample_data/resume.docx --jd-file sample_data/jd.txt
```

API directly:

```bash
# Synchronous — blocks until done, returns the full result:
# POST /analysis (multipart/form-data): resume_file + (jd_text or jd_file)

# Async — what the UI uses, returns immediately with a job_id to poll:
# POST /analysis/start  -> {"job_id": "..."}
# GET  /analysis/{job_id} -> {"status": "running"|"completed"|"rejected"|"failed", "steps": [...], "result"?: ...}

# GET  /health
```

## UI

`app/static/index.html`, mounted as static files by `app/main.py` — no
separate frontend project or build step. Upload a resume + JD, then:

- **Pipeline Progress** updates live from the actual LangGraph execution
  (`app/jobs.py` polls `graph.stream(..., stream_mode="updates")` under the
  hood) — it shows which node just finished, not a fake timer.
- Every score comes with its reasoning: document classification reasons,
  the qualitative-match agent's explanation for the responsibility/
  qualification bars, and a "Why" line under any skill match that wasn't a
  trivial exact match (semantic match or no-match), taken from the actual
  LLM judgment rather than summarized after the fact.

## Tests

```bash
.venv/bin/python -m pytest
```

27 tests, all offline — every LLM call is mocked, so no `GROQ_API_KEY` or
network access is needed to run the suite. `sample_data/` has a sample
resume/JD for a real end-to-end run once you've set your key.

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
  pipeline.py     Shared entry point used by the CLI, sync API, and job runner
  jobs.py         In-memory async job tracking (real per-step progress for the UI)
  main.py         FastAPI app (serves the UI + both sync and async endpoints)
  static/         Single-page UI (plain HTML/CSS/JS, no build step)
cli.py            Local runner
tests/            Unit tests (deterministic tools) + fully-mocked graph/job tests
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
