#!/usr/bin/env python3
"""Local runner, no server needed:

    python cli.py sample_data/resume.pdf sample_data/jd.txt
    python cli.py sample_data/resume.pdf --jd-file sample_data/jd.docx
"""

import argparse
import json
import sys

from app.pipeline import run_pipeline
from app.schemas.analysis import PipelineStatus


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the resume/JD analysis pipeline locally.")
    parser.add_argument("resume", help="Path to a resume PDF or DOCX file.")
    jd_group = parser.add_mutually_exclusive_group(required=True)
    jd_group.add_argument("jd_text", nargs="?", help="Job description as a plain text argument.")
    jd_group.add_argument("--jd-file", help="Path to a job description PDF or DOCX file.")
    args = parser.parse_args()

    if args.jd_file:
        outcome = run_pipeline(args.resume, args.jd_file, jd_is_file=True)
    else:
        outcome = run_pipeline(args.resume, args.jd_text, jd_is_file=False)

    if outcome.status in (PipelineStatus.REJECTED_RESUME, PipelineStatus.REJECTED_JD):
        print(f"REJECTED: {outcome.rejection_reason}", file=sys.stderr)
        return 1

    print(json.dumps(outcome.result.model_dump(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
