"""Instrumentation for run time, token usage, and accuracy metrics."""

import time
from dataclasses import dataclass, field


@dataclass
class RunMetrics:
    """Captures performance metrics for a single analysis run."""
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    tokens_used: int = 0
    gaps_identified: int = 0
    skills_matched: int = 0
    accuracy_score: float | None = None  # ragas evaluation or manual validation

    @property
    def runtime_seconds(self) -> float:
        """Total runtime in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def finish(self):
        """Mark the run as complete."""
        self.end_time = time.time()

    def to_dict(self) -> dict:
        """Serialize metrics for API response."""
        return {
            "runtime_seconds": round(self.runtime_seconds, 2),
            "tokens_used": self.tokens_used,
            "gaps_identified": self.gaps_identified,
            "skills_matched": self.skills_matched,
            "accuracy_score": self.accuracy_score,
        }
