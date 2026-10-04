"""Per-run metrics returned alongside every analysis."""

import time
from dataclasses import dataclass, field


@dataclass
class RunMetrics:
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    gaps_identified: int = 0
    skills_matched: int = 0

    @property
    def runtime_seconds(self) -> float:
        return (self.end_time or time.time()) - self.start_time

    def finish(self) -> None:
        self.end_time = time.time()

    def to_dict(self) -> dict:
        return {
            "runtime_seconds": round(self.runtime_seconds, 2),
            "gaps_identified": self.gaps_identified,
            "skills_matched": self.skills_matched,
        }
