import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobState:
    job_id: str
    prompt: str
    status: str
    created_at: float
    updated_at: float
    current_step_index: int
    plan: Optional[dict[str, Any]] = None  # Serialized ExecutionPlan
    analysis: Optional[dict[str, Any]] = None  # Serialized TaskAnalysis
    context_file: str = ""  # Path to separate context file
    error: Optional[str] = None
    results: list[dict[str, Any]] = None

    def __post_init__(self):
        if self.results is None:
            self.results = []


class JobStore:
    def __init__(self, base_dir: str = ".um_agent_jobs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_job_path(self, job_id: str) -> Path:
        return self.base_dir / f"{job_id}.json"

    def _get_context_path(self, job_id: str) -> Path:
        return self.base_dir / f"{job_id}_context.json"

    def create_job(self, job_id: str, prompt: str) -> JobState:
        job = JobState(
            job_id=job_id,
            prompt=prompt,
            status=JobStatus.PENDING.value,
            created_at=time.time(),
            updated_at=time.time(),
            current_step_index=0,
            context_file=str(self._get_context_path(job_id)),
        )
        self.save_job(job)
        return job

    def save_job(self, job: JobState):
        job.updated_at = time.time()
        with open(self._get_job_path(job.job_id), "w") as f:
            json.dump(asdict(job), f, indent=2)

    def load_job(self, job_id: str) -> Optional[JobState]:
        path = self._get_job_path(job_id)
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)
            return JobState(**data)

    def list_jobs(self) -> list[JobState]:
        jobs = []
        for file in self.base_dir.glob("*.json"):
            if not file.name.endswith("_context.json"):
                try:
                    with open(file) as f:
                        data = json.load(f)
                        jobs.append(JobState(**data))
                except Exception:
                    continue
        return sorted(jobs, key=lambda x: x.updated_at, reverse=True)

    def save_context(self, job_id: str, context_data: dict[str, Any]):
        with open(self._get_context_path(job_id), "w") as f:
            json.dump(context_data, f, indent=2)

    def load_context(self, job_id: str) -> dict[str, Any]:
        path = self._get_context_path(job_id)
        if not path.exists():
            return {"items": []}
        with open(path) as f:
            return json.load(f)
