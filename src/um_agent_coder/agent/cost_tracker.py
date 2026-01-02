import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class TaskMetric:
    task_id: str
    prompt: str
    start_time: float
    end_time: Optional[float]
    tokens_used: int
    cost: float
    success: bool
    error: Optional[str]
    steps_completed: int
    total_steps: int


class CostTracker:
    """Tracks costs and performance metrics for agent tasks."""

    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.tasks: list[TaskMetric] = []
        self.current_task: Optional[TaskMetric] = None
        self.session_start = time.time()

    def start_task(self, task_id: str, prompt: str, total_steps: int):
        """Start tracking a new task."""
        self.current_task = TaskMetric(
            task_id=task_id,
            prompt=prompt[:100],  # Truncate for storage
            start_time=time.time(),
            end_time=None,
            tokens_used=0,
            cost=0.0,
            success=False,
            error=None,
            steps_completed=0,
            total_steps=total_steps,
        )

    def track_step(self, tokens: int, cost: float):
        """Track progress of current task."""
        if self.current_task:
            self.current_task.tokens_used += tokens
            self.current_task.cost += cost
            self.current_task.steps_completed += 1

        self.total_tokens += tokens
        self.total_cost += cost

    def complete_task(self, success: bool = True, error: Optional[str] = None):
        """Mark current task as complete."""
        if self.current_task:
            self.current_task.end_time = time.time()
            self.current_task.success = success
            self.current_task.error = error
            self.tasks.append(self.current_task)
            self.current_task = None

    def calculate_effectiveness(self) -> float:
        """
        Calculate cost-effectiveness score.

        Formula:
        Effectiveness = (Success Rate * Completion Rate) / (Avg Cost * Avg Time)
        """
        if not self.tasks:
            return 0.0

        successful_tasks = [t for t in self.tasks if t.success]
        success_rate = len(successful_tasks) / len(self.tasks)

        # Average completion rate
        completion_rates = [
            t.steps_completed / t.total_steps for t in self.tasks if t.total_steps > 0
        ]
        avg_completion = sum(completion_rates) / len(completion_rates) if completion_rates else 0

        # Average cost per task
        avg_cost = self.total_cost / len(self.tasks) if self.tasks else 1.0

        # Average time per task (in minutes)
        task_times = [(t.end_time - t.start_time) / 60 for t in self.tasks if t.end_time]
        avg_time = sum(task_times) / len(task_times) if task_times else 1.0

        # Calculate effectiveness
        effectiveness = (success_rate * avg_completion) / (avg_cost * avg_time + 0.001)
        return min(100.0, effectiveness * 100)  # Scale to 0-100

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        if not self.tasks:
            return {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
                "effectiveness_score": 0.0,
                "session_duration_minutes": 0.0,
            }

        successful_tasks = [t for t in self.tasks if t.success]
        failed_tasks = [t for t in self.tasks if not t.success]

        # Calculate task statistics
        task_times = [(t.end_time - t.start_time) for t in self.tasks if t.end_time]

        return {
            "total_tasks": len(self.tasks),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": len(successful_tasks) / len(self.tasks) * 100,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "avg_tokens_per_task": self.total_tokens // len(self.tasks),
            "avg_cost_per_task": round(self.total_cost / len(self.tasks), 4),
            "avg_time_per_task_seconds": sum(task_times) / len(task_times) if task_times else 0,
            "effectiveness_score": round(self.calculate_effectiveness(), 2),
            "session_duration_minutes": (time.time() - self.session_start) / 60,
            "cost_per_minute": self.total_cost / ((time.time() - self.session_start) / 60),
        }

    def get_task_history(self) -> list[dict[str, Any]]:
        """Get detailed task history."""
        history = []
        for task in self.tasks:
            duration = (task.end_time - task.start_time) if task.end_time else 0
            history.append(
                {
                    "task_id": task.task_id,
                    "prompt": task.prompt,
                    "timestamp": datetime.fromtimestamp(task.start_time).isoformat(),
                    "duration_seconds": round(duration, 2),
                    "tokens": task.tokens_used,
                    "cost": round(task.cost, 4),
                    "success": task.success,
                    "completion_rate": (
                        task.steps_completed / task.total_steps * 100 if task.total_steps > 0 else 0
                    ),
                    "error": task.error,
                }
            )
        return history

    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        import json

        metrics = {
            "statistics": self.get_statistics(),
            "task_history": self.get_task_history(),
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)

    def estimate_remaining_budget(self, budget: float) -> dict[str, Any]:
        """Estimate how many more tasks can be completed within budget."""
        if not self.tasks:
            return {"remaining_budget": budget, "estimated_tasks": 0, "estimated_tokens": 0}

        avg_cost_per_task = self.total_cost / len(self.tasks)
        self.total_tokens / len(self.tasks)

        remaining = budget - self.total_cost
        estimated_tasks = int(remaining / avg_cost_per_task) if avg_cost_per_task > 0 else 0
        estimated_tokens = (
            int(remaining / self.total_cost * self.total_tokens) if self.total_cost > 0 else 0
        )

        return {
            "remaining_budget": round(remaining, 4),
            "estimated_tasks": estimated_tasks,
            "estimated_tokens": estimated_tokens,
            "current_burn_rate": round(avg_cost_per_task, 4),
        }

    def export_state(self) -> dict[str, Any]:
        """
        Export cost tracker state for checkpointing.

        Returns:
            Serializable dictionary of tracker state
        """
        current_task_data = None
        if self.current_task:
            current_task_data = {
                "task_id": self.current_task.task_id,
                "prompt": self.current_task.prompt,
                "start_time": self.current_task.start_time,
                "end_time": self.current_task.end_time,
                "tokens_used": self.current_task.tokens_used,
                "cost": self.current_task.cost,
                "success": self.current_task.success,
                "error": self.current_task.error,
                "steps_completed": self.current_task.steps_completed,
                "total_steps": self.current_task.total_steps,
            }

        tasks_data = []
        for task in self.tasks:
            tasks_data.append(
                {
                    "task_id": task.task_id,
                    "prompt": task.prompt,
                    "start_time": task.start_time,
                    "end_time": task.end_time,
                    "tokens_used": task.tokens_used,
                    "cost": task.cost,
                    "success": task.success,
                    "error": task.error,
                    "steps_completed": task.steps_completed,
                    "total_steps": task.total_steps,
                }
            )

        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "session_start": self.session_start,
            "current_task": current_task_data,
            "tasks": tasks_data,
        }

    def import_state(self, state: dict[str, Any]):
        """
        Import cost tracker state from checkpoint.

        Args:
            state: State dictionary from export_state()
        """
        self.total_tokens = state.get("total_tokens", 0)
        self.total_cost = state.get("total_cost", 0.0)
        self.session_start = state.get("session_start", time.time())

        # Restore current task if any
        if state.get("current_task"):
            ct = state["current_task"]
            self.current_task = TaskMetric(
                task_id=ct["task_id"],
                prompt=ct["prompt"],
                start_time=ct["start_time"],
                end_time=ct["end_time"],
                tokens_used=ct["tokens_used"],
                cost=ct["cost"],
                success=ct["success"],
                error=ct["error"],
                steps_completed=ct["steps_completed"],
                total_steps=ct["total_steps"],
            )
        else:
            self.current_task = None

        # Restore completed tasks
        self.tasks = []
        for task_data in state.get("tasks", []):
            self.tasks.append(
                TaskMetric(
                    task_id=task_data["task_id"],
                    prompt=task_data["prompt"],
                    start_time=task_data["start_time"],
                    end_time=task_data["end_time"],
                    tokens_used=task_data["tokens_used"],
                    cost=task_data["cost"],
                    success=task_data["success"],
                    error=task_data["error"],
                    steps_completed=task_data["steps_completed"],
                    total_steps=task_data["total_steps"],
                )
            )
