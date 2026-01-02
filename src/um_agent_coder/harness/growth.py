"""
Growth loop for continuous improvement.

After completing all roadmap tasks, enters growth mode to:
1. Analyze what was built
2. Identify improvement opportunities
3. Generate new tasks for optimization/growth
4. Continue executing until stopped
"""

import logging
from typing import Optional

from .codex_executor import CodexExecutor
from .models import Roadmap, Task, TaskStatus

logger = logging.getLogger(__name__)


class GrowthLoop:
    """Generate improvement tasks after initial completion."""

    GROWTH_PROMPT_TEMPLATE = """
## Growth Mode Analysis

The following project has been completed:

**Project**: {project_name}
**Objective**: {objective}

### Completed Tasks:
{completed_tasks}

### Success Criteria Status:
{success_criteria}

### Growth Instructions:
{growth_instructions}

---

## Your Task

Analyze this completed project and generate ONE specific, actionable improvement task that would:
1. Increase value (revenue, users, engagement)
2. Improve reliability (fewer bugs, better error handling)
3. Enhance performance (faster, more efficient)
4. Add requested features or integrations

Respond in this exact format:

TASK_ID: growth-XXX (where XXX is a unique number)
DESCRIPTION: Clear, one-line description of what to do
TIMEOUT: Estimated minutes to complete (15-60)
SUCCESS_CRITERIA: How to verify the task is complete
RATIONALE: Why this improvement matters (1-2 sentences)
"""

    def __init__(self, executor: CodexExecutor):
        self.executor = executor
        self._growth_counter = 0

    def should_enter_growth_mode(self, roadmap: Roadmap) -> bool:
        """Check if all tasks are complete and we should enter growth mode."""
        return roadmap.is_complete

    def generate_growth_task(
        self,
        roadmap: Roadmap,
        completed_tasks: list[Task],
    ) -> Optional[Task]:
        """Generate a new improvement task based on completed work."""
        self._growth_counter += 1

        # Build context from completed work
        completed_summary = self._summarize_completed_tasks(completed_tasks)
        criteria_status = self._format_success_criteria(roadmap.success_criteria)
        growth_instructions = (
            "\n".join(f"- {inst}" for inst in roadmap.growth_instructions)
            or "- Improve performance\n- Add features\n- Optimize for growth"
        )

        prompt = self.GROWTH_PROMPT_TEMPLATE.format(
            project_name=roadmap.name,
            objective=roadmap.objective,
            completed_tasks=completed_summary,
            success_criteria=criteria_status,
            growth_instructions=growth_instructions,
        )

        # Ask Codex to generate a growth task
        result = self.executor.execute(
            Task(
                id=f"_growth_analysis_{self._growth_counter}",
                description="Generate growth task",
                phase="growth",
                timeout_minutes=5,
            ),
            context=prompt,
        )

        if not result.success:
            logger.warning(f"Growth task generation failed: {result.error}")
            return None

        # Parse the generated task
        return self._parse_growth_task(result.output)

    def _summarize_completed_tasks(self, tasks: list[Task]) -> str:
        """Create a summary of completed tasks."""
        if not tasks:
            return "No tasks completed yet."

        lines = []
        for task in tasks[-10:]:  # Last 10 tasks for context
            status = "COMPLETED" if task.status == TaskStatus.COMPLETED else "FAILED"
            lines.append(f"- [{status}] {task.id}: {task.description}")
            if task.output:
                # Include brief output summary
                output_preview = task.output[:200].replace("\n", " ")
                lines.append(f"  Output: {output_preview}...")

        return "\n".join(lines)

    def _format_success_criteria(self, criteria: list[str]) -> str:
        """Format success criteria as checklist."""
        if not criteria:
            return "No success criteria defined."

        return "\n".join(f"- [x] {c}" for c in criteria)

    def _parse_growth_task(self, output: str) -> Optional[Task]:
        """Parse Codex output into a Task object."""
        try:
            lines = output.strip().split("\n")

            task_data = {}
            for line in lines:
                line = line.strip()
                for key in [
                    "TASK_ID:",
                    "DESCRIPTION:",
                    "TIMEOUT:",
                    "SUCCESS_CRITERIA:",
                    "RATIONALE:",
                ]:
                    if line.upper().startswith(key):
                        value = line[len(key) :].strip()
                        task_data[key.replace(":", "").lower()] = value
                        break

            if "task_id" not in task_data or "description" not in task_data:
                logger.warning(f"Could not parse growth task from output: {output[:200]}")
                return None

            # Ensure unique task ID
            task_id = task_data["task_id"]
            if not task_id.startswith("growth-"):
                task_id = f"growth-{self._growth_counter:03d}"

            timeout = 30  # default
            if "timeout" in task_data:
                try:
                    timeout = int("".join(c for c in task_data["timeout"] if c.isdigit()))
                except ValueError:
                    pass

            return Task(
                id=task_id,
                description=task_data["description"],
                phase="Growth",
                timeout_minutes=timeout,
                success_criteria=task_data.get("success_criteria", ""),
                status=TaskStatus.PENDING,
            )

        except Exception as e:
            logger.exception(f"Failed to parse growth task: {e}")
            return None

    def validate_growth_task(self, task: Task, roadmap: Roadmap) -> bool:
        """Validate that a growth task is reasonable and not duplicate."""
        # Check for duplicate task IDs
        existing_ids = {t.id for t in roadmap.all_tasks}
        if task.id in existing_ids:
            logger.warning(f"Duplicate growth task ID: {task.id}")
            return False

        # Check for very similar descriptions
        existing_descriptions = {t.description.lower() for t in roadmap.all_tasks}
        if task.description.lower() in existing_descriptions:
            logger.warning(f"Duplicate growth task description: {task.description}")
            return False

        # Ensure reasonable timeout
        if task.timeout_minutes < 5 or task.timeout_minutes > 120:
            task.timeout_minutes = 30  # Reset to default

        return True


class GrowthStrategy:
    """Different growth strategies for various objectives."""

    @staticmethod
    def revenue_focused() -> list[str]:
        """Growth instructions focused on revenue."""
        return [
            "Add payment integration or upgrade existing",
            "Implement pricing tiers or upsells",
            "Add analytics to track conversion",
            "Optimize checkout flow",
            "Add referral or affiliate system",
        ]

    @staticmethod
    def user_growth_focused() -> list[str]:
        """Growth instructions focused on user acquisition."""
        return [
            "Improve SEO and meta tags",
            "Add social sharing features",
            "Implement viral mechanics",
            "Add onboarding improvements",
            "Create landing page optimizations",
        ]

    @staticmethod
    def reliability_focused() -> list[str]:
        """Growth instructions focused on reliability."""
        return [
            "Add comprehensive error handling",
            "Implement logging and monitoring",
            "Add automated tests",
            "Set up health checks",
            "Improve data validation",
        ]

    @staticmethod
    def performance_focused() -> list[str]:
        """Growth instructions focused on performance."""
        return [
            "Optimize database queries",
            "Add caching layer",
            "Implement lazy loading",
            "Optimize bundle size",
            "Add CDN for static assets",
        ]
