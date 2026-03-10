"""Background task processor wrapping MultiModelOrchestrator and EnhancedAgent."""

from __future__ import annotations

import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, Optional, Set

if TYPE_CHECKING:
    from um_agent_coder.daemon.config import DaemonSettings
    from um_agent_coder.daemon.database import Database

logger = logging.getLogger(__name__)


class TaskWorker:
    """Processes tasks in background threads using the existing orchestrators."""

    def __init__(self, settings: DaemonSettings, db: Database):
        self.settings = settings
        self.db = db
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_tasks,
            thread_name_prefix="task-worker",
        )
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._running_tasks: Set[str] = set()
        self._cancelled: Set[str] = set()
        self._consumer_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._consumer_task = asyncio.create_task(self._consume_loop())
        logger.info("TaskWorker started")

    async def stop(self):
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)
        logger.info("TaskWorker stopped")

    async def enqueue(self, task_id: str):
        await self._queue.put(task_id)
        logger.info("Task %s enqueued (queue size: %d)", task_id, self._queue.qsize())

    async def cancel(self, task_id: str):
        self._cancelled.add(task_id)
        logger.info("Task %s marked for cancellation", task_id)

    async def _consume_loop(self):
        """Main consumer loop - pulls tasks from queue and executes them."""
        while True:
            task_id = await self._queue.get()

            if task_id in self._cancelled:
                self._cancelled.discard(task_id)
                self._queue.task_done()
                continue

            # Wait if at capacity
            while len(self._running_tasks) >= self.settings.max_concurrent_tasks:
                await asyncio.sleep(0.5)

            self._running_tasks.add(task_id)
            asyncio.create_task(self._run_task(task_id))
            self._queue.task_done()

    async def _run_task(self, task_id: str):
        """Execute a single task via run_in_executor."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            await self.db.update_task(task_id, status="running", started_at=now)
            await self.db.add_log(task_id, "Task execution started")

            task = await self.db.get_task(task_id)
            if not task:
                logger.error("Task %s not found in DB", task_id)
                return

            prompt = task["prompt"]
            spec_dict = task.get("spec")

            # Run the blocking orchestrator in a thread
            result = await self._loop.run_in_executor(
                self._executor,
                self._execute_sync,
                task_id,
                prompt,
                spec_dict,
            )

            if task_id in self._cancelled:
                self._cancelled.discard(task_id)
                return

            completed_at = datetime.now(timezone.utc).isoformat()
            if result.get("success"):
                await self.db.update_task(
                    task_id,
                    status="completed",
                    result=result,
                    completed_at=completed_at,
                )
                await self.db.add_log(task_id, "Task completed successfully")
            else:
                await self.db.update_task(
                    task_id,
                    status="failed",
                    error=result.get("error", "Unknown error"),
                    result=result,
                    completed_at=completed_at,
                )
                await self.db.add_log(task_id, f"Task failed: {result.get('error')}", level="error")

            # Send notifications
            self._send_notifications(task_id, task, result)

        except Exception as e:
            logger.exception("Unhandled error processing task %s", task_id)
            completed_at = datetime.now(timezone.utc).isoformat()
            await self.db.update_task(
                task_id,
                status="failed",
                error=str(e),
                completed_at=completed_at,
            )
            await self.db.add_log(
                task_id,
                f"Unhandled error: {e}\n{traceback.format_exc()}",
                level="error",
            )
        finally:
            self._running_tasks.discard(task_id)

    def _execute_sync(
        self,
        task_id: str,
        prompt: str,
        spec_dict: Optional[Dict],
    ) -> Dict:
        """Synchronous execution - runs in a thread pool."""
        try:
            # If a TaskSpec is provided, use orchestrator with full spec
            if spec_dict:
                return self._run_with_spec(task_id, spec_dict)

            # Try MultiModelOrchestrator first, fall back to EnhancedAgent
            return self._run_orchestrator(task_id, prompt)

        except Exception as e:
            logger.exception("Sync execution error for task %s", task_id)
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _run_with_spec(self, task_id: str, spec_dict: Dict) -> Dict:
        """Run task with a full TaskSpec via the orchestrator."""
        from um_agent_coder.orchestrator.task_spec import TaskSpec

        spec = TaskSpec.from_dict(spec_dict.copy())
        spec.task_id = task_id

        # Convert spec to prompt and run through orchestrator
        prompt = spec.to_prompt()
        return self._run_orchestrator(task_id, prompt)

    def _run_orchestrator(self, task_id: str, prompt: str) -> Dict:
        """Run through MultiModelOrchestrator with Gemini OAuth LLM."""
        try:
            gemini_llm = self._create_gemini_llm()

            from um_agent_coder.orchestrator.multi_model import MultiModelOrchestrator

            # Use Gemini for all model roles (single-model orchestration)
            orchestrator = MultiModelOrchestrator(
                gemini=gemini_llm,
                codex=gemini_llm,
                claude=gemini_llm,
                checkpoint_dir=self.settings.checkpoint_dir,
                verbose=self.settings.verbose,
            )

            result = orchestrator.run(prompt=prompt, task_id=task_id)
            return result

        except Exception as e:
            logger.warning(
                "MultiModelOrchestrator failed for %s, falling back to direct Gemini: %s",
                task_id,
                e,
            )
            return self._run_direct_gemini(task_id, prompt)

    def _create_gemini_llm(self):
        """Create an LLM adapter that calls Gemini via OAuth."""
        from um_agent_coder.daemon.gemini_client import create_gemini_client
        from um_agent_coder.llm.base import LLM

        class GeminiOAuthLLM(LLM):
            def __init__(self):
                self._model = create_gemini_client(model="gemini-2.5-pro")

            def chat(self, prompt: str) -> str:
                response = self._model.generate_content(prompt)
                return response.text

        return GeminiOAuthLLM()

    def _run_direct_gemini(self, task_id: str, prompt: str) -> Dict:
        """Fallback: run prompt directly through Gemini API."""
        try:
            from um_agent_coder.daemon.gemini_client import gemini_chat

            response = gemini_chat(prompt)
            return {
                "success": True,
                "task_id": task_id,
                "output": response,
                "model": "gemini-2.5-pro",
            }
        except Exception as e:
            return {"success": False, "error": f"Direct Gemini also failed: {e}"}

    def _send_notifications(self, task_id: str, task: Dict, result: Dict):
        """Send completion/failure notifications via WebhookNotifier."""
        try:
            source_meta = task.get("source_meta") or {}
            webhook_url = source_meta.get("webhook_url")
            if not webhook_url and not self.settings.default_webhook_url:
                return

            from um_agent_coder.orchestrator.task_spec import (
                TaskUpdate,
                UpdateType,
                WebhookNotifier,
            )

            notifier = WebhookNotifier(
                webhook_url=webhook_url or self.settings.default_webhook_url,
                slack_webhook=self.settings.default_slack_webhook,
                discord_webhook=self.settings.default_discord_webhook,
            )

            update_type = UpdateType.COMPLETED if result.get("success") else UpdateType.ERROR
            update = TaskUpdate(
                task_id=task_id,
                update_type=update_type,
                message=f"Task {task_id} {'completed' if result.get('success') else 'failed'}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                data={"prompt": task.get("prompt", "")[:200]},
            )
            notifier.notify(update)
        except Exception:
            logger.exception("Failed to send notification for task %s", task_id)
