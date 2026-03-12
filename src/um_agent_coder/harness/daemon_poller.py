"""Bridge between UMClaw daemon (Cloud Run) and local harness.

Polls the daemon for pending review tasks, generates roadmaps,
and executes them via the local harness.

Usage:
    from .daemon_poller import DaemonPoller, add_poll_args

    poller = DaemonPoller("https://um-agent-daemon-23o5bq3bfq-uc.a.run.app")
    poller.run_poll_loop(cli="codex")
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HARNESS_STOP_FILE = Path(".harness/stop")


class DaemonPoller:
    """Polls the UMClaw daemon for pending tasks and runs them locally."""

    def __init__(
        self,
        daemon_url: str,
        poll_interval: int = 60,
        api_key: str = "",
    ) -> None:
        self.daemon_url = daemon_url.rstrip("/")
        self.poll_interval = poll_interval
        self.api_key = api_key

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.daemon_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            logger.error("HTTP %s %s → %s %s", method, url, exc.code, exc.reason)
            raise
        except urllib.error.URLError as exc:
            logger.error("Connection error %s %s → %s", method, url, exc.reason)
            raise

    # ------------------------------------------------------------------
    # Daemon API methods
    # ------------------------------------------------------------------

    def poll_for_reviews(self) -> list[dict[str, Any]]:
        """GET /api/world-agent/tasks/pending — list pending tasks."""
        try:
            result = self._request("GET", "/api/world-agent/tasks/pending")
            tasks: list[dict[str, Any]] = result.get("tasks", [])
            logger.info("Polled daemon: %d pending task(s)", len(tasks))
            return tasks
        except Exception:
            logger.warning("Failed to poll daemon for pending tasks")
            return []

    def fetch_review(self, repo: str, goal_id: str) -> dict[str, Any]:
        """POST /api/world-agent/repos/{owner}/{repo}/review."""
        if "/" not in repo:
            raise ValueError(f"repo must be 'owner/repo', got: {repo!r}")
        path = f"/api/world-agent/repos/{repo}/review"
        return self._request("POST", path, {"goal_id": goal_id})

    def report_completion(
        self,
        task_id: str,
        success: bool,
        output: str = "",
        pr_url: str = "",
    ) -> None:
        """POST /api/world-agent/tasks/{task_id}/complete."""
        body: dict[str, Any] = {"success": success, "output": output}
        if pr_url:
            body["pr_url"] = pr_url
        self._request("POST", f"/api/world-agent/tasks/{task_id}/complete", body)
        logger.info("Reported task %s as %s", task_id, "success" if success else "failure")

    # ------------------------------------------------------------------
    # Roadmap generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_roadmap(
        task: dict[str, Any],
        review: dict[str, Any],
        roadmap_dir: str,
    ) -> Path:
        """Generate a harness-compatible roadmap from a daemon review."""
        from um_agent_coder.daemon.routes.world_agent._roadmap_gen import (
            generate_roadmap,
            write_roadmap,
        )

        rd = Path(roadmap_dir)
        rd.mkdir(parents=True, exist_ok=True)

        task_id = task.get("id", "unknown")
        goal_id = task.get("goal_id", task_id)
        goal_name = task.get("title", goal_id)
        repo_path = task.get("repo_path", ".")

        content = generate_roadmap(
            review_result=review,
            goal_id=goal_id,
            goal_name=goal_name,
            repo_path=repo_path,
        )
        output_path = str(rd / f"roadmap-{task_id}.md")
        write_roadmap(content, output_path)
        logger.info("Generated roadmap: %s", output_path)
        return Path(output_path)

    # ------------------------------------------------------------------
    # Main poll loop
    # ------------------------------------------------------------------

    def run_poll_loop(
        self,
        roadmap_dir: str = ".harness/roadmaps",
        cli: str = "codex",
    ) -> None:
        """Poll daemon, generate roadmaps, and run harness until stopped."""
        # Lazy import to avoid circular deps at module level
        from .main import Harness

        logger.info(
            "Starting poll loop: daemon=%s interval=%ds cli=%s",
            self.daemon_url,
            self.poll_interval,
            cli,
        )

        while True:
            if HARNESS_STOP_FILE.exists():
                logger.info("Stop file detected — exiting poll loop")
                break

            tasks = self.poll_for_reviews()

            for task in tasks:
                task_id = task.get("id", "unknown")
                repo = task.get("repo", "")
                goal_id = task.get("goal_id", "")

                logger.info("Processing task %s (repo=%s, goal=%s)", task_id, repo, goal_id)

                try:
                    review = self.fetch_review(repo, goal_id) if repo and goal_id else {}
                    roadmap_path = self._generate_roadmap(task, review, roadmap_dir)

                    harness = Harness(
                        roadmap_path=str(roadmap_path),
                        cli=cli,
                        dry_run=False,
                    )
                    harness.run()

                    self.report_completion(task_id, success=True, output="Harness run completed")
                    logger.info("Task %s completed successfully", task_id)
                except Exception:
                    logger.exception("Task %s failed", task_id)
                    self.report_completion(task_id, success=False, output="Harness run failed")

                if HARNESS_STOP_FILE.exists():
                    logger.info("Stop file detected — exiting poll loop")
                    return

            logger.debug("Sleeping %ds before next poll", self.poll_interval)
            time.sleep(self.poll_interval)


# ------------------------------------------------------------------
# CLI integration
# ------------------------------------------------------------------


def add_poll_args(parser: argparse.ArgumentParser) -> None:
    """Add --poll-daemon args to the harness argument parser."""
    parser.add_argument(
        "--poll-daemon",
        type=str,
        default="",
        help="Daemon URL to poll for tasks",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Poll interval in seconds",
    )
    parser.add_argument(
        "--daemon-api-key",
        type=str,
        default="",
        help="API key for daemon",
    )
