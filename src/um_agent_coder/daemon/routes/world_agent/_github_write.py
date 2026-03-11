"""GitHub REST API write client for the World Agent.

Provides async methods for creating branches, pushing files,
opening PRs, posting comments, and checking CI status.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

API = "https://api.github.com"


class GitHubWriteClient:
    """Async GitHub REST API write client."""

    def __init__(self, token: str):
        self._token = token
        self._headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(headers=self._headers, timeout=30.0)

    async def get_default_branch_sha(self, repo: str, branch: str = "main") -> str:
        """Get the HEAD SHA of a branch."""
        async with self._client() as c:
            r = await c.get(f"{API}/repos/{repo}/git/ref/heads/{branch}")
            r.raise_for_status()
            return r.json()["object"]["sha"]

    async def create_branch(self, repo: str, branch_name: str, from_sha: str) -> dict:
        """Create a new branch from a given SHA."""
        async with self._client() as c:
            r = await c.post(
                f"{API}/repos/{repo}/git/refs",
                json={"ref": f"refs/heads/{branch_name}", "sha": from_sha},
            )
            r.raise_for_status()
            return r.json()

    async def get_file(self, repo: str, path: str, branch: str = "main") -> Dict[str, Any]:
        """Get file content and metadata from a repo."""
        async with self._client() as c:
            r = await c.get(
                f"{API}/repos/{repo}/contents/{path}",
                params={"ref": branch},
            )
            r.raise_for_status()
            data = r.json()
            if data.get("encoding") == "base64" and data.get("content"):
                data["decoded_content"] = base64.b64decode(data["content"]).decode(
                    "utf-8", errors="replace"
                )
            return data

    async def create_or_update_file(
        self,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: str,
        sha: Optional[str] = None,
    ) -> dict:
        """Create or update a single file. If sha is provided, it's an update."""
        encoded = base64.b64encode(content.encode()).decode()
        body: Dict[str, Any] = {
            "message": message,
            "content": encoded,
            "branch": branch,
        }
        if sha:
            body["sha"] = sha
        async with self._client() as c:
            r = await c.put(f"{API}/repos/{repo}/contents/{path}", json=body)
            r.raise_for_status()
            return r.json()

    async def create_pull_request(
        self,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
    ) -> dict:
        """Open a pull request."""
        async with self._client() as c:
            r = await c.post(
                f"{API}/repos/{repo}/pulls",
                json={
                    "title": title,
                    "body": body,
                    "head": head,
                    "base": base,
                },
            )
            r.raise_for_status()
            return r.json()

    async def post_comment(self, repo: str, issue_number: int, body: str) -> dict:
        """Post a comment on an issue or PR."""
        async with self._client() as c:
            r = await c.post(
                f"{API}/repos/{repo}/issues/{issue_number}/comments",
                json={"body": body},
            )
            r.raise_for_status()
            return r.json()

    async def get_check_runs(self, repo: str, ref: str) -> List[dict]:
        """Get CI check runs for a git ref (SHA or branch)."""
        async with self._client() as c:
            r = await c.get(f"{API}/repos/{repo}/commits/{ref}/check-runs")
            r.raise_for_status()
            data = r.json()
            return [
                {
                    "name": cr["name"],
                    "status": cr["status"],
                    "conclusion": cr.get("conclusion"),
                    "html_url": cr.get("html_url"),
                    "started_at": cr.get("started_at"),
                    "completed_at": cr.get("completed_at"),
                }
                for cr in data.get("check_runs", [])
            ]

    async def push_files(
        self,
        repo: str,
        branch: str,
        files: List[Dict[str, str]],
    ) -> List[dict]:
        """Push multiple files to a branch. Each file dict has {path, content, message}."""
        results = []
        for f in files:
            # Try to get existing file SHA for update
            sha = None
            try:
                existing = await self.get_file(repo, f["path"], branch)
                sha = existing.get("sha")
            except httpx.HTTPStatusError:
                pass  # File doesn't exist yet — will create

            result = await self.create_or_update_file(
                repo=repo,
                path=f["path"],
                content=f["content"],
                message=f.get("message", f"Update {f['path']}"),
                branch=branch,
                sha=sha,
            )
            results.append(result)
        return results
