"""Event collectors for the World Agent.

Pluggable collectors that fetch external data. Each collector implements
the EventCollector ABC.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

from um_agent_coder.daemon.routes.world_agent.models import (
    Event,
    EventCategory,
    EventSeverity,
)

logger = logging.getLogger(__name__)


class EventCollector(ABC):
    """Base class for all event collectors."""

    @abstractmethod
    async def collect(self, since: Optional[datetime] = None) -> List[Event]:
        """Fetch new events since last collection."""
        ...

    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this source."""
        ...


# GitHub event type → severity mapping
_GITHUB_SEVERITY: Dict[str, EventSeverity] = {
    "PushEvent": EventSeverity.info,
    "CreateEvent": EventSeverity.info,
    "DeleteEvent": EventSeverity.notable,
    "PullRequestEvent": EventSeverity.notable,
    "PullRequestReviewEvent": EventSeverity.info,
    "IssuesEvent": EventSeverity.notable,
    "IssueCommentEvent": EventSeverity.info,
    "ReleaseEvent": EventSeverity.urgent,
    "ForkEvent": EventSeverity.info,
    "WatchEvent": EventSeverity.info,
    "MemberEvent": EventSeverity.notable,
    "PublicEvent": EventSeverity.notable,
}


def _github_event_title(gh_event: dict) -> str:
    """Build a human-readable title from a GitHub event."""
    event_type = gh_event.get("type", "Unknown")
    repo_name = gh_event.get("repo", {}).get("name", "unknown")
    actor = gh_event.get("actor", {}).get("login", "unknown")
    payload = gh_event.get("payload", {})

    if event_type == "PushEvent":
        count = payload.get("size", 0)
        return f"{actor} pushed {count} commit(s) to {repo_name}"
    elif event_type == "PullRequestEvent":
        action = payload.get("action", "")
        pr_title = payload.get("pull_request", {}).get("title", "")
        return f"{actor} {action} PR '{pr_title}' on {repo_name}"
    elif event_type == "IssuesEvent":
        action = payload.get("action", "")
        issue_title = payload.get("issue", {}).get("title", "")
        return f"{actor} {action} issue '{issue_title}' on {repo_name}"
    elif event_type == "ReleaseEvent":
        tag = payload.get("release", {}).get("tag_name", "")
        return f"{actor} released {tag} on {repo_name}"
    elif event_type == "CreateEvent":
        ref_type = payload.get("ref_type", "")
        ref = payload.get("ref", "")
        return f"{actor} created {ref_type} '{ref}' on {repo_name}"
    elif event_type == "DeleteEvent":
        ref_type = payload.get("ref_type", "")
        ref = payload.get("ref", "")
        return f"{actor} deleted {ref_type} '{ref}' on {repo_name}"
    else:
        return f"{actor}: {event_type} on {repo_name}"


class GitHubEventsCollector(EventCollector):
    """Collects events from GitHub Events API with ETag caching."""

    def __init__(self, repos: List[str], token: Optional[str] = None):
        """
        Args:
            repos: List of "owner/repo" strings.
            token: GitHub personal access token (optional, raises rate limit).
        """
        self._repos = repos
        self._token = token
        self._etags: Dict[str, str] = {}  # repo → ETag

    def source_id(self) -> str:
        return "dev.github_events"

    async def collect(self, since: Optional[datetime] = None) -> List[Event]:
        """Fetch events from all configured repos."""
        events: List[Event] = []
        headers: Dict[str, str] = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            for repo in self._repos:
                try:
                    repo_events = await self._collect_repo(client, repo, headers, since)
                    events.extend(repo_events)
                except Exception as e:
                    logger.warning("Failed to collect GitHub events for %s: %s", repo, e)

        logger.info("Collected %d GitHub events from %d repos", len(events), len(self._repos))
        return events

    async def _collect_repo(
        self,
        client: httpx.AsyncClient,
        repo: str,
        headers: Dict[str, str],
        since: Optional[datetime],
    ) -> List[Event]:
        """Fetch events for a single repo with ETag caching."""
        url = f"https://api.github.com/repos/{repo}/events"
        req_headers = {**headers}

        # Use ETag for conditional requests
        etag = self._etags.get(repo)
        if etag:
            req_headers["If-None-Match"] = etag

        resp = await client.get(url, headers=req_headers, params={"per_page": 30})

        if resp.status_code == 304:
            logger.debug("No new events for %s (ETag match)", repo)
            return []

        if resp.status_code != 200:
            logger.warning("GitHub API returned %d for %s", resp.status_code, repo)
            return []

        # Cache new ETag
        new_etag = resp.headers.get("ETag")
        if new_etag:
            self._etags[repo] = new_etag

        gh_events = resp.json()
        events: List[Event] = []

        for gh_event in gh_events:
            # Parse timestamp
            created_str = gh_event.get("created_at", "")
            try:
                ts = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = datetime.now(timezone.utc)

            # Skip events older than 'since'
            if since and ts < since:
                continue

            event_type = gh_event.get("type", "Unknown")
            severity = _GITHUB_SEVERITY.get(event_type, EventSeverity.info)

            events.append(
                Event(
                    id=f"gh-{gh_event.get('id', uuid.uuid4().hex[:12])}",
                    source=self.source_id(),
                    timestamp=ts,
                    category=EventCategory.dev,
                    severity=severity,
                    title=_github_event_title(gh_event),
                    body=str(gh_event.get("payload", {}))[:2000],
                    metadata={
                        "github_event_type": event_type,
                        "repo": repo,
                        "actor": gh_event.get("actor", {}).get("login", ""),
                    },
                )
            )

        return events
