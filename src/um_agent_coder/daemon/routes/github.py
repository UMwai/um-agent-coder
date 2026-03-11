"""GitHub webhook handler - processes issue comments with /agent command and PR events."""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request

from um_agent_coder.daemon.auth import verify_github_signature
from um_agent_coder.daemon.routes.world_agent._github_write import GitHubWriteClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def get_db():
    from um_agent_coder.daemon.app import get_db as _get

    return _get()


def get_worker():
    from um_agent_coder.daemon.app import get_worker as _get

    return _get()


def get_settings():
    from um_agent_coder.daemon.app import get_settings as _get

    return _get()


@router.post("/github")
async def github_webhook(request: Request):
    """Handle GitHub webhooks (issue_comment with /agent, pull_request events)."""
    settings = get_settings()
    body = await request.body()

    # Verify signature if secret is configured
    if settings.github_webhook_secret:
        sig = request.headers.get("X-Hub-Signature-256", "")
        if not verify_github_signature(body, sig, settings.github_webhook_secret):
            raise HTTPException(status_code=401, detail="Invalid signature")

    event = request.headers.get("X-GitHub-Event", "")
    payload = json.loads(body)

    if event == "ping":
        return {"status": "pong"}

    if event == "issue_comment":
        return await _handle_issue_comment(payload)

    if event == "pull_request":
        return await _handle_pull_request(payload)

    return {"status": "ignored", "event": event}


async def _handle_issue_comment(payload: dict):
    """Handle issue_comment events - looks for /agent command."""
    action = payload.get("action")
    if action != "created":
        return {"status": "ignored", "reason": "not a new comment"}

    comment = payload.get("comment", {})
    body = comment.get("body", "")

    # Check for /agent command
    if not body.strip().startswith("/agent"):
        return {"status": "ignored", "reason": "no /agent command"}

    # Extract the prompt (everything after /agent)
    prompt = body.strip().removeprefix("/agent").strip()
    if not prompt:
        return {"status": "ignored", "reason": "empty prompt"}

    issue = payload.get("issue", {})
    repo = payload.get("repository", {})

    # Build context-rich prompt
    full_prompt = (
        f"GitHub Issue #{issue.get('number')}: {issue.get('title', '')}\n"
        f"Repo: {repo.get('full_name', '')}\n"
        f"Request: {prompt}\n"
        f"\nIssue body:\n{issue.get('body', '')[:2000]}"
    )

    db = get_db()
    worker = get_worker()
    task_id = f"gh-{uuid.uuid4().hex[:12]}"

    source_meta = {
        "github_event": "issue_comment",
        "repo": repo.get("full_name"),
        "issue_number": issue.get("number"),
        "comment_id": comment.get("id"),
        "user": comment.get("user", {}).get("login"),
    }

    await db.create_task(
        task_id=task_id,
        prompt=full_prompt,
        source="github",
        source_meta=source_meta,
    )
    await db.add_log(
        task_id,
        f"Created from GitHub issue comment by {source_meta.get('user')}",
    )

    # Register completion callback to post result back as a GitHub comment
    async def _post_result_callback(tid: str, result: str):
        try:
            settings = get_settings()
            if not settings.github_token:
                logger.warning("No GitHub token; skipping comment feedback for %s", tid)
                return
            gh = GitHubWriteClient(token=settings.github_token)
            repo_name = source_meta.get("repo", "")
            issue_num = source_meta.get("issue_number")
            if repo_name and issue_num:
                body = f"**Agent result** (task `{tid}`):\n\n{result[:60000]}"
                await gh.post_comment(repo_name, issue_num, body)
                logger.info("Posted result for %s back to %s#%s", tid, repo_name, issue_num)
        except Exception as e:
            logger.error("Failed to post GitHub comment for %s: %s", tid, e)

    await worker.enqueue(task_id, on_complete=_post_result_callback)

    return {"status": "accepted", "task_id": task_id}


async def _handle_pull_request(payload: dict):
    """Handle pull_request events (opened, synchronize)."""
    action = payload.get("action")
    if action not in ("opened", "synchronize"):
        return {"status": "ignored", "reason": f"PR action '{action}' not handled"}

    pr = payload.get("pull_request", {})
    repo = payload.get("repository", {})

    prompt = (
        f"Review PR #{pr.get('number')}: {pr.get('title', '')}\n"
        f"Repo: {repo.get('full_name', '')}\n"
        f"Branch: {pr.get('head', {}).get('ref', '')} → {pr.get('base', {}).get('ref', '')}\n"
        f"\nPR body:\n{pr.get('body', '')[:2000]}"
    )

    db = get_db()
    worker = get_worker()
    task_id = f"gh-pr-{uuid.uuid4().hex[:12]}"

    source_meta = {
        "github_event": "pull_request",
        "repo": repo.get("full_name"),
        "pr_number": pr.get("number"),
        "action": action,
        "user": pr.get("user", {}).get("login"),
    }

    await db.create_task(
        task_id=task_id,
        prompt=prompt,
        source="github",
        source_meta=source_meta,
    )
    await db.add_log(task_id, f"Created from GitHub PR #{pr.get('number')} ({action})")
    await worker.enqueue(task_id)

    return {"status": "accepted", "task_id": task_id}
