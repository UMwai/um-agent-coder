"""Codex CLI subprocess client for rate-limit fallback.

Calls ``codex exec --json "<prompt>"`` and parses JSONL output to extract
the final assistant message.  Returns the same dict shape as
GeminiCodeAssistClient.generate() so it can be used as a drop-in fallback.

Constraint: ChatGPT Plus only supports the default model (no -m override).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class CodexSubprocessClient:
    """Async wrapper around ``codex exec --json``."""

    def __init__(
        self,
        cli_path: str = "codex",
        timeout: int = 45,
    ):
        self._cli_path = cli_path
        self._timeout = timeout

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        **_kwargs,
    ) -> dict:
        """Run codex exec and return ``{"text", "usage", "model"}``."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        cmd = [
            self._cli_path,
            "exec",
            "--json",
            "--skip-git-repo-check",  # don't scan repo
            "-C",
            "/tmp",  # run from /tmp to avoid loading project context
            full_prompt,
        ]
        logger.info("Codex fallback: invoking %s (timeout=%ds)", self._cli_path, self._timeout)
        t0 = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/tmp",
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            raise RuntimeError(f"Codex CLI timed out after {self._timeout}s")
        except FileNotFoundError:
            raise RuntimeError(f"Codex CLI not found at {self._cli_path}")

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if proc.returncode != 0:
            err = stderr.decode(errors="replace")[:500]
            raise RuntimeError(f"Codex CLI exited {proc.returncode}: {err}")

        # Parse JSONL output
        text = ""
        usage = {}
        for line in stdout.decode(errors="replace").strip().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            # item.completed contains the assistant's response text
            if etype == "item.completed":
                item = event.get("item", {})
                item_text = item.get("text", "")
                if item_text:
                    text += item_text

            # turn.completed contains token usage
            if etype == "turn.completed":
                u = event.get("usage", {})
                usage = {
                    "prompt_tokens": u.get("input_tokens", 0),
                    "completion_tokens": u.get("output_tokens", 0),
                    "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
                    "cached_tokens": u.get("cached_input_tokens", 0),
                    "finish_reason": "stop",
                }

            # error events
            if etype == "error":
                raise RuntimeError(f"Codex error: {event.get('message', 'unknown')}")

        if not text:
            raw = stdout.decode(errors="replace")[:500]
            logger.warning("Codex CLI returned no text. Raw: %s", raw)
            raise RuntimeError("Codex CLI returned empty response")

        if not usage:
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "finish_reason": "stop",
            }

        logger.info("Codex fallback: got %d chars in %dms", len(text), elapsed_ms)

        return {
            "text": text,
            "usage": usage,
            "model": "codex/gpt-5.2",
        }
