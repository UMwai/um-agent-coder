"""POST /api/gemini/agent — Agentic tool-use loop via prompt simulation."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from um_agent_coder.daemon.auth import verify_api_key

from .models import (
    AgentRequest,
    AgentResponse,
    AgentStatus,
    AgentStep,
    AgentToolCall,
    GeminiModelTier,
    GEMINI_MODEL_MAP,
    UsageInfo,
)
from ._tools import execute_tool, format_tools_for_prompt, get_tools

logger = logging.getLogger(__name__)

router = APIRouter()

AGENT_SYSTEM_PROMPT = """You are an autonomous agent that can use tools to complete tasks.

{tools_section}

You MUST respond in ONE of these two formats:

Format 1 - When you need to use a tool:
THOUGHT: <your reasoning about what to do next>
ACTION: tool_name({{"param1": "value1", "param2": "value2"}})

Format 2 - When you have the final answer:
THOUGHT: <your final reasoning>
ANSWER: <your complete answer to the task>

Rules:
- Use EXACTLY one ACTION per response, OR provide an ANSWER
- ACTION arguments must be valid JSON
- Always think before acting
- When you have enough information, provide the ANSWER
- Be concise but thorough in your ANSWER"""

# Parsing patterns
THOUGHT_PATTERN = re.compile(r"THOUGHT:\s*(.+?)(?=\n(?:ACTION|ANSWER):|\Z)", re.DOTALL)
ACTION_PATTERN = re.compile(r"ACTION:\s*(\w+)\((.+)\)\s*$", re.DOTALL | re.MULTILINE)
ANSWER_PATTERN = re.compile(r"ANSWER:\s*(.+)", re.DOTALL)


def _get_client():
    from um_agent_coder.daemon.app import get_gemini_client
    return get_gemini_client()


def _get_settings():
    from um_agent_coder.daemon.app import get_settings
    return get_settings()


def _parse_action(text: str) -> Optional[tuple[str, dict]]:
    """Parse an ACTION line into (tool_name, args_dict)."""
    match = ACTION_PATTERN.search(text)
    if not match:
        return None
    tool_name = match.group(1)
    args_str = match.group(2).strip()
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        # Try fixing common issues
        try:
            args = json.loads(args_str.replace("'", '"'))
        except json.JSONDecodeError:
            return tool_name, {"_raw": args_str}
    return tool_name, args if isinstance(args, dict) else {"_raw": str(args)}


def _parse_agent_response(text: str) -> tuple[str, Optional[tuple[str, dict]], Optional[str]]:
    """Parse agent response into (thought, action, answer)."""
    thought_match = THOUGHT_PATTERN.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    answer_match = ANSWER_PATTERN.search(text)
    if answer_match:
        return thought, None, answer_match.group(1).strip()

    action = _parse_action(text)
    return thought, action, None


@router.post("/agent", response_model=AgentResponse)
async def run_agent(
    req: AgentRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Execute an agentic task with tool-use loop."""
    client = _get_client()
    settings = _get_settings()
    agent_id = f"ga-{uuid.uuid4().hex[:12]}"
    start = time.monotonic()

    max_steps = min(req.max_steps, settings.gemini_agent_max_steps)
    model_name = GEMINI_MODEL_MAP.get(req.model.value, GEMINI_MODEL_MAP["pro-3.1"])

    # Get tools
    tools = get_tools(req.tools)
    tools_section = format_tools_for_prompt(tools)
    system_prompt = AGENT_SYSTEM_PROMPT.format(tools_section=tools_section)

    # Build initial conversation
    contents = [
        {"role": "user", "parts": [{"text": system_prompt}]},
        {"role": "model", "parts": [{"text": "I understand. I will use the THOUGHT/ACTION/ANSWER format."}]},
        {"role": "user", "parts": [{"text": f"Task: {req.task}"}]},
    ]

    steps: list[AgentStep] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    answer = None
    status = AgentStatus.running

    for step_num in range(1, max_steps + 1):
        try:
            result = await client.generate_multi_turn(
                contents=contents,
                model=model_name,
                temperature=req.temperature,
                max_tokens=4096,
                timeout=60.0,
            )
        except Exception as e:
            from um_agent_coder.daemon.gemini_client import RateLimitError
            if isinstance(e, RateLimitError):
                raise HTTPException(status_code=429, detail=str(e))
            logger.error("Agent step %d failed: %s", step_num, e)
            status = AgentStatus.failed
            steps.append(AgentStep(step=step_num, thought=f"Error: {e}"))
            break

        # Accumulate usage
        usage = result.get("usage", {})
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

        response_text = result.get("text", "")
        thought, action, step_answer = _parse_agent_response(response_text)

        # Add model response to conversation
        contents.append({"role": "model", "parts": [{"text": response_text}]})

        if step_answer is not None:
            # Agent provided final answer
            steps.append(AgentStep(step=step_num, thought=thought))
            answer = step_answer
            status = AgentStatus.completed
            break

        if action is not None:
            tool_name, tool_args = action
            # Execute tool
            observation = execute_tool(tool_name, tool_args, tools)

            steps.append(AgentStep(
                step=step_num,
                thought=thought,
                action=AgentToolCall(tool=tool_name, args=tool_args, result=observation),
                observation=observation,
            ))

            # Feed observation back as user message
            contents.append({
                "role": "user",
                "parts": [{"text": f"OBSERVATION: {observation}"}],
            })
        else:
            # No action or answer parsed — treat as confused, nudge
            steps.append(AgentStep(step=step_num, thought=thought or response_text))
            contents.append({
                "role": "user",
                "parts": [{"text": "Please respond with either an ACTION or an ANSWER."}],
            })
    else:
        # Max steps reached without answer
        status = AgentStatus.max_steps_reached
        # Use last thought as partial answer
        if steps:
            answer = steps[-1].thought

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "Agent %s finished: status=%s steps=%d duration=%dms",
        agent_id, status, len(steps), duration_ms,
    )

    return AgentResponse(
        id=agent_id,
        task=req.task,
        status=status,
        answer=answer,
        steps=steps,
        total_steps=len(steps),
        duration_ms=duration_ms,
        usage=UsageInfo(**total_usage),
    )
