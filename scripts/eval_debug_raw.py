#!/usr/bin/env python3
"""Debug: hit each model via the daemon's /enhance endpoint with a tiny eval-like
prompt to see raw output behavior. Then also test /evaluate to compare."""

import asyncio
import json
import time

import httpx

BASE = "https://um-agent-daemon-23o5bq3bfq-uc.a.run.app"

# Use the eval system prompt as a system_prompt + eval task as the user prompt
# This replicates what the evaluator does internally
EVAL_SYSTEM = (
    "You are a strict JSON-only response evaluator. "
    "Output EXACTLY this JSON format and NOTHING else — no markdown, no explanation:\n"
    '{"accuracy": N, "completeness": N, "clarity": N, "actionability": N, "issues": ["issue1"]}\n'
    "Scoring 1-10 scale."
)

EVAL_PROMPT = (
    "=== TASK ===\nWrite a Python function to add two numbers.\n=== END TASK ===\n\n"
    "=== RESPONSE ===\n"
    "```python\ndef add(a, b):\n    return a + b\n```\n"
    "=== END RESPONSE ===\n\n"
    "Now output your evaluation as a single JSON object. "
    "Do NOT output anything except the JSON:\n"
    '{"accuracy": N, "completeness": N, "clarity": N, "actionability": N, "issues": ["..."]}'
)

MODELS = ["flash", "pro", "pro-3.1"]


async def test_via_enhance(client: httpx.AsyncClient, model: str):
    """Use /enhance (no eval) to see what raw text each model produces for an eval prompt."""
    start = time.monotonic()
    resp = await client.post(
        f"{BASE}/api/gemini/enhance",
        json={
            "prompt": EVAL_PROMPT,
            "system_prompt": EVAL_SYSTEM,
            "model": model,
            "temperature": 0.1,
            "max_tokens": 2048,
            "enable_enhancement": False,
            "enable_self_eval": False,
        },
    )
    elapsed = time.monotonic() - start

    if resp.status_code != 200:
        return {"model": model, "error": f"HTTP {resp.status_code}: {resp.text[:300]}", "elapsed": elapsed}

    data = resp.json()
    return {
        "model": model,
        "actual_model": data.get("model", "?"),
        "response": data.get("response", ""),
        "duration_ms": data.get("duration_ms", 0),
        "usage": data.get("usage", {}),
        "elapsed": elapsed,
    }


async def test_via_evaluate(client: httpx.AsyncClient, model: str):
    """Use /evaluate directly to see parsed result."""
    start = time.monotonic()
    resp = await client.post(
        f"{BASE}/api/gemini/evaluate",
        json={
            "prompt": "Write a Python function to add two numbers.",
            "response": "```python\ndef add(a, b):\n    return a + b\n```",
            "model": model,
        },
    )
    elapsed = time.monotonic() - start

    if resp.status_code != 200:
        return {"model": model, "error": f"HTTP {resp.status_code}: {resp.text[:300]}", "elapsed": elapsed}

    return {"model": model, "data": resp.json(), "elapsed": elapsed}


async def main():
    print("=" * 70)
    print("TEST 1: Raw model output via /enhance (no eval, no enhancement)")
    print("  Sending eval-formatted prompt to see what each model returns")
    print("=" * 70)

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        # Run enhance tests in parallel
        results = await asyncio.gather(*[
            test_via_enhance(client, m) for m in MODELS
        ])

        for r in results:
            model = r["model"]
            print(f"\n{'─' * 60}")
            print(f"MODEL: {model} (actual: {r.get('actual_model', '?')})")
            print(f"Duration: {r.get('duration_ms', '?')}ms | Wall: {r['elapsed']:.1f}s")

            if "error" in r:
                print(f"ERROR: {r['error']}")
                continue

            text = r["response"]
            usage = r.get("usage", {})
            print(f"Finish: {usage.get('finish_reason', '?')} | Tokens: {usage.get('total_tokens', '?')}")
            print(f"Response ({len(text)} chars):")
            print(f"  >>>{text[:500]}<<<")
            if len(text) > 500:
                print(f"  [... {len(text) - 500} more chars]")

            # Try parsing
            from um_agent_coder.daemon.routes.gemini._evaluator import _parse_eval_response
            parsed = _parse_eval_response(text)
            print(f"  Parse result: {json.dumps(parsed, indent=2) if parsed else 'FAILED'}")

        # Now test /evaluate endpoint
        print(f"\n{'=' * 70}")
        print("TEST 2: /evaluate endpoint (uses evaluator internally)")
        print("=" * 70)

        eval_results = await asyncio.gather(*[
            test_via_evaluate(client, m) for m in MODELS
        ])

        for r in eval_results:
            model = r["model"]
            print(f"\n{'─' * 60}")
            print(f"MODEL: {model} | Wall: {r['elapsed']:.1f}s")

            if "error" in r:
                print(f"ERROR: {r['error']}")
                continue

            data = r["data"]
            ev = data["evaluation"]
            print(f"Eval model: {data['eval_model']} | Duration: {data['duration_ms']}ms")
            print(f"Scores: overall={ev['score']:.2f} acc={ev['accuracy']:.2f} "
                  f"comp={ev['completeness']:.2f} clarity={ev['clarity']:.2f} "
                  f"action={ev['actionability']:.2f}")
            print(f"Issues: {ev['issues']}")


if __name__ == "__main__":
    asyncio.run(main())
