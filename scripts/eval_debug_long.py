#!/usr/bin/env python3
"""Debug: test eval with the actual long um-cfo response to find what breaks."""

import asyncio
import json
import time

import httpx

BASE = "https://um-agent-daemon-23o5bq3bfq-uc.a.run.app"

EVAL_CONTEXT = """
## Actual API Signatures (from um-cfo source code)

### CashFlowEngine (src/core/cashflow/__init__.py):
- project_bills(bills, start, end) -> List[ProjectedBill(date, amount, name)]
- forecast_daily(accounts, bills, transactions, days=90) -> List[DayForecast(date, projected_balance, bills_due, net_flow)]

### DebtOptimizer (src/core/debt/optimizer.py):
- simulate(debts, extra_payment, strategy='avalanche'|'snowball') -> DebtPayoffResult(months, total_interest, schedule[])

CRITICAL CHECKS:
1. Code must call existing engines with EXACTLY these signatures
2. Priority 10 = highest urgency, 1 = lowest
"""

PROMPT = (
    "Implement a complete AI Financial Advisor Engine (src/core/ai/) with these files:\n"
    "1. schemas.py 2. prompts.py 3. engine.py 4-9. pipelines 10. jobs.py\n"
    "Requirements: async, LLM calls in every pipeline."
)

MODELS = ["flash", "pro", "pro-3.1"]


async def test_eval(client: httpx.AsyncClient, model: str, response_text: str):
    """Test /evaluate with the long response."""
    start = time.monotonic()
    try:
        resp = await client.post(
            f"{BASE}/api/gemini/evaluate",
            json={
                "prompt": PROMPT,
                "response": response_text,
                "eval_context": EVAL_CONTEXT,
                "model": model,
            },
        )
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            print(f"\n{model}: HTTP {resp.status_code} in {elapsed:.1f}s")
            print(f"  Body: {resp.text[:500]}")
            return

        data = resp.json()
        ev = data["evaluation"]
        print(f"\n{model} -> {data['eval_model']}")
        print(f"  Duration: {data['duration_ms']}ms | Wall: {elapsed:.1f}s")
        print(f"  Scores: overall={ev['score']:.2f} acc={ev['accuracy']:.2f} "
              f"comp={ev['completeness']:.2f} clarity={ev['clarity']:.2f} "
              f"action={ev['actionability']:.2f}")
        print(f"  Issues ({len(ev['issues'])}): {ev['issues'][:3]}")

        # Check if it's the default fallback (all zeros except score=0.7)
        if ev['accuracy'] == 0 and ev['completeness'] == 0 and ev['clarity'] == 0:
            print(f"  ⚠️  ALL ZEROS — this is the parse-failure fallback!")
    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"\n{model}: EXCEPTION after {elapsed:.1f}s: {type(e).__name__}: {e}")


async def test_enhance_raw(client: httpx.AsyncClient, model: str, response_text: str):
    """Use /enhance to get raw text and see what the model actually returns."""
    # Build same eval prompt as the evaluator
    MAX_RESPONSE_CHARS = 40_000
    if len(response_text) > MAX_RESPONSE_CHARS:
        half = MAX_RESPONSE_CHARS // 2
        truncated = (
            response_text[:half]
            + f"\n\n[... {len(response_text) - MAX_RESPONSE_CHARS} chars omitted ...]\n\n"
            + response_text[-half:]
        )
    else:
        truncated = response_text

    eval_prompt = (
        "=== TASK DESCRIPTION ===\n"
        f"{PROMPT[:8000]}\n"
        "=== END TASK DESCRIPTION ===\n\n"
        "=== REFERENCE MATERIAL ===\n"
        f"{EVAL_CONTEXT}\n"
        "=== END REFERENCE MATERIAL ===\n\n"
        "=== RESPONSE TO EVALUATE ===\n"
        f"{truncated}\n"
        "=== END RESPONSE ===\n\n"
        "Now output your evaluation as a single JSON object. "
        "Do NOT output anything except the JSON:\n"
        '{"accuracy": N, "completeness": N, "clarity": N, "actionability": N, "issues": ["..."]}'
    )

    system = (
        "You are a strict JSON-only response evaluator. "
        "Output EXACTLY this JSON format and NOTHING else — no markdown, no explanation:\n"
        '{"accuracy": N, "completeness": N, "clarity": N, "actionability": N, "issues": ["issue1"]}\n'
        "Scoring 1-10 scale."
    )

    print(f"\n{model}: Sending {len(eval_prompt)} char eval prompt...")
    start = time.monotonic()
    try:
        resp = await client.post(
            f"{BASE}/api/gemini/enhance",
            json={
                "prompt": eval_prompt,
                "system_prompt": system,
                "model": model,
                "temperature": 0.1,
                "max_tokens": 8192,
                "enable_enhancement": False,
                "enable_self_eval": False,
            },
        )
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} in {elapsed:.1f}s: {resp.text[:300]}")
            return

        data = resp.json()
        text = data.get("response", "")
        usage = data.get("usage", {})
        finish = usage.get("finish_reason", "?")

        print(f"  Duration: {data.get('duration_ms', '?')}ms | Wall: {elapsed:.1f}s")
        print(f"  Finish: {finish} | Tokens: {usage.get('total_tokens', '?')}")
        print(f"  Response ({len(text)} chars):")
        # Show first and last 300 chars
        if len(text) > 600:
            print(f"  START: >>>{text[:300]}<<<")
            print(f"  END:   >>>{text[-300:]}<<<")
        else:
            print(f"  >>>{text}<<<")

        from um_agent_coder.daemon.routes.gemini._evaluator import _parse_eval_response
        parsed = _parse_eval_response(text)
        if parsed:
            print(f"  ✅ PARSED: {json.dumps(parsed)}")
        else:
            print(f"  ❌ PARSE FAILED")

    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"  EXCEPTION after {elapsed:.1f}s: {type(e).__name__}: {e}")


async def main():
    with open("evals/gemini-intelligence-layer/iteration-003-multiturn.md") as f:
        full_text = f.read()

    response_start = full_text.index("### Turn 1")
    response_text = full_text[response_start:]
    print(f"Response: {len(response_text):,} chars")

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        # Test 1: /evaluate endpoint (same as side-by-side)
        print("\n" + "=" * 60)
        print("TEST 1: /evaluate endpoint with long response")
        print("=" * 60)
        await asyncio.gather(*[
            test_eval(client, m, response_text) for m in MODELS
        ])

        # Test 2: Raw output via /enhance
        print("\n" + "=" * 60)
        print("TEST 2: Raw model output via /enhance (see actual text)")
        print("=" * 60)
        # Run sequentially to avoid rate limits and see each clearly
        for model in MODELS:
            await test_enhance_raw(client, model, response_text)


if __name__ == "__main__":
    asyncio.run(main())
