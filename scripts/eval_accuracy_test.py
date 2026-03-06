#!/usr/bin/env python3
"""Test the new accuracy-first evaluator against the um-cfo response.

Calls the daemon's /enhance endpoint to run the accuracy system prompt
against the um-cfo response, then parses + scores the checklist.
"""

import asyncio
import json
import time

import httpx

from um_agent_coder.daemon.routes.gemini._evaluator import (
    ACCURACY_SYSTEM_PROMPT,
    AccuracyCheck,
    _parse_accuracy_checks,
    _score_accuracy_checks,
    SEVERITY_WEIGHTS,
)

BASE = "https://um-agent-daemon-23o5bq3bfq-uc.a.run.app"

EVAL_CONTEXT = """
## Actual API Signatures (from um-cfo source code)

### CashFlowEngine (src/core/cashflow/__init__.py):
- project_bills(bills, start, end) -> List[ProjectedBill(date, amount, name)]
- forecast_daily(accounts, bills, transactions, days=90) -> List[DayForecast(date, projected_balance, bills_due, net_flow)]

### DebtOptimizer (src/core/debt/optimizer.py):
- simulate(debts, extra_payment, strategy='avalanche'|'snowball') -> DebtPayoffResult(months, total_interest, schedule[])

### ApprovalWorkflow (src/core/approval/workflow.py):
- submit(recommendation) -> sets status=PENDING
- approve(recommendation_id) -> PENDING->APPROVED
- execute(recommendation_id) -> APPROVED->EXECUTED
- reject(recommendation_id, reason) -> PENDING->REJECTED

### NotificationDispatcher (src/notifications/dispatcher.py):
- send(channel, title, message, priority) -> sends to Discord/Slack

### TradingProdBridge (src/integrations/trading_prod.py):
- get_positions() -> List[Position(symbol, qty, avg_cost, current_price, pnl)]
- get_summary() -> PortfolioSummary(total_value, day_pnl, positions_count)

### Recommendation model fields:
id, type (RecommendationType enum), title, description, impact_amount, confidence,
priority (1-10, where 10=HIGHEST priority), status, action_payload (JSONB), expires_at

### RecommendationType enum:
BUDGET_ADJUSTMENT, DEBT_PAYMENT, INVESTMENT_REBALANCE, BILL_ALERT,
SAVINGS_OPPORTUNITY, TAX_OPTIMIZATION, SPENDING_ALERT, CASHFLOW_WARNING, GENERAL_INSIGHT

### Database pattern:
async with get_db() as db:
    result = await db.execute(select(Model).where(...))
    items = result.scalars().all()

CRITICAL CHECKS:
1. Code must call existing engines with EXACTLY these signatures (no extra/missing params)
2. Priority 10 = highest urgency, 1 = lowest
3. Recommendation fields must match the model exactly (no invented fields)
4. All pipelines must be async and use SQLAlchemy 2.0 async patterns
5. Each pipeline MUST actually call an LLM for analysis, not just rule-based logic
6. All 10 files must be present and complete (schemas.py, prompts.py, engine.py, 6 pipelines, jobs.py)
"""

PROMPT = (
    "Implement a complete AI Financial Advisor Engine (src/core/ai/) with these files:\n"
    "1. schemas.py 2. prompts.py 3. engine.py 4. spending_analyzer.py\n"
    "5. cashflow_advisor.py 6. debt_strategist.py 7. investment_monitor.py\n"
    "8. savings_detector.py 9. tax_planner.py 10. jobs.py\n"
    "Requirements: async, LLM calls in every pipeline."
)


async def main():
    with open("evals/gemini-intelligence-layer/iteration-003-multiturn.md") as f:
        full_text = f.read()

    response_start = full_text.index("### Turn 1")
    response_text = full_text[response_start:]

    # Build the accuracy eval prompt (same as evaluate_accuracy does internally)
    MAX_RESPONSE_CHARS = 60_000
    if len(response_text) > MAX_RESPONSE_CHARS:
        half = MAX_RESPONSE_CHARS // 2
        truncated = response_text[:half] + f"\n\n[...omitted...]\n\n" + response_text[-half:]
    else:
        truncated = response_text

    eval_prompt = (
        "=== TASK DESCRIPTION ===\n"
        f"{PROMPT}\n"
        "=== END TASK DESCRIPTION ===\n\n"
        "=== REFERENCE MATERIAL (ground truth — check ALL code against this) ===\n"
        f"{EVAL_CONTEXT}\n"
        "=== END REFERENCE MATERIAL ===\n\n"
        "=== CODE RESPONSE TO AUDIT ===\n"
        f"{truncated}\n"
        "=== END CODE RESPONSE ===\n\n"
        "Now audit this code response against the reference material. "
        "Check EVERY function call, import, file path, and pattern. "
        "Output your checklist as JSON:\n"
        '{"checks": [{"check": "...", "status": "pass"|"fail", '
        '"severity": "breaking"|"foreign"|"style", "detail": "..."}]}'
    )

    print(f"Response: {len(response_text):,} chars")
    print(f"Eval prompt: {len(eval_prompt):,} chars")
    print(f"Model: gemini-3.1-pro-preview")
    print()

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        start = time.monotonic()
        resp = await client.post(
            f"{BASE}/api/gemini/enhance",
            json={
                "prompt": eval_prompt,
                "system_prompt": ACCURACY_SYSTEM_PROMPT,
                "model": "pro-3.1",
                "temperature": 0.1,
                "max_tokens": 16384,
                "enable_enhancement": False,
                "enable_self_eval": False,
            },
        )
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            print(f"FAILED: {resp.status_code}")
            print(resp.text[:500])
            return

        data = resp.json()
        text = data.get("response", "")
        usage = data.get("usage", {})

        print(f"Duration: {data.get('duration_ms', '?')}ms | Wall: {elapsed:.1f}s")
        print(f"Finish: {usage.get('finish_reason', '?')} | Tokens: {usage.get('total_tokens', '?')}")
        print(f"Raw response: {len(text)} chars")

        # Parse
        parsed = _parse_accuracy_checks(text)
        if not parsed:
            print(f"\nPARSE FAILED. Raw text:")
            print(text[:2000])
            return

        checks_raw = parsed.get("checks", [])
        checks = [
            AccuracyCheck(
                check=c.get("check", ""),
                status=c.get("status", "fail"),
                severity=c.get("severity", "breaking"),
                detail=c.get("detail", ""),
            )
            for c in checks_raw
        ]

        score = _score_accuracy_checks(checks)
        passed = [c for c in checks if c.status == "pass"]
        failed = [c for c in checks if c.status == "fail"]

        print(f"\n{'=' * 70}")
        print(f"ACCURACY CHECKLIST: {len(checks)} checks, {len(passed)} passed, {len(failed)} failed")
        print(f"ACCURACY SCORE: {score:.3f}")
        print(f"{'=' * 70}")

        # Show passing checks
        print(f"\n✅ PASSED ({len(passed)}):")
        for c in passed:
            print(f"  [{c.severity}] {c.check}")

        # Show failing checks grouped by severity
        breaking = [c for c in failed if c.severity == "breaking"]
        foreign = [c for c in failed if c.severity == "foreign"]
        style = [c for c in failed if c.severity == "style"]

        if breaking:
            print(f"\n🚨 BREAKING FAILURES ({len(breaking)}) — 0 points each:")
            for c in breaking:
                print(f"  ✗ {c.check}")
                print(f"    → {c.detail}")

        if foreign:
            print(f"\n⚠️ FOREIGN DEP FAILURES ({len(foreign)}) — 0.5 points each:")
            for c in foreign:
                print(f"  ✗ {c.check}")
                print(f"    → {c.detail}")

        if style:
            print(f"\n📋 STYLE DEVIATIONS ({len(style)}) — 0.75 points each:")
            for c in style:
                print(f"  ✗ {c.check}")
                print(f"    → {c.detail}")

        # Compare to Claude manual eval
        print(f"\n{'=' * 70}")
        print("COMPARISON TO CLAUDE MANUAL EVAL")
        print(f"{'=' * 70}")
        print(f"  Automated accuracy:  {score:.3f}")
        print(f"  Claude manual:       0.600")
        print(f"  Gemini Pro (iter003): 0.800")
        print(f"  Gemini Flash:        0.900-1.000")


if __name__ == "__main__":
    asyncio.run(main())
