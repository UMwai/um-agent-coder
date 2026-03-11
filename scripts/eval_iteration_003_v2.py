#!/usr/bin/env python3
"""Evaluate iteration 003 output via the new /api/gemini/evaluate endpoint."""

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

PROMPT_SUMMARY = (
    "Implement a complete AI Financial Advisor Engine (src/core/ai/) with these files:\n"
    "1. schemas.py - FinancialContext, RecommendationCreate, PipelineResult\n"
    "2. prompts.py - Structured prompt templates per pipeline\n"
    "3. engine.py - AIFinancialAdvisorEngine (gather_context, run_analysis, _deduplicate)\n"
    "4. spending_analyzer.py - Compare spending vs 90-day avg, flag price increases\n"
    "5. cashflow_advisor.py - Use CashFlowEngine.forecast_daily(), detect overdraft risk\n"
    "6. debt_strategist.py - Use DebtOptimizer.simulate(), compare avalanche vs snowball\n"
    "7. investment_monitor.py - Use TradingProdBridge.get_positions(), detect concentration\n"
    "8. savings_detector.py - Find duplicate subscriptions, idle cash\n"
    "9. tax_planner.py - Tax-loss harvesting, quarterly liability\n"
    "10. jobs.py - Scheduler integration with notification dispatch\n\n"
    "Requirements: async, asyncio.gather, confidence >= 0.7, dedup, LLM calls in every pipeline."
)


async def main():
    # Read iteration 003 response
    with open("evals/gemini-intelligence-layer/iteration-003-multiturn.md") as f:
        full_text = f.read()

    # Extract response (after metadata header)
    response_start = full_text.index("### Turn 1")
    response_text = full_text[response_start:]

    print("=" * 70)
    print("Evaluating Iteration 003 via POST /api/gemini/evaluate")
    print(f"Prompt: {len(PROMPT_SUMMARY)} chars")
    print(f"Response: {len(response_text)} chars")
    print(f"Eval context: {len(EVAL_CONTEXT)} chars")
    print("=" * 70)

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        start = time.monotonic()
        resp = await client.post(
            f"{BASE}/api/gemini/evaluate",
            json={
                "prompt": PROMPT_SUMMARY,
                "response": response_text,
                "eval_context": EVAL_CONTEXT,
                "model": "pro",
            },
        )
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            print(f"FAILED: {resp.status_code}")
            print(resp.text[:500])
            return

        data = resp.json()
        ev = data["evaluation"]

        print(f"\nEval model: {data['eval_model']}")
        print(f"Duration: {data['duration_ms']}ms (wall: {elapsed:.1f}s)")
        print(f"\nScores:")
        print(f"  Overall:       {ev['score']:.2f}")
        print(f"  Accuracy:      {ev['accuracy']:.2f}")
        print(f"  Completeness:  {ev['completeness']:.2f}")
        print(f"  Clarity:       {ev['clarity']:.2f}")
        print(f"  Actionability: {ev['actionability']:.2f}")

        if ev["issues"]:
            print(f"\nIssues ({len(ev['issues'])}):")
            for issue in ev["issues"]:
                print(f"  - {issue}")
        else:
            print("\nNo issues found.")

        print(f"\nFull result: {json.dumps(ev, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
