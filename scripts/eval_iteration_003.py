#!/usr/bin/env python3
"""Run Gemini self-eval on iteration 003 output via the enhance endpoint."""

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
2. Priority 10 = highest urgency, 1 = lowest. Check priority assignments make sense.
3. Recommendation fields must match the model exactly (no invented fields)
4. All pipelines must be async and use SQLAlchemy 2.0 async patterns
5. Each pipeline MUST actually call an LLM for analysis, not just do rule-based logic
"""


async def main():
    # Read the iteration 003 response
    with open("evals/gemini-intelligence-layer/iteration-003-multiturn.md") as f:
        full_text = f.read()

    # Extract just the response part (after "## Response")
    response_start = full_text.index("### Turn 1")
    response_text = full_text[response_start:]

    # The original prompt (abbreviated for the evaluator)
    prompt_summary = (
        "Implement a complete AI Financial Advisor Engine with 10 files: "
        "schemas.py, prompts.py, engine.py, 6 pipeline modules "
        "(spending_analyzer, cashflow_advisor, debt_strategist, investment_monitor, "
        "savings_detector, tax_planner), and scheduler jobs.py. "
        "Must use existing CashFlowEngine, DebtOptimizer, TradingProdBridge APIs. "
        "All async, must call LLM, confidence >= 0.7 threshold, deduplication."
    )

    print("=" * 70)
    print("Evaluating Iteration 003 via Gemini /enhance endpoint")
    print(f"Response length: {len(response_text)} chars")
    print(f"Eval context: {len(EVAL_CONTEXT)} chars")
    print("=" * 70)

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        # Use enhance endpoint with a dummy prompt that contains the eval material
        # The trick: we send the response AS the prompt, disable enhancement,
        # and enable self-eval with eval_context
        start = time.monotonic()
        resp = await client.post(
            f"{BASE}/api/gemini/enhance",
            json={
                "prompt": prompt_summary,
                "system_prompt": f"Here is the code to evaluate:\n\n{response_text[:20000]}",
                "enable_enhancement": False,
                "enable_self_eval": True,
                "eval_model": "flash",
                "eval_context": EVAL_CONTEXT,
                "model": "flash",
                "temperature": 0.1,
                "max_tokens": 256,
            },
        )
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            print(f"FAILED: {resp.status_code}")
            print(resp.text[:500])
            return

        data = resp.json()
        print(f"\nDuration: {data['duration_ms']}ms (wall: {elapsed:.1f}s)")

        if data.get("evaluation"):
            ev = data["evaluation"]
            print(f"\nGemini Self-Evaluation (Flash + eval_context):")
            print(f"  Overall:       {ev['score']:.2f}")
            print(f"  Accuracy:      {ev['accuracy']:.2f}")
            print(f"  Completeness:  {ev['completeness']:.2f}")
            print(f"  Clarity:       {ev['clarity']:.2f}")
            print(f"  Actionability: {ev['actionability']:.2f}")
            print(f"  Retries:       {ev['retry_count']}")
            if ev["issues"]:
                print(f"\n  Issues:")
                for issue in ev["issues"]:
                    print(f"    - {issue}")

            # Save eval results
            print(f"\n  Result: {json.dumps(ev, indent=2)}")
        else:
            print("No evaluation returned!")
            print(json.dumps(data, indent=2)[:1000])


if __name__ == "__main__":
    asyncio.run(main())
