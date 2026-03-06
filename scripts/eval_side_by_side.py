#!/usr/bin/env python3
"""Side-by-side evaluation: Flash vs Pro vs Claude on the same response.

Sends the same prompt+response+eval_context to all three model tiers
concurrently and displays results in a comparison table.
"""

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

MODELS = ["flash", "pro", "pro-3.1"]


async def eval_with_model(client: httpx.AsyncClient, model: str, response_text: str) -> dict:
    """Run evaluation with a specific model tier."""
    start = time.monotonic()
    try:
        resp = await client.post(
            f"{BASE}/api/gemini/evaluate",
            json={
                "prompt": PROMPT_SUMMARY,
                "response": response_text,
                "eval_context": EVAL_CONTEXT,
                "model": model,
            },
        )
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            return {
                "model": model, "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "elapsed": elapsed,
            }

        data = resp.json()
        ev = data["evaluation"]
        return {
            "model": model,
            "eval_model": data["eval_model"],
            "duration_ms": data["duration_ms"],
            "elapsed": elapsed,
            "score": ev["score"],
            "accuracy": ev["accuracy"],
            "completeness": ev["completeness"],
            "clarity": ev["clarity"],
            "actionability": ev["actionability"],
            "issues": ev["issues"],
        }
    except Exception as e:
        return {"model": model, "error": str(e), "elapsed": time.monotonic() - start}


def print_comparison(results: list[dict]):
    """Print side-by-side comparison table."""
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not valid:
        print("All evaluations failed!")
        for r in errors:
            print(f"  {r['model']}: {r['error']}")
        return

    # Header
    print("\n" + "=" * 78)
    print("SIDE-BY-SIDE EVALUATION COMPARISON")
    print("=" * 78)

    # Model names row
    col_w = 18
    dim_w = 16
    print(f"\n{'Dimension':<{dim_w}}", end="")
    for r in valid:
        label = r["eval_model"].replace("gemini-", "").replace("-preview", "")
        print(f"  {label:>{col_w}}", end="")
    print()
    print("-" * (dim_w + (col_w + 2) * len(valid)))

    # Score rows
    dimensions = [
        ("Overall", "score"),
        ("Accuracy", "accuracy"),
        ("Completeness", "completeness"),
        ("Clarity", "clarity"),
        ("Actionability", "actionability"),
    ]

    for label, key in dimensions:
        print(f"{label:<{dim_w}}", end="")
        values = [r[key] for r in valid]
        min_val = min(values)
        max_val = max(values)
        for r in valid:
            val = r[key]
            # Mark min with ▼ and max with ▲
            marker = ""
            if len(valid) > 1:
                if val == min_val and min_val != max_val:
                    marker = " ▼"
                elif val == max_val and min_val != max_val:
                    marker = " ▲"
            print(f"  {val:>{col_w - len(marker)}.3f}{marker}", end="")
        print()

    # Duration row
    print(f"\n{'Duration (ms)':<{dim_w}}", end="")
    for r in valid:
        print(f"  {r['duration_ms']:>{col_w},}", end="")
    print()

    print(f"{'Wall time (s)':<{dim_w}}", end="")
    for r in valid:
        print(f"  {r['elapsed']:>{col_w}.1f}", end="")
    print()

    # Consensus (min per dimension)
    if len(valid) > 1:
        print(f"\n{'─' * (dim_w + (col_w + 2) * len(valid))}")
        print(f"{'CONSENSUS (min)':<{dim_w}}", end="")
        for _, key in dimensions:
            min_val = min(r[key] for r in valid)
            print(f"  {min_val:>{col_w}.3f}", end="")
        print()

    # Issues per model
    print("\n" + "=" * 78)
    print("ISSUES BY MODEL")
    print("=" * 78)

    for r in valid:
        label = r["eval_model"].replace("gemini-", "").replace("-preview", "")
        issues = r.get("issues", [])
        print(f"\n{label} ({len(issues)} issues):")
        if issues:
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("  (none)")

    # Unique issues (deduped)
    if len(valid) > 1:
        all_issues = []
        seen = set()
        for r in valid:
            for issue in r.get("issues", []):
                norm = issue.strip().lower()
                if norm not in seen:
                    seen.add(norm)
                    all_issues.append(issue)
        print(f"\nDEDUPLICATED ISSUES ({len(all_issues)} total):")
        for issue in all_issues:
            print(f"  • {issue}")

    # Errors
    if errors:
        print(f"\n{'=' * 78}")
        print("ERRORS")
        for r in errors:
            print(f"  {r['model']}: {r['error']}")

    # Strategy recommendation
    if len(valid) > 1:
        consensus_scores = {
            key: min(r[key] for r in valid)
            for _, key in dimensions
        }
        failing = [label for label, key in dimensions[1:] if consensus_scores[key] < 0.7]
        if failing:
            print(f"\n{'=' * 78}")
            print(f"RECOMMENDED STRATEGIES (dimensions < 0.7): {', '.join(failing)}")
            from um_agent_coder.daemon.routes.gemini._evaluator import EvalResult
            from um_agent_coder.daemon.routes.gemini._strategies import select_strategies
            er = EvalResult(
                score=consensus_scores["score"],
                accuracy=consensus_scores["accuracy"],
                completeness=consensus_scores["completeness"],
                clarity=consensus_scores["clarity"],
                actionability=consensus_scores["actionability"],
                issues=all_issues,
            )
            strategies = select_strategies(er, eval_context=EVAL_CONTEXT)
            for s in strategies:
                print(f"  → {s.name} (targets {s.dimension}, temp_delta={s.temperature_delta})")


async def main():
    # Read iteration 003 response
    with open("evals/gemini-intelligence-layer/iteration-003-multiturn.md") as f:
        full_text = f.read()

    response_start = full_text.index("### Turn 1")
    response_text = full_text[response_start:]

    print(f"Response: {len(response_text):,} chars")
    print(f"Eval context: {len(EVAL_CONTEXT):,} chars")
    print(f"Models: {', '.join(MODELS)}")
    print(f"\nRunning {len(MODELS)} evaluations in parallel...")

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        start = time.monotonic()
        results = await asyncio.gather(*[
            eval_with_model(client, model, response_text)
            for model in MODELS
        ])
        total = time.monotonic() - start
        print(f"All evaluations completed in {total:.1f}s")

    print_comparison(results)


if __name__ == "__main__":
    asyncio.run(main())
