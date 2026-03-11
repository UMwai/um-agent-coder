#!/usr/bin/env python3
"""Submit the AI Financial Advisor Engine task (iteration 002) with eval_context."""

import asyncio
import json
import time

import httpx

BASE = "https://um-agent-daemon-23o5bq3bfq-uc.a.run.app"

# Reference material for the evaluator — actual API signatures from um-cfo
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

TASK = (
    "You are designing the complete implementation for an AI Financial Advisor Engine "
    "(src/core/ai/) for a personal finance app built with FastAPI + PostgreSQL + SQLAlchemy 2.0 async.\n\n"
    "## Existing Codebase (already implemented, do NOT rewrite)\n\n"
    "### Models (SQLAlchemy ORM, UUID PKs, created_at/updated_at):\n"
    "- Account: id, name, account_type (CHECKING/SAVINGS/CREDIT_CARD/BROKERAGE/CRYPTO/LOAN/OTHER), balance, institution, currency, is_active\n"
    "- Transaction: id, account_id, amount, description, date, category_id, merchant, is_recurring, fingerprint (SHA256 dedup)\n"
    "- Budget: id, category_id, amount, period (WEEKLY/BIWEEKLY/MONTHLY/QUARTERLY/ANNUAL), start_date, end_date\n"
    "- Bill: id, name, amount, due_date, frequency, account_id, is_auto_pay, alert_days_before, is_active\n"
    "- Debt: id, name, balance, apr, minimum_payment, account_id, debt_type\n"
    "- Investment: id, account_id, symbol, quantity, cost_basis, current_price, asset_type\n"
    "- Recommendation: id, type (RecommendationType enum), title, description, impact_amount, confidence, "
    "priority (1-10), status (PENDING/APPROVED/REJECTED/EXECUTED), action_payload (JSONB), expires_at\n"
    "- Category: id, name, parent_id (hierarchical)\n\n"
    "### RecommendationType enum values:\n"
    "BUDGET_ADJUSTMENT, DEBT_PAYMENT, INVESTMENT_REBALANCE, BILL_ALERT, SAVINGS_OPPORTUNITY, "
    "TAX_OPTIMIZATION, SPENDING_ALERT, CASHFLOW_WARNING, GENERAL_INSIGHT\n\n"
    "### Existing engines:\n"
    "1. CashFlowEngine (src/core/cashflow/__init__.py):\n"
    "   - project_bills(bills, start, end) -> List[ProjectedBill(date, amount, name)]\n"
    "   - forecast_daily(accounts, bills, transactions, days=90) -> List[DayForecast(date, projected_balance, bills_due, net_flow)]\n\n"
    "2. DebtOptimizer (src/core/debt/optimizer.py):\n"
    "   - simulate(debts, extra_payment, strategy='avalanche'|'snowball') -> DebtPayoffResult(months, total_interest, schedule[])\n\n"
    "3. ApprovalWorkflow (src/core/approval/workflow.py):\n"
    "   - submit(recommendation) -> sets status=PENDING\n"
    "   - approve(recommendation_id) -> PENDING->APPROVED\n"
    "   - execute(recommendation_id) -> APPROVED->EXECUTED\n"
    "   - reject(recommendation_id, reason) -> PENDING->REJECTED\n\n"
    "4. NotificationDispatcher (src/notifications/dispatcher.py):\n"
    "   - send(channel, title, message, priority) -> sends to Discord/Slack\n\n"
    "5. TradingProdBridge (src/integrations/trading_prod.py):\n"
    "   - get_positions() -> List[Position(symbol, qty, avg_cost, current_price, pnl)]\n"
    "   - get_summary() -> PortfolioSummary(total_value, day_pnl, positions_count)\n\n"
    "### Database pattern:\n"
    "```python\n"
    "async with get_db() as db:\n"
    "    result = await db.execute(select(Account).where(Account.is_active == True))\n"
    "    accounts = result.scalars().all()\n"
    "```\n\n"
    "## What to implement\n\n"
    "### File: src/core/ai/engine.py - AIFinancialAdvisorEngine class\n\n"
    "```python\n"
    "class AIFinancialAdvisorEngine:\n"
    "    def __init__(self, db: AsyncSession): ...\n"
    "    async def gather_context(self) -> FinancialContext: ...\n"
    "    async def run_analysis(self) -> list[RecommendationCreate]: ...\n"
    "    async def _deduplicate(self, recs: list[RecommendationCreate]) -> list[RecommendationCreate]: ...\n"
    "```\n\n"
    "### File: src/core/ai/pipelines/ - 6 analysis pipeline modules\n"
    "Each pipeline is a class with `async def analyze(self, context: FinancialContext) -> list[RecommendationCreate]`\n\n"
    "1. spending_analyzer.py - SpendingAnalyzer\n"
    "   - Compare current month spending per category vs 90-day average\n"
    "   - Flag merchants with >20% price increase on recurring charges\n"
    "   - Detect category drift (spending shifting between categories)\n"
    "   - Output: SPENDING_ALERT, BUDGET_ADJUSTMENT recommendations\n\n"
    "2. cashflow_advisor.py - CashflowAdvisor\n"
    "   - Use existing CashFlowEngine.forecast_daily()\n"
    "   - Detect upcoming overdraft risk (balance < $500)\n"
    "   - Identify large upcoming bills vs available balance\n"
    "   - Generate natural language narrative\n"
    "   - Output: CASHFLOW_WARNING, GENERAL_INSIGHT recommendations\n\n"
    "3. debt_strategist.py - DebtStrategist\n"
    "   - Use existing DebtOptimizer.simulate() with both strategies\n"
    "   - Compare avalanche vs snowball: if savings > $100, recommend switch\n"
    "   - Detect high-APR debts that should be prioritized\n"
    "   - Calculate optimal extra payment allocation\n"
    "   - Output: DEBT_PAYMENT recommendations\n\n"
    "4. investment_monitor.py - InvestmentMonitor\n"
    "   - Use TradingProdBridge.get_positions() for current holdings\n"
    "   - Detect portfolio concentration (any position > 20% of total)\n"
    "   - Flag significant unrealized losses (> $1000)\n"
    "   - Check cash drag (uninvested cash > 10% of portfolio)\n"
    "   - Output: INVESTMENT_REBALANCE recommendations\n\n"
    "5. savings_detector.py - SavingsDetector\n"
    "   - Identify recurring subscription charges by merchant pattern\n"
    "   - Flag duplicate services (multiple streaming, multiple cloud storage)\n"
    "   - Detect idle cash in low-yield checking accounts\n"
    "   - Calculate potential savings from switching/cancelling\n"
    "   - Output: SAVINGS_OPPORTUNITY, BILL_ALERT recommendations\n\n"
    "6. tax_planner.py - TaxPlanner\n"
    "   - Identify tax-loss harvesting opportunities (unrealized losses > $1000)\n"
    "   - Estimate quarterly tax liability from realized gains\n"
    "   - Flag large deductible expenses\n"
    "   - Output: TAX_OPTIMIZATION recommendations\n\n"
    "### File: src/core/ai/prompts.py - Prompt templates\n"
    "Each pipeline constructs a structured prompt with:\n"
    "- Financial summary metrics (NOT raw data)\n"
    "- Specific analysis questions\n"
    "- Required JSON output format matching RecommendationCreate schema\n\n"
    "### File: src/core/ai/schemas.py - Internal data models\n"
    "- FinancialContext: the summarized snapshot passed to all pipelines\n"
    "- PipelineResult: wrapper for pipeline output with metadata\n\n"
    "### Scheduler integration (update src/scheduler/jobs.py):\n"
    "```python\n"
    "async def daily_analysis_job(db: AsyncSession):\n"
    "    engine = AIFinancialAdvisorEngine(db)\n"
    "    recommendations = await engine.run_analysis()\n"
    "    for rec in recommendations:\n"
    "        db_rec = Recommendation(**rec.model_dump())\n"
    "        db.add(db_rec)\n"
    "    await db.commit()\n"
    "    urgent = [r for r in recommendations if r.priority >= 8]\n"
    "    if urgent:\n"
    "        await NotificationDispatcher.send(...)\n"
    "```\n\n"
    "## Requirements\n"
    "- All code must be async (SQLAlchemy 2.0 async session)\n"
    "- Use asyncio.gather for parallel pipeline execution\n"
    "- Each pipeline must handle its own errors gracefully\n"
    "- Confidence threshold: only save recommendations with confidence >= 0.7\n"
    "- Deduplicate against existing PENDING recommendations by (type, title similarity)\n"
    "- action_payload must contain machine-readable execution parameters\n"
    "- Include type hints throughout\n\n"
    "Provide the COMPLETE implementation for all files. Every function body filled in. Production-ready code."
)


async def main():
    async with httpx.AsyncClient() as client:
        print("=" * 70)
        print("Iteration 002: AI Financial Advisor Engine")
        print(f"Target: {BASE}/api/gemini/enhance")
        print(f"Prompt: {len(TASK)} chars")
        print(f"Eval context: {len(EVAL_CONTEXT)} chars")
        print("Eval model: flash (with eval_context for reference checking)")
        print("=" * 70)

        start = time.monotonic()
        resp = await client.post(
            f"{BASE}/api/gemini/enhance",
            json={
                "prompt": TASK,
                "enable_enhancement": True,
                "enable_self_eval": True,
                "eval_model": "flash",
                "eval_context": EVAL_CONTEXT,
                "domain_hint": "code",
                "model": "pro-3.1",
                "temperature": 0.3,
                "max_tokens": 65536,
            },
            timeout=600.0,
        )
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            print(f"\nFAILED: {resp.status_code}")
            print(resp.text[:500])
            return

        data = resp.json()
        print(f"\nModel: {data['model']}")
        print(f"Duration: {data['duration_ms']}ms (wall: {elapsed:.1f}s)")
        print(f"Response: {len(data['response'])} chars")

        if data.get("enhancement"):
            e = data["enhancement"]
            print(f"\nEnhancement pipeline:")
            print(f"  Stages: {e['stages_applied']}")
            print(f"  Complexity: {e['complexity_score']:.2f}")
            print(f"  Model routed: {e['model_selected']}")

        if data.get("evaluation"):
            ev = data["evaluation"]
            print(f"\nSelf-evaluation (Flash + eval_context):")
            print(f"  Overall: {ev['score']:.2f}")
            print(f"  Accuracy: {ev['accuracy']:.2f}")
            print(f"  Completeness: {ev['completeness']:.2f}")
            print(f"  Clarity: {ev['clarity']:.2f}")
            print(f"  Actionability: {ev['actionability']:.2f}")
            if ev["issues"]:
                print(f"  Issues:")
                for issue in ev["issues"]:
                    print(f"    - {issue}")
            print(f"  Retries needed: {ev['retry_count']}")

        usage = data.get("usage", {})
        print(f"\nToken usage:")
        print(f"  Prompt: {usage.get('prompt_tokens', 0):,}")
        print(f"  Completion: {usage.get('completion_tokens', 0):,}")
        print(f"  Total: {usage.get('total_tokens', 0):,}")

        # Save full response
        out_path = "/home/umwai/um-agent-coder/evals/gemini-intelligence-layer/iteration-002.md"
        with open(out_path, "w") as f:
            f.write("# Iteration 002 — AI Financial Advisor Engine\n\n")
            f.write("## Changes from Iteration 001\n")
            f.write("- Eval model: Flash with eval_context (was: Flash without context)\n")
            f.write("- Eval context: actual API signatures provided as reference\n")
            f.write("- Same generation model (Pro 3.1) and prompt\n\n")
            f.write("## Pipeline Metadata\n\n")
            f.write(f"- **Model**: {data['model']}\n")
            f.write(f"- **Duration**: {data['duration_ms']}ms\n")
            f.write(f"- **Tokens**: {usage.get('total_tokens', 0):,}\n")

            if data.get("enhancement"):
                e = data["enhancement"]
                f.write(f"- **Complexity**: {e['complexity_score']:.2f}\n")
                f.write(f"- **Stages**: {', '.join(e['stages_applied'])}\n")

            if data.get("evaluation"):
                ev = data["evaluation"]
                f.write(f"\n### Self-Evaluation (Pro 3.1)\n\n")
                f.write(f"| Dimension | Score |\n|---|---|\n")
                f.write(f"| Overall | {ev['score']:.2f} |\n")
                f.write(f"| Accuracy | {ev['accuracy']:.2f} |\n")
                f.write(f"| Completeness | {ev['completeness']:.2f} |\n")
                f.write(f"| Clarity | {ev['clarity']:.2f} |\n")
                f.write(f"| Actionability | {ev['actionability']:.2f} |\n")
                f.write(f"| Retries | {ev['retry_count']} |\n")
                if ev["issues"]:
                    f.write(f"\n**Issues flagged by evaluator:**\n")
                    for issue in ev["issues"]:
                        f.write(f"- {issue}\n")

            f.write(f"\n## Eval Context Provided\n\n```\n{EVAL_CONTEXT}\n```\n\n")
            f.write(f"## Prompt\n\n<details><summary>Full prompt ({len(TASK)} chars)</summary>\n\n")
            f.write(f"```\n{TASK}\n```\n\n</details>\n\n")
            f.write(f"## Response\n\n")
            f.write(data["response"])

        print(f"\nFull output saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
