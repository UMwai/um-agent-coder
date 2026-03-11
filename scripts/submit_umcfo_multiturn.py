#!/usr/bin/env python3
"""Submit the AI Financial Advisor Engine task via multi-turn session.

When the model stops mid-output, sends "continue" to get the rest.
Concatenates all turns into the final response.
"""

import asyncio
import time

import httpx

BASE = "https://um-agent-daemon-23o5bq3bfq-uc.a.run.app"

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
    "priority (1-10, where 10=highest urgency), status (PENDING/APPROVED/REJECTED/EXECUTED), action_payload (JSONB), expires_at\n"
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
    "- Include type hints throughout\n"
    "- Each pipeline MUST call an LLM (via a generate function) to analyze the data — do NOT just use rule-based logic\n\n"
    "Provide the COMPLETE implementation for ALL files listed above. Every function body filled in. "
    "Production-ready code. Do NOT stop until every single file is complete. "
    "There are 10+ files to produce — you must output all of them."
)

# Files we expect in the output
EXPECTED_FILES = [
    "schemas.py",
    "prompts.py",
    "engine.py",
    "spending_analyzer.py",
    "cashflow_advisor.py",
    "debt_strategist.py",
    "investment_monitor.py",
    "savings_detector.py",
    "tax_planner.py",
    "jobs.py",
]

CONTINUE_PROMPTS = [
    "You stopped mid-output. Continue EXACTLY where you left off — do not repeat any code already produced. Keep going until every file is complete.",
    "Continue. Do not repeat previous files. Output the remaining files that haven't been shown yet.",
    "Keep going. Output any remaining files not yet produced.",
    "Continue outputting the remaining code files.",
]


def check_completeness(full_text: str) -> list[str]:
    """Return list of expected files NOT found in the output."""
    missing = []
    for f in EXPECTED_FILES:
        if f not in full_text:
            missing.append(f)
    return missing


async def main():
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        print("=" * 70)
        print("Multi-turn: AI Financial Advisor Engine")
        print(f"Target: {BASE}")
        print("=" * 70)

        # 1. Create session
        print("\n[1] Creating session...")
        resp = await client.post(
            f"{BASE}/api/gemini/sessions",
            json={
                "model": "pro-3.1",
                "temperature": 0.3,
                "max_tokens": 65536,
                "system_prompt": (
                    "You are a senior Python developer. When asked to produce code, "
                    "output COMPLETE file implementations with no placeholders or ellipses. "
                    "Never stop mid-file. If a response would be very long, that's fine — "
                    "keep going until every file is complete."
                ),
            },
        )
        if resp.status_code != 200:
            print(f"Failed to create session: {resp.status_code} {resp.text[:200]}")
            return

        session = resp.json()
        session_id = session["id"]
        print(f"  Session: {session_id}")

        # 2. Send initial prompt
        all_responses = []
        total_tokens = 0
        start = time.monotonic()

        print(f"\n[2] Sending initial prompt ({len(TASK)} chars)...")
        resp = await client.post(
            f"{BASE}/api/gemini/sessions/{session_id}/message",
            json={"content": TASK, "enable_enhancement": False},
        )
        if resp.status_code != 200:
            print(f"Failed: {resp.status_code} {resp.text[:300]}")
            return

        msg = resp.json()
        response_text = msg["content"]
        tokens = msg.get("token_count", 0)
        total_tokens += tokens
        all_responses.append(response_text)
        print(f"  Turn 1: {len(response_text)} chars, {tokens} tokens")

        # 3. Check completeness and continue
        full_text = "\n".join(all_responses)
        missing = check_completeness(full_text)
        turn = 1

        while missing and turn < len(CONTINUE_PROMPTS) + 1:
            print(f"\n  Missing files: {missing}")
            continue_prompt = CONTINUE_PROMPTS[min(turn - 1, len(CONTINUE_PROMPTS) - 1)]
            continue_prompt += f"\n\nStill needed: {', '.join(missing)}"

            turn += 1
            print(f"\n[{turn + 1}] Sending continuation...")
            resp = await client.post(
                f"{BASE}/api/gemini/sessions/{session_id}/message",
                json={"content": continue_prompt, "enable_enhancement": False},
            )
            if resp.status_code != 200:
                print(f"Failed: {resp.status_code} {resp.text[:300]}")
                break

            msg = resp.json()
            response_text = msg["content"]
            tokens = msg.get("token_count", 0)
            total_tokens += tokens
            all_responses.append(response_text)
            print(f"  Turn {turn}: {len(response_text)} chars, {tokens} tokens")

            full_text = "\n".join(all_responses)
            missing = check_completeness(full_text)

        elapsed = time.monotonic() - start

        # 4. Report
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Turns: {turn}")
        print(f"Total chars: {len(full_text)}")
        print(f"Total tokens: {total_tokens}")
        print(f"Wall time: {elapsed:.1f}s")

        final_missing = check_completeness(full_text)
        found = [f for f in EXPECTED_FILES if f not in final_missing]
        print(f"Files found: {len(found)}/{len(EXPECTED_FILES)}")
        if found:
            print(f"  Present: {', '.join(found)}")
        if final_missing:
            print(f"  Missing: {', '.join(final_missing)}")

        # 5. Save
        out_path = "/home/umwai/um-agent-coder/evals/gemini-intelligence-layer/iteration-003-multiturn.md"
        with open(out_path, "w") as f:
            f.write("# Iteration 003 — Multi-Turn AI Financial Advisor Engine\n\n")
            f.write("## Approach\n")
            f.write("- Used session-based multi-turn conversation\n")
            f.write("- Model: Pro 3.1, temperature 0.3\n")
            f.write("- Sent continuation prompts when output was incomplete\n\n")
            f.write(f"## Stats\n\n")
            f.write(f"- **Turns**: {turn}\n")
            f.write(f"- **Total chars**: {len(full_text):,}\n")
            f.write(f"- **Total tokens**: {total_tokens:,}\n")
            f.write(f"- **Wall time**: {elapsed:.1f}s\n")
            f.write(f"- **Files found**: {len(found)}/{len(EXPECTED_FILES)}\n")
            if final_missing:
                f.write(f"- **Missing**: {', '.join(final_missing)}\n")
            f.write(f"\n## Response (all turns concatenated)\n\n")
            for i, resp_text in enumerate(all_responses):
                f.write(f"### Turn {i + 1}\n\n")
                f.write(resp_text)
                f.write("\n\n---\n\n")

        print(f"\nSaved: {out_path}")

        # 6. Cleanup session
        await client.delete(f"{BASE}/api/gemini/sessions/{session_id}")
        print("Session deleted.")


if __name__ == "__main__":
    asyncio.run(main())
