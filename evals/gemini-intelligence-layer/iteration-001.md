# Gemini Intelligence Layer — Iteration 001

**Date**: 2026-03-06
**Task**: AI Financial Advisor Engine for um-cfo
**Endpoint**: `POST /api/gemini/enhance`
**Target**: `https://um-agent-daemon-23o5bq3bfq-uc.a.run.app`

---

## Pipeline Metadata

| Field | Value |
|-------|-------|
| Model routed | gemini-3.1-pro-preview |
| Complexity score | 0.58 |
| Enhancement stages | chain_of_thought, context_enrichment, constraint_clarification, output_format |
| Self-eval score | 0.70 |
| Retries | 0 |
| Duration | 128,978ms |
| Tokens (prompt) | 1,755 |
| Tokens (completion) | 5,337 |
| Tokens (total) | 17,739 |
| Response length | 20,841 chars |

---

## Prompt

```
You are designing the complete implementation for an AI Financial Advisor Engine
(src/core/ai/) for a personal finance app built with FastAPI + PostgreSQL + SQLAlchemy 2.0 async.

## Existing Codebase (already implemented, do NOT rewrite)

### Models (SQLAlchemy ORM, UUID PKs, created_at/updated_at):
- Account: id, name, account_type (CHECKING/SAVINGS/CREDIT_CARD/BROKERAGE/CRYPTO/LOAN/OTHER), balance, institution, currency, is_active
- Transaction: id, account_id, amount, description, date, category_id, merchant, is_recurring, fingerprint (SHA256 dedup)
- Budget: id, category_id, amount, period (WEEKLY/BIWEEKLY/MONTHLY/QUARTERLY/ANNUAL), start_date, end_date
- Bill: id, name, amount, due_date, frequency, account_id, is_auto_pay, alert_days_before, is_active
- Debt: id, name, balance, apr, minimum_payment, account_id, debt_type
- Investment: id, account_id, symbol, quantity, cost_basis, current_price, asset_type
- Recommendation: id, type (RecommendationType enum), title, description, impact_amount, confidence, priority (1-10), status (PENDING/APPROVED/REJECTED/EXECUTED), action_payload (JSONB), expires_at
- Category: id, name, parent_id (hierarchical)

### RecommendationType enum values:
BUDGET_ADJUSTMENT, DEBT_PAYMENT, INVESTMENT_REBALANCE, BILL_ALERT, SAVINGS_OPPORTUNITY, TAX_OPTIMIZATION, SPENDING_ALERT, CASHFLOW_WARNING, GENERAL_INSIGHT

### Existing engines:
1. CashFlowEngine (src/core/cashflow/__init__.py):
   - project_bills(bills, start, end) -> List[ProjectedBill(date, amount, name)]
   - forecast_daily(accounts, bills, transactions, days=90) -> List[DayForecast(date, projected_balance, bills_due, net_flow)]

2. DebtOptimizer (src/core/debt/optimizer.py):
   - simulate(debts, extra_payment, strategy='avalanche'|'snowball') -> DebtPayoffResult(months, total_interest, schedule[])

3. ApprovalWorkflow (src/core/approval/workflow.py):
   - submit(recommendation) -> sets status=PENDING
   - approve(recommendation_id) -> PENDING->APPROVED
   - execute(recommendation_id) -> APPROVED->EXECUTED
   - reject(recommendation_id, reason) -> PENDING->REJECTED

4. NotificationDispatcher (src/notifications/dispatcher.py):
   - send(channel, title, message, priority) -> sends to Discord/Slack

5. TradingProdBridge (src/integrations/trading_prod.py):
   - get_positions() -> List[Position(symbol, qty, avg_cost, current_price, pnl)]
   - get_summary() -> PortfolioSummary(total_value, day_pnl, positions_count)

### Database pattern:
async with get_db() as db:
    result = await db.execute(select(Account).where(Account.is_active == True))
    accounts = result.scalars().all()

## What to implement

### File: src/core/ai/engine.py - AIFinancialAdvisorEngine class

class AIFinancialAdvisorEngine:
    def __init__(self, db: AsyncSession): ...
    async def gather_context(self) -> FinancialContext: ...
    async def run_analysis(self) -> list[RecommendationCreate]: ...
    async def _deduplicate(self, recs: list[RecommendationCreate]) -> list[RecommendationCreate]: ...

### File: src/core/ai/pipelines/ - 6 analysis pipeline modules
Each pipeline is a class with `async def analyze(self, context: FinancialContext) -> list[RecommendationCreate]`

1. spending_analyzer.py - SpendingAnalyzer
   - Compare current month spending per category vs 90-day average
   - Flag merchants with >20% price increase on recurring charges
   - Detect category drift (spending shifting between categories)
   - Output: SPENDING_ALERT, BUDGET_ADJUSTMENT recommendations

2. cashflow_advisor.py - CashflowAdvisor
   - Use existing CashFlowEngine.forecast_daily()
   - Detect upcoming overdraft risk (balance < $500)
   - Identify large upcoming bills vs available balance
   - Generate natural language narrative
   - Output: CASHFLOW_WARNING, GENERAL_INSIGHT recommendations

3. debt_strategist.py - DebtStrategist
   - Use existing DebtOptimizer.simulate() with both strategies
   - Compare avalanche vs snowball: if savings > $100, recommend switch
   - Detect high-APR debts that should be prioritized
   - Calculate optimal extra payment allocation
   - Output: DEBT_PAYMENT recommendations

4. investment_monitor.py - InvestmentMonitor
   - Use TradingProdBridge.get_positions() for current holdings
   - Detect portfolio concentration (any position > 20% of total)
   - Flag significant unrealized losses (> $1000)
   - Check cash drag (uninvested cash > 10% of portfolio)
   - Output: INVESTMENT_REBALANCE recommendations

5. savings_detector.py - SavingsDetector
   - Identify recurring subscription charges by merchant pattern
   - Flag duplicate services (multiple streaming, multiple cloud storage)
   - Detect idle cash in low-yield checking accounts
   - Calculate potential savings from switching/cancelling
   - Output: SAVINGS_OPPORTUNITY, BILL_ALERT recommendations

6. tax_planner.py - TaxPlanner
   - Identify tax-loss harvesting opportunities (unrealized losses > $1000)
   - Estimate quarterly tax liability from realized gains
   - Flag large deductible expenses
   - Output: TAX_OPTIMIZATION recommendations

### File: src/core/ai/prompts.py - Prompt templates
Each pipeline constructs a structured prompt with:
- Financial summary metrics (NOT raw data)
- Specific analysis questions
- Required JSON output format matching RecommendationCreate schema

### File: src/core/ai/schemas.py - Internal data models
- FinancialContext: the summarized snapshot passed to all pipelines
- PipelineResult: wrapper for pipeline output with metadata

### Scheduler integration (update src/scheduler/jobs.py):
async def daily_analysis_job(db: AsyncSession):
    engine = AIFinancialAdvisorEngine(db)
    recommendations = await engine.run_analysis()
    for rec in recommendations:
        db_rec = Recommendation(**rec.model_dump())
        db.add(db_rec)
    await db.commit()
    urgent = [r for r in recommendations if r.priority >= 8]
    if urgent:
        await NotificationDispatcher.send(...)

## Requirements
- All code must be async (SQLAlchemy 2.0 async session)
- Use asyncio.gather for parallel pipeline execution
- Each pipeline must handle its own errors gracefully
- Confidence threshold: only save recommendations with confidence >= 0.7
- Deduplicate against existing PENDING recommendations by (type, title similarity)
- action_payload must contain machine-readable execution parameters
- Include type hints throughout

Provide the COMPLETE implementation for all files. Every function body filled in. Production-ready code.
```

---

## Response

```json
{
  "src/core/ai/schemas.py": "from typing import List, Dict, Any, Optional\nfrom datetime import datetime\nfrom pydantic import BaseModel, Field\nfrom src.models import Account, Transaction, Budget, Bill, Debt, Investment, Category, RecommendationType\n\nclass FinancialContext(BaseModel):\n    accounts: List[Any]\n    transactions: List[Any]\n    budgets: List[Any]\n    bills: List[Any]\n    debts: List[Any]\n    investments: List[Any]\n    categories: List[Any]\n\n    class Config:\n        arbitrary_types_allowed = True\n\nclass RecommendationCreate(BaseModel):\n    type: RecommendationType\n    title: str\n    description: str\n    impact_amount: float\n    confidence: float = Field(..., ge=0.0, le=1.0)\n    priority: int = Field(..., ge=1, le=10)\n    action_payload: Dict[str, Any]\n    expires_at: Optional[datetime] = None\n\nclass PipelineResult(BaseModel):\n    pipeline_name: str\n    recommendations: List[RecommendationCreate]\n    error: Optional[str] = None\n",
  "src/core/ai/prompts.py": "SPENDING_ANALYZER_PROMPT = \"\"\"\nAnalyze the following spending data:\n{financial_summary}\n\nIdentify:\n1. Categories where current month spending exceeds the 90-day average by >20%.\n2. Recurring merchants with >20% price increases.\n3. Category drift (spending shifting between categories).\n\nOutput JSON matching the RecommendationCreate schema.\n\"\"\"\n\nCASHFLOW_ADVISOR_PROMPT = \"\"\"\nReview the 90-day cashflow forecast:\n{financial_summary}\n\nIdentify:\n1. Periods where the projected balance drops below $500.\n2. Large upcoming bills compared to available balance.\n\nProvide a natural language narrative for the description.\nOutput JSON matching the RecommendationCreate schema.\n\"\"\"\n\nDEBT_STRATEGIST_PROMPT = \"\"\"\nReview the debt profile and payoff simulations:\n{financial_summary}\n\nIdentify:\n1. Whether Avalanche or Snowball strategy saves >$100 in interest.\n2. High-APR debts (>15%) that should be prioritized.\n3. Optimal extra payment allocation.\n\nOutput JSON matching the RecommendationCreate schema.\n\"\"\"\n\nINVESTMENT_MONITOR_PROMPT = \"\"\"\nReview the current investment portfolio:\n{financial_summary}\n\nIdentify:\n1. Portfolio concentration (any position > 20% of total).\n2. Significant unrealized losses (> $1000).\n3. Cash drag (uninvested cash > 10% of portfolio).\n\nOutput JSON matching the RecommendationCreate schema.\n\"\"\"\n\nSAVINGS_DETECTOR_PROMPT = \"\"\"\nReview the recurring transactions and account balances:\n{financial_summary}\n\nIdentify:\n1. Duplicate services (e.g., multiple streaming or cloud storage).\n2. Idle cash in low-yield checking accounts (> $5000).\n3. Potential savings from switching/cancelling.\n\nOutput JSON matching the RecommendationCreate schema.\n\"\"\"\n\nTAX_PLANNER_PROMPT = \"\"\"\nReview the investment positions and recent transactions:\n{financial_summary}\n\nIdentify:\n1. Tax-loss harvesting opportunities (unrealized losses > $1000).\n2. Estimated quarterly tax liability from realized gains.\n3. Large deductible expenses (medical, education, charity).\n\nOutput JSON matching the RecommendationCreate schema.\n\"\"\"\n",
  "src/core/ai/engine.py": "import asyncio\nimport logging\nfrom difflib import SequenceMatcher\nfrom datetime import datetime, timedelta, timezone\nfrom sqlalchemy.ext.asyncio import AsyncSession\nfrom sqlalchemy import select\n\nfrom src.models import Account, Transaction, Budget, Bill, Debt, Investment, Category, Recommendation\nfrom src.core.ai.schemas import FinancialContext, RecommendationCreate\nfrom src.core.ai.pipelines.spending_analyzer import SpendingAnalyzer\nfrom src.core.ai.pipelines.cashflow_advisor import CashflowAdvisor\nfrom src.core.ai.pipelines.debt_strategist import DebtStrategist\nfrom src.core.ai.pipelines.investment_monitor import InvestmentMonitor\nfrom src.core.ai.pipelines.savings_detector import SavingsDetector\nfrom src.core.ai.pipelines.tax_planner import TaxPlanner\n\nlogger = logging.getLogger(__name__)\n\nclass AIFinancialAdvisorEngine:\n    def __init__(self, db: AsyncSession):\n        self.db = db\n        self.pipelines = [\n            SpendingAnalyzer(),\n            CashflowAdvisor(),\n            DebtStrategist(),\n            InvestmentMonitor(),\n            SavingsDetector(),\n            TaxPlanner()\n        ]\n\n    async def gather_context(self) -> FinancialContext:\n        now = datetime.now(timezone.utc)\n        ninety_days_ago = now - timedelta(days=90)\n\n        accounts = (await self.db.execute(select(Account).where(Account.is_active == True))).scalars().all()\n        transactions = (await self.db.execute(select(Transaction).where(Transaction.date >= ninety_days_ago))).scalars().all()\n        budgets = (await self.db.execute(select(Budget))).scalars().all()\n        bills = (await self.db.execute(select(Bill).where(Bill.is_active == True))).scalars().all()\n        debts = (await self.db.execute(select(Debt))).scalars().all()\n        investments = (await self.db.execute(select(Investment))).scalars().all()\n        categories = (await self.db.execute(select(Category))).scalars().all()\n\n        return FinancialContext(\n            accounts=list(accounts),\n            transactions=list(transactions),\n            budgets=list(budgets),\n            bills=list(bills),\n            debts=list(debts),\n            investments=list(investments),\n            categories=list(categories)\n        )\n\n    async def _deduplicate(self, recs: list[RecommendationCreate]) -> list[RecommendationCreate]:\n        existing_recs = (await self.db.execute(\n            select(Recommendation).where(Recommendation.status == \"PENDING\")\n        )).scalars().all()\n\n        unique_recs = []\n        for rec in recs:\n            is_duplicate = False\n            for ex in existing_recs:\n                if ex.type == rec.type:\n                    similarity = SequenceMatcher(None, ex.title.lower(), rec.title.lower()).ratio()\n                    if similarity > 0.8:\n                        is_duplicate = True\n                        break\n            if not is_duplicate:\n                unique_recs.append(rec)\n\n        return unique_recs\n\n    async def run_analysis(self) -> list[RecommendationCreate]:\n        context = await self.gather_context()\n\n        tasks = [pipeline.analyze(context) for pipeline in self.pipelines]\n        results = await asyncio.gather(*tasks, return_exceptions=True)\n\n        all_recommendations = []\n        for i, res in enumerate(results):\n            if isinstance(res, Exception):\n                logger.error(f\"Pipeline {self.pipelines[i].__class__.__name__} failed: {res}\")\n                continue\n            \n            valid_recs = [r for r in res if r.confidence >= 0.7]\n            all_recommendations.extend(valid_recs)\n\n        return await self._deduplicate(all_recommendations)\n",
  "src/core/ai/pipelines/spending_analyzer.py": "from datetime import datetime, timedelta, timezone\nfrom collections import defaultdict\nfrom src.core.ai.schemas import FinancialContext, RecommendationCreate\nfrom src.models import RecommendationType\n\nclass SpendingAnalyzer:\n    async def analyze(self, context: FinancialContext) -> list[RecommendationCreate]:\n        recs = []\n        now = datetime.now(timezone.utc)\n        thirty_days_ago = now - timedelta(days=30)\n        \n        current_month = defaultdict(float)\n        past_two_months = defaultdict(float)\n        merchant_txs = defaultdict(list)\n        \n        for t in context.transactions:\n            t_date = t.date if isinstance(t.date, datetime) else datetime.combine(t.date, datetime.min.time()).replace(tzinfo=timezone.utc)\n            \n            if t.amount > 0:\n                if t_date >= thirty_days_ago:\n                    current_month[t.category_id] += t.amount\n                else:\n                    past_two_months[t.category_id] += t.amount\n                    \n            if t.is_recurring and t.merchant:\n                merchant_txs[t.merchant].append(t)\n                \n        for cat_id, current_spend in current_month.items():\n            avg_past = past_two_months[cat_id] / 2.0 if past_two_months[cat_id] > 0 else 0\n            if avg_past > 0 and current_spend > avg_past * 1.2:\n                recs.append(RecommendationCreate(\n                    type=RecommendationType.SPENDING_ALERT,\n                    title=\"High Spending Alert\",\n                    description=f\"Spending in category {cat_id} is 20% higher than your 90-day average.\",\n                    impact_amount=current_spend - avg_past,\n                    confidence=0.85,\n                    priority=7,\n                    action_payload={\"category_id\": str(cat_id), \"excess_amount\": current_spend - avg_past}\n                ))\n                \n        for merchant, txs in merchant_txs.items():\n            if len(txs) >= 2:\n                txs.sort(key=lambda x: x.date)\n                latest = txs[-1].amount\n                previous = txs[-2].amount\n                if previous > 0 and latest > previous * 1.2:\n                    recs.append(RecommendationCreate(\n                        type=RecommendationType.BUDGET_ADJUSTMENT,\n                        title=f\"Price Increase: {merchant}\",\n                        description=f\"Your recurring charge for {merchant} increased by over 20%.\",\n                        impact_amount=latest - previous,\n                        confidence=0.9,\n                        priority=6,\n                        action_payload={\"merchant\": merchant, \"old_amount\": previous, \"new_amount\": latest}\n                    ))\n                    \n        return recs\n",
  "src/core/ai/pipelines/cashflow_advisor.py": "from src.core.ai.schemas import FinancialContext, RecommendationCreate\nfrom src.models import RecommendationType\nfrom src.core.cashflow import CashFlowEngine\n\nclass CashflowAdvisor:\n    async def analyze(self, context: FinancialContext) -> list[RecommendationCreate]:\n        recs = []\n        if not context.accounts:\n            return recs\n        \n        forecasts = CashFlowEngine.forecast_daily(context.accounts, context.bills, context.transactions, days=30)\n        \n        min_forecast = min(forecasts, key=lambda x: x.projected_balance, default=None)\n        if min_forecast and min_forecast.projected_balance < 500:\n            recs.append(RecommendationCreate(\n                type=RecommendationType.CASHFLOW_WARNING,\n                title=\"Overdraft Risk Detected\",\n                description=f\"Your balance is projected to drop to ${min_forecast.projected_balance:.2f} on {min_forecast.date}. Ensure you have sufficient funds for upcoming bills.\",\n                impact_amount=500 - min_forecast.projected_balance,\n                confidence=0.9,\n                priority=9,\n                action_payload={\"date\": str(min_forecast.date), \"projected_balance\": min_forecast.projected_balance}\n            ))\n            \n        return recs\n",
  "src/core/ai/pipelines/debt_strategist.py": "from src.core.ai.schemas import FinancialContext, RecommendationCreate\nfrom src.models import RecommendationType\nfrom src.core.debt.optimizer import DebtOptimizer\n\nclass DebtStrategist:\n    async def analyze(self, context: FinancialContext) -> list[RecommendationCreate]:\n        recs = []\n        if not context.debts:\n            return recs\n        \n        avalanche = DebtOptimizer.simulate(context.debts, 0.0, strategy='avalanche')\n        snowball = DebtOptimizer.simulate(context.debts, 0.0, strategy='snowball')\n        \n        savings = snowball.total_interest - avalanche.total_interest\n        if savings > 100:\n            recs.append(RecommendationCreate(\n                type=RecommendationType.DEBT_PAYMENT,\n                title=\"Optimize Debt Payoff Strategy\",\n                description=f\"Switching to the Avalanche payoff strategy could save you ${savings:.2f} in interest over the life of your loans.\",\n                impact_amount=savings,\n                confidence=0.95,\n                priority=8,\n                action_payload={\"recommended_strategy\": \"avalanche\", \"projected_savings\": savings}\n            ))\n            \n        for debt in context.debts:\n            if debt.apr > 15.0:\n                recs.append(RecommendationCreate(\n                    type=RecommendationType.DEBT_PAYMENT,\n                    title=f\"High Interest Debt: {debt.name}\",\n                    description=f\"Consider allocating extra funds to pay down {debt.name} due to its high APR of {debt.apr}%.\",\n                    impact_amount=debt.balance,\n                    confidence=0.9,\n                    priority=7,\n                    action_payload={\"debt_id\": str(debt.id), \"apr\": debt.apr}\n                ))\n                \n        return recs\n",
  "src/core/ai/pipelines/investment_monitor.py": "from src.core.ai.schemas import FinancialContext, RecommendationCreate\nfrom src.models import RecommendationType\nfrom src.integrations.trading_prod import TradingProdBridge\n\nclass InvestmentMonitor:\n    async def analyze(self, context: FinancialContext) -> list[RecommendationCreate]:\n        recs = []\n        positions = TradingProdBridge.get_positions()\n        summary = TradingProdBridge.get_summary()\n        \n        if summary.total_value == 0:\n            return recs\n            \n        for pos in positions:\n            pos_value = pos.qty * pos.current_price\n            if pos_value > 0.2 * summary.total_value:\n                recs.append(RecommendationCreate(\n                    type=RecommendationType.INVESTMENT_REBALANCE,\n                    title=f\"High Concentration in {pos.symbol}\",\n                    description=f\"{pos.symbol} makes up more than 20% of your portfolio. Consider rebalancing to reduce risk.\",\n                    impact_amount=pos_value - (0.2 * summary.total_value),\n                    confidence=0.85,\n                    priority=6,\n                    action_payload={\"symbol\": pos.symbol, \"concentration_pct\": pos_value / summary.total_value}\n                ))\n            \n            if pos.pnl < -1000:\n                recs.append(RecommendationCreate(\n                    type=RecommendationType.INVESTMENT_REBALANCE,\n                    title=f\"Significant Loss in {pos.symbol}\",\n                    description=f\"You have an unrealized loss of ${abs(pos.pnl):.2f} in {pos.symbol}. Review this position.\",\n                    impact_amount=abs(pos.pnl),\n                    confidence=0.8,\n                    priority=5,\n                    action_payload={\"symbol\": pos.symbol, \"unrealized_loss\": pos.pnl}\n                ))\n        \n        invested_value = sum(p.qty * p.current_price for p in positions)\n        cash = summary.total_value - invested_value\n        if cash > 0.1 * summary.total_value:\n            recs.append(RecommendationCreate(\n                type=RecommendationType.INVESTMENT_REBALANCE,\n                title=\"High Uninvested Cash\",\n                description=\"More than 10% of your portfolio is in cash. Consider investing it to avoid cash drag.\",\n                impact_amount=cash,\n                confidence=0.9,\n                priority=6,\n                action_payload={\"cash_amount\": cash, \"cash_pct\": cash / summary.total_value}\n            ))\n            \n        return recs\n",
  "src/core/ai/pipelines/savings_detector.py": "from src.core.ai.schemas import FinancialContext, RecommendationCreate\nfrom src.models import RecommendationType\n\nclass SavingsDetector:\n    async def analyze(self, context: FinancialContext) -> list[RecommendationCreate]:\n        recs = []\n        \n        for acc in context.accounts:\n            if acc.account_type == 'CHECKING' and acc.balance > 5000:\n                excess = acc.balance - 5000\n                recs.append(RecommendationCreate(\n                    type=RecommendationType.SAVINGS_OPPORTUNITY,\n                    title=\"Idle Cash Detected\",\n                    description=f\"You have ${excess:.2f} in excess cash in your checking account. Consider moving it to a high-yield savings account.\",\n                    impact_amount=excess,\n                    confidence=0.9,\n                    priority=7,\n                    action_payload={\"account_id\": str(acc.id), \"excess_cash\": excess}\n                ))\n        \n        streaming_services = ['netflix', 'hulu', 'max', 'disney+', 'prime video', 'spotify', 'apple music']\n        found_streaming = set()\n        for t in context.transactions:\n            if t.is_recurring and t.merchant:\n                merchant_lower = t.merchant.lower()\n                for service in streaming_services:\n                    if service in merchant_lower:\n                        found_streaming.add(service)\n        \n        if len(found_streaming) > 1:\n            recs.append(RecommendationCreate(\n                type=RecommendationType.BILL_ALERT,\n                title=\"Multiple Streaming Subscriptions\",\n                description=f\"You are paying for multiple streaming services ({', '.join(found_streaming)}). Consider pausing one to save money.\",\n                impact_amount=15.0 * (len(found_streaming) - 1),\n                confidence=0.75,\n                priority=4,\n                action_payload={\"services\": list(found_streaming)}\n            ))\n            \n        return recs\n",
  "src/core/ai/pipelines/tax_planner.py": "from src.core.ai.schemas import FinancialContext, RecommendationCreate\nfrom src.models import RecommendationType\nfrom src.integrations.trading_prod import TradingProdBridge\n\nclass TaxPlanner:\n    async def analyze(self, context: FinancialContext) -> list[RecommendationCreate]:\n        recs = []\n        positions = TradingProdBridge.get_positions()\n        \n        harvestable_losses = sum(abs(p.pnl) for p in positions if p.pnl < -1000)\n        if harvestable_losses > 0:\n            recs.append(RecommendationCreate(\n                type=RecommendationType.TAX_OPTIMIZATION,\n                title=\"Tax-Loss Harvesting Opportunity\",\n                description=f\"You have ${harvestable_losses:.2f} in unrealized losses that could be harvested to offset capital gains.\",\n                impact_amount=harvestable_losses,\n                confidence=0.85,\n                priority=7,\n                action_payload={\"harvestable_losses\": harvestable_losses}\n            ))\n            \n        deductible_spend = 0\n        for t in context.transactions:\n            desc = (t.description or \"\").lower()\n            merchant = (t.merchant or \"\").lower()\n            if 'hospital' in desc or 'university' in desc or 'donation' in desc or 'charity' in merchant:\n                deductible_spend += t.amount\n                \n        if deductible_spend > 2000:\n            recs.append(RecommendationCreate(\n                type=RecommendationType.TAX_OPTIMIZATION,\n                title=\"Large Deductible Expenses\",\n                description=f\"You have over ${deductible_spend:.2f} in potentially tax-deductible expenses. Ensure you keep receipts for tax season.\",\n                impact_amount=deductible_spend * 0.2,\n                confidence=0.7,\n                priority=5,\n                action_payload={\"deductible_spend\": deductible_spend}\n            ))\n            \n        return recs\n",
  "src/scheduler/jobs.py": "import logging\nfrom sqlalchemy.ext.asyncio import AsyncSession\nfrom src.core.ai.engine import AIFinancialAdvisorEngine\nfrom src.models import Recommendation\nfrom src.notifications.dispatcher import NotificationDispatcher\n\nlogger = logging.getLogger(__name__)\n\nasync def daily_analysis_job(db: AsyncSession):\n    logger.info(\"Starting daily AI financial analysis job...\")\n    engine = AIFinancialAdvisorEngine(db)\n    \n    try:\n        recommendations = await engine.run_analysis()\n        \n        for rec in recommendations:\n            db_rec = Recommendation(**rec.model_dump())\n            db.add(db_rec)\n        \n        await db.commit()\n        logger.info(f\"Saved {len(recommendations)} new recommendations.\")\n        \n        urgent = [r for r in recommendations if r.priority >= 8]\n        for u in urgent:\n            await NotificationDispatcher.send(\n                channel=\"alerts\",\n                title=u.title,\n                message=u.description,\n                priority=u.priority\n            )\n    except Exception as e:\n        logger.error(f\"Error in daily_analysis_job: {e}\")\n        await db.rollback()\n"
}
```

---

## Evaluation

### Scores

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Architecture | A | Correct shape: engine orchestrator, parallel pipelines, confidence filtering, dedup |
| API compatibility | F | 5/5 external APIs called with wrong names/signatures |
| Field names | F | Uses `type`/`description` but actual model has `recommendation_type`/`summary` |
| Completeness | C | Missing LLM integration (prompts.py is dead code), missing __init__.py files |
| Code quality | B | Clean, typed, well-structured Python |
| Usability as-is | D | Would not run without ~20 fixes |
| Usability as scaffold | B+ | Good starting skeleton to build from |

### Bugs (would crash at runtime)

1. **Recommendation field names wrong** — Model uses `recommendation_type` and `summary`, not `type` and `description`. Every pipeline has this bug.

2. **CashFlowEngine API wrong** — Called as `CashFlowEngine.forecast_daily(accounts, bills, transactions, days=30)`. Actual code: standalone function `forecast_cashflow()` with signature `(accounts, bills, transactions, start_date, end_date)`. Return type `DailyForecastRow` has no `projected_balance` field.

3. **DebtOptimizer API wrong** — Called as `DebtOptimizer.simulate(debts, 0.0, strategy)`. Actual code: standalone function `simulate_payoff(debts: list[DebtItem], monthly_budget, strategy)`. SQLAlchemy `Debt` objects not converted to `DebtItem` dataclasses.

4. **TradingProdBridge API wrong** — Called as class methods `TradingProdBridge.get_positions()` / `.get_summary()`. Actual code: standalone async functions `get_positions()` / `get_portfolio_summary()` returning raw `list[dict]` / `dict`, not typed objects. Accessing `.symbol`, `.pnl` on dicts would crash.

5. **NotificationDispatcher API wrong** — Called as `NotificationDispatcher.send(channel, title, message, priority)`. Actual function: `dispatch(subject, body, channels)`.

### Missing pieces

1. **No actual LLM calls** — `prompts.py` defines templates but no pipeline ever calls an LLM. The "AI" part is entirely absent. All pipelines are pure heuristic rules.

2. **`priority` semantics inverted** — Actual model: lower = higher priority. Gemini assigns `priority=9` for urgent (treating higher = more urgent).

3. **No `__init__.py` files** — Missing for `src/core/ai/` and `src/core/ai/pipelines/`.

4. **No ORM-to-dataclass conversion** — `simulate_payoff()` expects `DebtItem` dataclasses, not SQLAlchemy models.

5. **No error boundaries in pipelines** — `investment_monitor` and `tax_planner` call external HTTP APIs without try/except.

6. **`spending_analyzer` math off** — Divides by 2.0 for "90-day average" but data is 60 days (90 minus current 30).

### What it got right

- Engine orchestrator pattern (gather → parallel pipelines → filter → deduplicate)
- `asyncio.gather(*tasks, return_exceptions=True)` usage
- `SequenceMatcher` dedup on (type, title) at 0.8 threshold
- `action_payload` fields are meaningful and structured
- Confidence threshold filtering at 0.7
- Early returns on empty data

### Root cause analysis

The prompt provided API signatures but Gemini hallucinated different ones. The prompt said:
- `forecast_daily(accounts, bills, transactions, days=90)` — Gemini used `CashFlowEngine.forecast_daily()` (class method, close but wrong)
- `simulate(debts, extra_payment, strategy)` — Gemini used `DebtOptimizer.simulate()` (class method, wrong)
- `send(channel, title, message, priority)` — Gemini used `NotificationDispatcher.send()` (kept the hallucinated API from prompt)

The prompt itself had inaccurate API signatures (intentionally — they were simplified from reality). Gemini faithfully used those inaccurate signatures but added class prefixes that don't exist. This is a **prompt quality issue** as much as a model issue.

### Recommendations for iteration 002

1. **Fix the prompt** — Provide exact function signatures from the actual source code, not simplified versions
2. **Include actual source snippets** — Paste the real function headers so Gemini can't hallucinate alternatives
3. **Require LLM integration** — Explicitly demand that pipelines call an LLM and parse JSON responses
4. **Specify priority semantics** — State "lower number = higher priority" explicitly
5. **Add a code validation step** — After generation, parse the Python AST and check imports resolve
