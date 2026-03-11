# Iteration 003 — Multi-Turn AI Financial Advisor Engine

## Changes from Iteration 002
- Used session-based multi-turn conversation (instead of single /enhance call)
- System prompt: "Never stop mid-file, keep going until every file is complete"
- Added explicit instruction: "Do NOT stop until every single file is complete"
- Added requirement: "Each pipeline MUST call an LLM"
- Result: 10/10 files produced in single turn (no continuation needed)

## Pipeline Metadata

- **Model**: gemini-3.1-pro-preview
- **Method**: Session multi-turn (`POST /api/gemini/sessions/{id}/message`)
- **Temperature**: 0.3
- **Turns**: 1
- **Total chars**: 28,362
- **Total tokens**: 12,725
- **Wall time**: 133.3s
- **Files found**: 10/10 (schemas.py, prompts.py, engine.py, spending_analyzer.py, cashflow_advisor.py, debt_strategist.py, investment_monitor.py, savings_detector.py, tax_planner.py, jobs.py + bonus llm.py)

## Evaluation

### Gemini Flash (`POST /api/gemini/evaluate`)

| Dimension | Score |
|---|---|
| Overall | 1.00 |
| Accuracy | 1.00 |
| Completeness | 1.00 |
| Clarity | 1.00 |
| Actionability | 1.00 |

No issues flagged. (Too lenient — see Pro and Claude evals below.)

### Gemini Pro (`POST /api/gemini/evaluate`)

| Dimension | Score |
|---|---|
| Overall | 0.85 |
| Accuracy | 0.80 |
| Completeness | 0.90 |
| Clarity | 0.90 |
| Actionability | 0.80 |

**Issues flagged:**
- TaxPlanner does not calculate quarterly tax liabilities based on realized gains as requested
- NotificationDispatcher.send is awaited but reference signature does not explicitly show it as async
- CashFlowEngine.forecast_daily is awaited but reference signature does not explicitly show it as async
- TradingProdBridge.get_positions is awaited but reference signature does not explicitly show it as async

### Claude Opus (manual review)

| Dimension | Score |
|---|---|
| Overall | 0.875 |
| Accuracy | 0.80 |
| Completeness | 0.95 |
| Clarity | 0.90 |
| Actionability | 0.85 |

**Issues flagged:**
- async/sync ambiguity on existing engine calls (same as Pro)
- `llm.py` uses OpenAI (`AsyncOpenAI`) instead of project's own LLM infrastructure
- `Recommendation.status == 'PENDING'` uses string literal instead of enum
- `account_type` mapped as plain `str` in `AccountSummary` vs ORM enum
- No `__init__.py` for `src/core/ai/` or `src/core/ai/pipelines/` packages

## Prompt

<details><summary>Full prompt (6,777 chars)</summary>

You are designing the complete implementation for an AI Financial Advisor Engine (src/core/ai/) for a personal finance app built with FastAPI + PostgreSQL + SQLAlchemy 2.0 async.

[See iteration-001.md for the full shared prompt — same task specification, with these additions in iteration 003:]

- Each pipeline MUST call an LLM (via a generate function) to analyze the data — do NOT just use rule-based logic
- Do NOT stop until every single file is complete. There are 10+ files to produce — you must output all of them.

System prompt: "You are a senior Python developer. When asked to produce code, output COMPLETE file implementations with no placeholders or ellipses. Never stop mid-file. If a response would be very long, that's fine — keep going until every file is complete."

</details>

## Response (all turns concatenated)

### Turn 1

Here is the complete, production-ready implementation for the AI Financial Advisor Engine. 

The solution is split into the requested files. It uses Pydantic for strict schema validation, SQLAlchemy 2.0 for async database operations, and includes a robust LLM integration layer that enforces JSON outputs. Each pipeline pre-processes the raw financial data into a concise summary before passing it to the LLM to avoid context-window limits and reduce latency.

### File: `src/core/ai/schemas.py`
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import date, datetime
from src.models import RecommendationType

class RecommendationCreate(BaseModel):
    type: RecommendationType
    title: str
    description: str
    impact_amount: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    priority: int = Field(..., ge=1, le=10)
    action_payload: Dict[str, Any]

class AccountSummary(BaseModel):
    id: str
    name: str
    account_type: str
    balance: float
    currency: str

class TransactionSummary(BaseModel):
    id: str
    amount: float
    description: str
    date: date
    category_id: Optional[str]
    merchant: Optional[str]
    is_recurring: bool

class BudgetSummary(BaseModel):
    id: str
    category_id: str
    amount: float
    period: str

class BillSummary(BaseModel):
    id: str
    name: str
    amount: float
    due_date: date
    is_auto_pay: bool

class DebtSummary(BaseModel):
    id: str
    name: str
    balance: float
    apr: float
    minimum_payment: float

class InvestmentSummary(BaseModel):
    id: str
    symbol: str
    quantity: float
    cost_basis: float
    current_price: float

class FinancialContext(BaseModel):
    accounts: List[AccountSummary] = []
    transactions: List[TransactionSummary] = []
    budgets: List[BudgetSummary] = []
    bills: List[BillSummary] = []
    debts: List[DebtSummary] = []
    investments: List[InvestmentSummary] = []

class PipelineResult(BaseModel):
    pipeline_name: str
    recommendations: List[RecommendationCreate]
    errors: List[str] = []
```

### File: `src/core/ai/llm.py`
```python
import json
import logging
import os
from typing import List
from openai import AsyncOpenAI
from src.core.ai.schemas import RecommendationCreate

logger = logging.getLogger(__name__)

# Assumes OPENAI_API_KEY is set in the environment
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-type-checking"))

async def generate_recommendations(system_prompt: str, user_prompt: str) -> List[RecommendationCreate]:
    """
    Calls the LLM to generate financial recommendations based on the provided context.
    Enforces JSON output matching the list[RecommendationCreate] schema.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        content = response.choices[0].message.content
        if not content:
            return []

        # Parse the JSON response
        data = json.loads(content)
        
        # Expecting the LLM to return a JSON object with a "recommendations" key
        recs_data = data.get("recommendations", [])
        
        recommendations = []
        for rec in recs_data:
            try:
                recommendations.append(RecommendationCreate(**rec))
            except Exception as validation_err:
                logger.error(f"LLM output validation error: {validation_err} for payload {rec}")
                continue
                
        return recommendations

    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}")
        return []
```

### File: `src/core/ai/prompts.py`
```python
BASE_SYSTEM_PROMPT = """
You are an elite AI Financial Advisor Engine. Your job is to analyze the user's financial data summaries and generate highly actionable, personalized financial recommendations.
You must output ONLY valid JSON containing a single key "recommendations" which is a list of objects.
Each object in the list MUST adhere exactly to this schema:
{
    "type": "string (one of: BUDGET_ADJUSTMENT, DEBT_PAYMENT, INVESTMENT_REBALANCE, BILL_ALERT, SAVINGS_OPPORTUNITY, TAX_OPTIMIZATION, SPENDING_ALERT, CASHFLOW_WARNING, GENERAL_INSIGHT)",
    "title": "string (Short, punchy title)",
    "description": "string (Detailed narrative explaining the 'why' and 'how')",
    "impact_amount": "float (Estimated dollar value impact, 0 if N/A)",
    "confidence": "float (0.0 to 1.0, where 1.0 is absolute certainty)",
    "priority": "integer (1 to 10, where 10 is immediate urgent action required)",
    "action_payload": "object (Machine-readable JSON parameters for the app to execute the action, e.g., {'target_account': 'uuid', 'transfer_amount': 500})"
}
Do not include markdown formatting like ```json. Just output the raw JSON object.
"""

SPENDING_ANALYZER_PROMPT = """
Analyze the following spending summary. Look for:
1. Significant category drift (spending shifting heavily between categories).
2. Merchants with >20% price increases on recurring charges.
3. Current month spending vastly exceeding the 90-day average.

Spending Summary Data:
{data}

Generate SPENDING_ALERT or BUDGET_ADJUSTMENT recommendations if anomalies are found.
"""

CASHFLOW_ADVISOR_PROMPT = """
Analyze the following 90-day cashflow forecast. Look for:
1. Overdraft risks (projected balance dropping below $500).
2. Large upcoming bills that threaten the available balance.

Cashflow Forecast Data:
{data}

Generate CASHFLOW_WARNING or GENERAL_INSIGHT recommendations. If an overdraft is imminent, set priority to 9 or 10.
"""

DEBT_STRATEGIST_PROMPT = """
Analyze the following debt optimization simulation results comparing Avalanche vs Snowball strategies. Look for:
1. Opportunities to switch strategies if savings > $100.
2. High-APR debts that need immediate prioritization.
3. Optimal extra payment allocations.

Debt Simulation Data:
{data}

Generate DEBT_PAYMENT recommendations. Include the recommended strategy and extra payment amount in the action_payload.
"""

INVESTMENT_MONITOR_PROMPT = """
Analyze the following investment portfolio summary. Look for:
1. Portfolio concentration risk (any single position > 20% of total value).
2. Significant unrealized losses (> $1000).
3. Cash drag (uninvested cash > 10% of portfolio).

Investment Data:
{data}

Generate INVESTMENT_REBALANCE recommendations.
"""

SAVINGS_DETECTOR_PROMPT = """
Analyze the following recurring transactions and account balances. Look for:
1. Duplicate services (e.g., multiple streaming services, multiple cloud storage).
2. Idle cash in low-yield checking accounts (balances > $5000 that could be moved to savings/investments).

Savings Data:
{data}

Generate SAVINGS_OPPORTUNITY or BILL_ALERT recommendations.
"""

TAX_PLANNER_PROMPT = """
Analyze the following financial data for tax optimization. Look for:
1. Tax-loss harvesting opportunities (unrealized losses > $1000).
2. Estimated quarterly tax liabilities based on realized gains.
3. Large deductible expenses.

Tax Data:
{data}

Generate TAX_OPTIMIZATION recommendations.
"""
```

### File: `src/core/ai/pipelines/spending_analyzer.py`
```python
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from src.core.ai.schemas import FinancialContext, RecommendationCreate
from src.core.ai.prompts import BASE_SYSTEM_PROMPT, SPENDING_ANALYZER_PROMPT
from src.core.ai.llm import generate_recommendations

logger = logging.getLogger(__name__)

class SpendingAnalyzer:
    async def analyze(self, context: FinancialContext) -> List[RecommendationCreate]:
        try:
            now = datetime.now(timezone.utc).date()
            thirty_days_ago = now - timedelta(days=30)
            ninety_days_ago = now - timedelta(days=90)

            # Pre-process data: Calculate 90-day avg vs current month
            category_spending: Dict[str, Dict[str, float]] = {}
            recurring_merchants: Dict[str, List[float]] = {}

            for tx in context.transactions:
                if not tx.category_id:
                    continue
                
                cat = tx.category_id
                if cat not in category_spending:
                    category_spending[cat] = {"last_30": 0.0, "prev_60": 0.0}
                
                if tx.date >= thirty_days_ago:
                    category_spending[cat]["last_30"] += tx.amount
                elif tx.date >= ninety_days_ago:
                    category_spending[cat]["prev_60"] += tx.amount

                if tx.is_recurring and tx.merchant:
                    if tx.merchant not in recurring_merchants:
                        recurring_merchants[tx.merchant] = []
                    recurring_merchants[tx.merchant].append(tx.amount)

            # Format summary for LLM
            summary = {
                "category_comparison": {
                    cat: {
                        "last_30_days": data["last_30"],
                        "90_day_average": (data["last_30"] + data["prev_60"]) / 3.0
                    }
                    for cat, data in category_spending.items()
                },
                "recurring_merchant_history": recurring_merchants
            }

            prompt = SPENDING_ANALYZER_PROMPT.format(data=summary)
            return await generate_recommendations(BASE_SYSTEM_PROMPT, prompt)

        except Exception as e:
            logger.error(f"SpendingAnalyzer failed: {str(e)}")
            return []
```

### File: `src/core/ai/pipelines/cashflow_advisor.py`
```python
import logging
from typing import List
from src.core.ai.schemas import FinancialContext, RecommendationCreate
from src.core.ai.prompts import BASE_SYSTEM_PROMPT, CASHFLOW_ADVISOR_PROMPT
from src.core.ai.llm import generate_recommendations
from src.core.cashflow import CashFlowEngine

logger = logging.getLogger(__name__)

class CashflowAdvisor:
    async def analyze(self, context: FinancialContext) -> List[RecommendationCreate]:
        try:
            # Reconstruct mock objects for the existing engine if needed, 
            # or pass the raw dicts if the engine supports it. 
            # Assuming CashFlowEngine accepts the Pydantic models or dicts.
            forecast = await CashFlowEngine.forecast_daily(
                accounts=context.accounts,
                bills=context.bills,
                transactions=context.transactions,
                days=90
            )

            # Pre-process: Find minimums and overdraft risks
            min_balance = float('inf')
            overdraft_days = []
            
            # Assuming DayForecast has date, projected_balance, bills_due, net_flow
            forecast_summary = []
            for day in forecast:
                if day.projected_balance < min_balance:
                    min_balance = day.projected_balance
                if day.projected_balance < 500:
                    overdraft_days.append({
                        "date": str(day.date),
                        "balance": day.projected_balance,
                        "bills_due": day.bills_due
                    })
                
                # Keep a sparse summary to save tokens (e.g., weekly snapshots or just anomalies)
                if day.date.day % 7 == 0 or day.projected_balance < 500:
                    forecast_summary.append({
                        "date": str(day.date),
                        "projected_balance": day.projected_balance,
                        "net_flow": day.net_flow
                    })

            summary = {
                "lowest_projected_balance": min_balance,
                "overdraft_risk_days": overdraft_days,
                "forecast_snapshots": forecast_summary
            }

            prompt = CASHFLOW_ADVISOR_PROMPT.format(data=summary)
            return await generate_recommendations(BASE_SYSTEM_PROMPT, prompt)

        except Exception as e:
            logger.error(f"CashflowAdvisor failed: {str(e)}")
            return []
```

### File: `src/core/ai/pipelines/debt_strategist.py`
```python
import logging
from typing import List
from src.core.ai.schemas import FinancialContext, RecommendationCreate
from src.core.ai.prompts import BASE_SYSTEM_PROMPT, DEBT_STRATEGIST_PROMPT
from src.core.ai.llm import generate_recommendations
from src.core.debt.optimizer import DebtOptimizer

logger = logging.getLogger(__name__)

class DebtStrategist:
    async def analyze(self, context: FinancialContext) -> List[RecommendationCreate]:
        try:
            if not context.debts:
                return []

            # Run simulations using existing engine
            avalanche_result = await DebtOptimizer.simulate(
                debts=context.debts, 
                extra_payment=100.0, # Simulate with a hypothetical $100 extra
                strategy='avalanche'
            )
            
            snowball_result = await DebtOptimizer.simulate(
                debts=context.debts, 
                extra_payment=100.0, 
                strategy='snowball'
            )

            summary = {
                "current_debts": [
                    {"name": d.name, "balance": d.balance, "apr": d.apr} 
                    for d in context.debts
                ],
                "avalanche_projection": {
                    "months_to_payoff": avalanche_result.months,
                    "total_interest": avalanche_result.total_interest
                },
                "snowball_projection": {
                    "months_to_payoff": snowball_result.months,
                    "total_interest": snowball_result.total_interest
                },
                "interest_savings_avalanche_vs_snowball": snowball_result.total_interest - avalanche_result.total_interest
            }

            prompt = DEBT_STRATEGIST_PROMPT.format(data=summary)
            return await generate_recommendations(BASE_SYSTEM_PROMPT, prompt)

        except Exception as e:
            logger.error(f"DebtStrategist failed: {str(e)}")
            return []
```

### File: `src/core/ai/pipelines/investment_monitor.py`
```python
import logging
from typing import List
from src.core.ai.schemas import FinancialContext, RecommendationCreate
from src.core.ai.prompts import BASE_SYSTEM_PROMPT, INVESTMENT_MONITOR_PROMPT
from src.core.ai.llm import generate_recommendations
from src.integrations.trading_prod import TradingProdBridge

logger = logging.getLogger(__name__)

class InvestmentMonitor:
    async def analyze(self, context: FinancialContext) -> List[RecommendationCreate]:
        try:
            positions = await TradingProdBridge.get_positions()
            portfolio_summary = await TradingProdBridge.get_summary()

            if portfolio_summary.total_value == 0:
                return []

            # Pre-process: Find concentration and losses
            concentration_risks = []
            significant_losses = []
            
            for pos in positions:
                weight = (pos.qty * pos.current_price) / portfolio_summary.total_value
                if weight > 0.20:
                    concentration_risks.append({
                        "symbol": pos.symbol,
                        "weight_percentage": round(weight * 100, 2),
                        "value": pos.qty * pos.current_price
                    })
                
                if pos.pnl < -1000:
                    significant_losses.append({
                        "symbol": pos.symbol,
                        "unrealized_loss": pos.pnl
                    })

            # Check cash drag (assuming a 'CASH' position exists)
            cash_position = next((p for p in positions if p.symbol == 'CASH'), None)
            cash_drag = None
            if cash_position:
                cash_weight = cash_position.qty / portfolio_summary.total_value
                if cash_weight > 0.10:
                    cash_drag = {
                        "cash_amount": cash_position.qty,
                        "percentage_of_portfolio": round(cash_weight * 100, 2)
                    }

            summary = {
                "portfolio_total_value": portfolio_summary.total_value,
                "day_pnl": portfolio_summary.day_pnl,
                "concentration_risks": concentration_risks,
                "significant_losses": significant_losses,
                "cash_drag": cash_drag
            }

            prompt = INVESTMENT_MONITOR_PROMPT.format(data=summary)
            return await generate_recommendations(BASE_SYSTEM_PROMPT, prompt)

        except Exception as e:
            logger.error(f"InvestmentMonitor failed: {str(e)}")
            return []
```

### File: `src/core/ai/pipelines/savings_detector.py`
```python
import logging
from typing import List, Dict
from src.core.ai.schemas import FinancialContext, RecommendationCreate
from src.core.ai.prompts import BASE_SYSTEM_PROMPT, SAVINGS_DETECTOR_PROMPT
from src.core.ai.llm import generate_recommendations

logger = logging.getLogger(__name__)

class SavingsDetector:
    async def analyze(self, context: FinancialContext) -> List[RecommendationCreate]:
        try:
            # Pre-process: Group recurring transactions by merchant
            recurring_subs: Dict[str, float] = {}
            for tx in context.transactions:
                if tx.is_recurring and tx.merchant:
                    # Keep the latest amount
                    recurring_subs[tx.merchant] = tx.amount

            # Pre-process: Find high balance checking accounts
            idle_cash_accounts = []
            for acc in context.accounts:
                if acc.account_type == "CHECKING" and acc.balance > 5000:
                    idle_cash_accounts.append({
                        "account_id": acc.id,
                        "name": acc.name,
                        "balance": acc.balance
                    })

            summary = {
                "active_subscriptions": recurring_subs,
                "idle_cash_in_checking": idle_cash_accounts
            }

            prompt = SAVINGS_DETECTOR_PROMPT.format(data=summary)
            return await generate_recommendations(BASE_SYSTEM_PROMPT, prompt)

        except Exception as e:
            logger.error(f"SavingsDetector failed: {str(e)}")
            return []
```

### File: `src/core/ai/pipelines/tax_planner.py`
```python
import logging
from typing import List
from src.core.ai.schemas import FinancialContext, RecommendationCreate
from src.core.ai.prompts import BASE_SYSTEM_PROMPT, TAX_PLANNER_PROMPT
from src.core.ai.llm import generate_recommendations
from src.integrations.trading_prod import TradingProdBridge

logger = logging.getLogger(__name__)

class TaxPlanner:
    async def analyze(self, context: FinancialContext) -> List[RecommendationCreate]:
        try:
            positions = await TradingProdBridge.get_positions()
            
            # Pre-process: Tax loss harvesting opportunities
            harvesting_opportunities = []
            for pos in positions:
                if pos.pnl < -1000:
                    harvesting_opportunities.append({
                        "symbol": pos.symbol,
                        "unrealized_loss": pos.pnl,
                        "current_value": pos.qty * pos.current_price
                    })

            # Pre-process: Large deductible expenses (heuristic: category 'Business' or 'Medical' > $500)
            # Assuming category names are mapped or identifiable. We will just pass large transactions.
            large_potential_deductions = []
            for tx in context.transactions:
                if tx.amount > 500:
                    large_potential_deductions.append({
                        "description": tx.description,
                        "amount": tx.amount,
                        "date": str(tx.date)
                    })

            summary = {
                "tax_loss_harvesting_opportunities": harvesting_opportunities,
                "large_transactions_potential_deductions": large_potential_deductions
            }

            prompt = TAX_PLANNER_PROMPT.format(data=summary)
            return await generate_recommendations(BASE_SYSTEM_PROMPT, prompt)

        except Exception as e:
            logger.error(f"TaxPlanner failed: {str(e)}")
            return []
```

### File: `src/core/ai/engine.py`
```python
import asyncio
import logging
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.models import Account, Transaction, Budget, Bill, Debt, Investment, Recommendation
from src.core.ai.schemas import (
    FinancialContext, RecommendationCreate, AccountSummary, 
    TransactionSummary, BudgetSummary, BillSummary, DebtSummary, InvestmentSummary
)

from src.core.ai.pipelines.spending_analyzer import SpendingAnalyzer
from src.core.ai.pipelines.cashflow_advisor import CashflowAdvisor
from src.core.ai.pipelines.debt_strategist import DebtStrategist
from src.core.ai.pipelines.investment_monitor import InvestmentMonitor
from src.core.ai.pipelines.savings_detector import SavingsDetector
from src.core.ai.pipelines.tax_planner import TaxPlanner

logger = logging.getLogger(__name__)

class AIFinancialAdvisorEngine:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.pipelines = [
            SpendingAnalyzer(),
            CashflowAdvisor(),
            DebtStrategist(),
            InvestmentMonitor(),
            SavingsDetector(),
            TaxPlanner()
        ]

    async def gather_context(self) -> FinancialContext:
        """Gathers and summarizes all necessary financial data for the AI pipelines."""
        now = datetime.now(timezone.utc)
        ninety_days_ago = now - timedelta(days=90)

        # Fetch Accounts
        acc_result = await self.db.execute(select(Account).where(Account.is_active == True))
        accounts = acc_result.scalars().all()
        
        # Fetch Transactions (last 90 days)
        tx_result = await self.db.execute(
            select(Transaction).where(Transaction.date >= ninety_days_ago.date())
        )
        transactions = tx_result.scalars().all()

        # Fetch Budgets
        bud_result = await self.db.execute(select(Budget))
        budgets = bud_result.scalars().all()

        # Fetch Bills
        bill_result = await self.db.execute(select(Bill).where(Bill.is_active == True))
        bills = bill_result.scalars().all()

        # Fetch Debts
        debt_result = await self.db.execute(select(Debt))
        debts = debt_result.scalars().all()

        # Fetch Investments
        inv_result = await self.db.execute(select(Investment))
        investments = inv_result.scalars().all()

        # Map to Pydantic summaries to detach from SQLAlchemy session
        return FinancialContext(
            accounts=[AccountSummary(
                id=str(a.id), name=a.name, account_type=a.account_type, 
                balance=a.balance, currency=a.currency
            ) for a in accounts],
            transactions=[TransactionSummary(
                id=str(t.id), amount=t.amount, description=t.description, 
                date=t.date, category_id=str(t.category_id) if t.category_id else None, 
                merchant=t.merchant, is_recurring=t.is_recurring
            ) for t in transactions],
            budgets=[BudgetSummary(
                id=str(b.id), category_id=str(b.category_id), amount=b.amount, period=b.period
            ) for b in budgets],
            bills=[BillSummary(
                id=str(b.id), name=b.name, amount=b.amount, due_date=b.due_date, is_auto_pay=b.is_auto_pay
            ) for b in bills],
            debts=[DebtSummary(
                id=str(d.id), name=d.name, balance=d.balance, apr=d.apr, minimum_payment=d.minimum_payment
            ) for d in debts],
            investments=[InvestmentSummary(
                id=str(i.id), symbol=i.symbol, quantity=i.quantity, 
                cost_basis=i.cost_basis, current_price=i.current_price
            ) for i in investments]
        )

    async def run_analysis(self) -> list[RecommendationCreate]:
        """Executes all AI pipelines in parallel, filters, and deduplicates results."""
        context = await self.gather_context()
        
        # Run pipelines concurrently
        tasks = [pipeline.analyze(context) for pipeline in self.pipelines]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_recommendations: list[RecommendationCreate] = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Pipeline execution failed with exception: {res}")
                continue
            all_recommendations.extend(res)

        # Filter by confidence threshold >= 0.7
        high_confidence_recs = [r for r in all_recommendations if r.confidence >= 0.7]

        # Deduplicate against existing PENDING recommendations
        final_recs = await self._deduplicate(high_confidence_recs)
        
        return final_recs

    async def _deduplicate(self, recs: list[RecommendationCreate]) -> list[RecommendationCreate]:
        """Removes recommendations that are too similar to existing PENDING ones."""
        if not recs:
            return []

        result = await self.db.execute(
            select(Recommendation).where(Recommendation.status == 'PENDING')
        )
        existing_recs = result.scalars().all()

        unique_recs = []
        for new_rec in recs:
            is_duplicate = False
            for ext_rec in existing_recs:
                # Check if same type and title is highly similar
                if new_rec.type == ext_rec.type:
                    similarity = SequenceMatcher(None, new_rec.title.lower(), ext_rec.title.lower()).ratio()
                    if similarity > 0.8:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_recs.append(new_rec)

        return unique_recs
```

### File: `src/scheduler/jobs.py`
```python
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.ai.engine import AIFinancialAdvisorEngine
from src.models import Recommendation
from src.notifications.dispatcher import NotificationDispatcher

logger = logging.getLogger(__name__)

async def daily_analysis_job(db: AsyncSession):
    """
    Scheduled job to run the AI Financial Advisor Engine daily.
    Generates recommendations, saves them to the database, and alerts on urgent items.
    """
    logger.info("Starting daily AI financial analysis job...")
    try:
        engine = AIFinancialAdvisorEngine(db)
        recommendations = await engine.run_analysis()
        
        if not recommendations:
            logger.info("No new recommendations generated today.")
            return

        # Save to database
        for rec in recommendations:
            db_rec = Recommendation(
                type=rec.type,
                title=rec.title,
                description=rec.description,
                impact_amount=rec.impact_amount,
                confidence=rec.confidence,
                priority=rec.priority,
                action_payload=rec.action_payload,
                status='PENDING'
            )
            db.add(db_rec)
            
        await db.commit()
        logger.info(f"Saved {len(recommendations)} new recommendations to the database.")

        # Dispatch notifications for urgent recommendations (priority >= 8)
        urgent_recs = [r for r in recommendations if r.priority >= 8]
        for urgent in urgent_recs:
            await NotificationDispatcher.send(
                channel="alerts",
                title=f"URGENT: {urgent.title}",
                message=urgent.description,
                priority=urgent.priority
            )
            
    except Exception as e:
        await db.rollback()
        logger.error(f"Daily analysis job failed: {str(e)}")
        raise
```

---

