Here is the fully corrected, production-ready implementation of the Smart Budget Alert and Notification System. It adheres strictly to your existing architecture, models, database patterns, and configuration, while resolving all syntax, dependency, and logic issues from the previous attempt.

### File Structure
```text
src/
├── __init__.py
├── models/
│   ├── __init__.py
│   └── preference.py           # Alert preferences model
├── schemas/
│   ├── __init__.py
│   └── alerts.py               # Pydantic schemas
├── services/
│   ├── __init__.py
│   └── alert_engine/
│       ├── __init__.py
│       ├── notifications.py    # Multi-channel dispatch
│       ├── ai_insights.py      # Anthropic API integration
│       └── engine.py           # Core monitoring logic
├── api/
│   ├── __init__.py
│   └── routers/
│       ├── __init__.py
│       └── alerts.py           # FastAPI router
├── worker.py                   # Asyncio background task
frontend/
└── AlertsDashboard.tsx         # React frontend
```

### 1. Models & Schemas

**`src/models/preference.py`**
```python
from sqlalchemy import String, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from src.models.base import BaseModel

class AlertPreference(BaseModel):
    __tablename__ = "alert_preferences"
    
    user_id: Mapped[str] = mapped_column(String(50), unique=True, index=True, default="default")
    alerts_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    telegram_chat_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    channels: Mapped[list[str]] = mapped_column(JSONB, default=["DASHBOARD"])
```

**`src/schemas/alerts.py`**
```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class AlertPreferenceCreate(BaseModel):
    alerts_enabled: bool = True
    telegram_chat_id: Optional[str] = None
    channels: list[str] = ["DASHBOARD"]

class AlertPreferenceUpdate(BaseModel):
    alerts_enabled: Optional[bool] = None
    telegram_chat_id: Optional[str] = None
    channels: Optional[list[str]] = None

class AlertPreferenceResponse(BaseModel):
    id: str
    user_id: str
    alerts_enabled: bool
    telegram_chat_id: Optional[str]
    channels: list[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

class NotificationResponse(BaseModel):
    id: str
    channel: str
    subject: str
    body: str
    metadata_json: dict | None
    sent_ok: bool
    error: str | None
    created_at: datetime

    model_config = {"from_attributes": True}
```

### 2. Alert Engine Services

**`src/services/alert_engine/notifications.py`**
```python
import logging
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.notification import Notification, NotificationChannel
from src.config import settings

logger = logging.getLogger(__name__)

async def dispatch_alert(
    db: AsyncSession,
    channels: list[str],
    subject: str,
    body: str,
    metadata: dict,
    telegram_chat_id: str | None = None
) -> None:
    for channel_str in channels:
        try:
            channel_enum = NotificationChannel[channel_str]
        except KeyError:
            logger.warning(f"Unknown notification channel {channel_str}")
            continue

        sent_ok = True
        error_msg = None

        if channel_enum == NotificationChannel.TELEGRAM:
            if settings.telegram_bot_token and telegram_chat_id:
                try:
                    async with httpx.AsyncClient() as client:
                        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
                        payload = {
                            "chat_id": telegram_chat_id,
                            "text": f"*{subject}*\n\n{body}",
                            "parse_mode": "Markdown"
                        }
                        resp = await client.post(url, json=payload, timeout=10.0)
                        resp.raise_for_status()
                except Exception as e:
                    logger.error(f"Telegram dispatch failed: {e}")
                    sent_ok = False
                    error_msg = str(e)
            else:
                sent_ok = False
                error_msg = "Telegram config missing"

        elif channel_enum == NotificationChannel.DASHBOARD:
            pass  # Dashboard alerts are simply stored in the DB logs
        
        else:
            sent_ok = False
            error_msg = "Channel dispatch not implemented"

        notification = Notification(
            channel=channel_enum,
            subject=subject,
            body=body,
            metadata_json=metadata,
            sent_ok=sent_ok,
            error=error_msg
        )
        db.add(notification)
```

**`src/services/alert_engine/ai_insights.py`**
```python
import logging
import json
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.recommendation import Recommendation, RecommendationType, RecommendationStatus
from src.config import settings

logger = logging.getLogger(__name__)

async def generate_weekly_digest(db: AsyncSession, financial_summary: dict) -> None:
    if not settings.anthropic_api_key:
        logger.warning("Anthropic API key not set. Skipping AI insights.")
        return

    prompt = f"""
    You are a financial advisor AI. Based on the following user financial summary for the week, generate a personalized financial insight and recommendation.
    Summary: {json.dumps(financial_summary)}
    
    Return ONLY a JSON object with these keys:
    - title: A short title for the recommendation.
    - summary: A one-sentence summary.
    - detail: A detailed paragraph explaining the insight.
    - recommendation_type: One of [BUDGET_ADJUSTMENT, DEBT_PAYMENT, INVESTMENT_REBALANCE, SAVINGS_OPPORTUNITY, TAX_OPTIMIZATION, SPENDING_ALERT, CASHFLOW_WARNING, GENERAL_INSIGHT].
    - impact_amount: Estimated monetary impact (number, or null).
    - confidence: Float between 0.0 and 1.0.
    """

    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": settings.ai_model,
                "max_tokens": 500,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": prompt}]
            }
            resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=30.0)
            resp.raise_for_status()
            
            data = resp.json()
            content = data["content"][0]["text"]
            
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                insight = json.loads(json_str)
                
                rec_type_str = insight.get("recommendation_type", "GENERAL_INSIGHT")
                try:
                    rec_type = RecommendationType[rec_type_str]
                except KeyError:
                    rec_type = RecommendationType.GENERAL_INSIGHT

                recommendation = Recommendation(
                    recommendation_type=rec_type,
                    status=RecommendationStatus.PENDING,
                    title=insight.get("title", "Weekly AI Insight"),
                    summary=insight.get("summary", "Your weekly financial summary is ready."),
                    detail=insight.get("detail", ""),
                    impact_amount=insight.get("impact_amount"),
                    confidence=insight.get("confidence", 0.8),
                    priority=2
                )
                db.add(recommendation)
    except Exception as e:
        logger.error(f"Failed to generate AI insights: {e}")
```

**`src/services/alert_engine/engine.py`**
```python
import logging
from datetime import date, timedelta
from sqlalchemy import select, func, and_
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.budget import Budget
from src.models.transaction import Transaction, TransactionType
from src.models.category import Category
from src.models.bill import RecurringBill
from src.models.account import FinancialAccount
from src.models.preference import AlertPreference
from src.models.notification import Notification

from src.services.alert_engine.notifications import dispatch_alert
from src.services.alert_engine.ai_insights import generate_weekly_digest
from src.core.cashflow import project_bills, aggregate_daily_forecast

logger = logging.getLogger(__name__)

async def run_alert_engine(db: AsyncSession) -> None:
    logger.info("Starting alert engine run")
    try:
        prefs_result = await db.execute(select(AlertPreference).where(AlertPreference.user_id == "default"))
        prefs = prefs_result.scalar_one_or_none()
        
        if not prefs or not prefs.alerts_enabled:
            logger.info("Alerts disabled or preferences not found. Skipping.")
            return

        channels = prefs.channels
        telegram_chat_id = prefs.telegram_chat_id

        await _check_budgets(db, channels, telegram_chat_id)
        await _check_bills(db, channels, telegram_chat_id)
        await _check_anomalies(db, channels, telegram_chat_id)
        await _check_cashflow(db, channels, telegram_chat_id)
        
        if date.today().weekday() == 6:  # Run Weekly Digest on Sundays
            await _run_weekly_digest(db)
            
        await db.commit()
        logger.info("Alert engine run completed successfully")
    except Exception as e:
        logger.exception(f"Error in alert engine: {e}")
        await db.rollback()

async def _check_budgets(db: AsyncSession, channels: list[str], telegram_chat_id: str | None) -> None:
    current_month = date.today().strftime("%Y-%m")
    
    # Eager load category to prevent MissingGreenlet errors
    budgets_result = await db.execute(
        select(Budget).options(joinedload(Budget.category)).where(Budget.month == current_month)
    )
    budgets = budgets_result.scalars().all()

    for budget in budgets:
        start_date = date.today().replace(day=1)
        spending_result = await db.execute(
            select(func.sum(Transaction.amount)).where(
                Transaction.category_id == budget.category_id,
                Transaction.transaction_type == TransactionType.EXPENSE,
                Transaction.date >= start_date
            )
        )
        spent = spending_result.scalar() or 0.0
        
        if budget.budgeted_amount > 0:
            ratio = spent / budget.budgeted_amount
            if ratio >= budget.alert_threshold:
                subject = f"Budget Alert: {budget.category.name}"
                already_sent = await db.execute(
                    select(Notification).where(
                        Notification.subject == subject,
                        Notification.metadata_json['month'].astext == current_month,
                        Notification.metadata_json['threshold_crossed'].astext == "true"
                    )
                )
                if not already_sent.first():
                    await dispatch_alert(
                        db, channels, subject,
                        f"You have spent ${spent:.2f} of your ${budget.budgeted_amount:.2f} budget for {budget.category.name}.",
                        {"month": current_month, "budget_id": str(budget.id), "threshold_crossed": "true"},
                        telegram_chat_id
                    )

async def _check_bills(db: AsyncSession, channels: list[str], telegram_chat_id: str | None) -> None:
    today = date.today()
    bills_result = await db.execute(select(RecurringBill).where(RecurringBill.is_active == True))
    
    for bill in bills_result.scalars().all():
        if not bill.next_due_date:
            continue
            
        days_until_due = (bill.next_due_date - today).days
        
        # Only trigger exactly on the requested alert day to handle frequencies safely
        if days_until_due == bill.alert_days_before:
            subject = f"Bill Due Reminder: {bill.name}"
            already_sent = await db.execute(
                select(Notification).where(
                    Notification.subject == subject,
                    Notification.metadata_json['due_date'].astext == bill.next_due_date.isoformat()
                )
            )
            if not already_sent.first():
                await dispatch_alert(
                    db, channels, subject,
                    f"Your bill '{bill.name}' for ${bill.amount:.2f} is due in {days_until_due} days on {bill.next_due_date}.",
                    {"bill_id": str(bill.id), "due_date": bill.next_due_date.isoformat()},
                    telegram_chat_id
                )

async def _check_anomalies(db: AsyncSession, channels: list[str], telegram_chat_id: str | None) -> None:
    today = date.today()
    
    # 1. Large single transactions (Efficient N+1 free checking)
    avg_query = await db.execute(
        select(Transaction.category_id, func.avg(Transaction.amount).label('avg_amount'))
        .where(
            Transaction.transaction_type == TransactionType.EXPENSE,
            Transaction.category_id.is_not(None)
        ).group_by(Transaction.category_id)
    )
    category_averages = {row.category_id: row.avg_amount for row in avg_query.all()}
    
    recent_tx_result = await db.execute(
        select(Transaction).options(joinedload(Transaction.category)).where(
            Transaction.date >= today - timedelta(days=1),
            Transaction.transaction_type == TransactionType.EXPENSE,
            Transaction.category_id.is_not(None)
        )
    )
    for tx in recent_tx_result.scalars().all():
        avg_amt = category_averages.get(tx.category_id)
        if avg_amt and avg_amt > 10.0 and tx.amount > (avg_amt * 2):
            subject = f"Large Transaction Detected: {tx.merchant or tx.description}"
            already_sent = await db.execute(
                select(Notification).where(Notification.metadata_json['transaction_id'].astext == str(tx.id))
            )
            if not already_sent.first():
                await dispatch_alert(
                    db, channels, subject,
                    f"A transaction of ${tx.amount:.2f} was recorded in {tx.category.name}, which is more than twice your average.",
                    {"transaction_id": str(tx.id), "anomaly_type": "LARGE_TRANSACTION"},
                    telegram_chat_id
                )

    # 2. Sudden spending spikes (Week-over-Week)
    last_7_days = today - timedelta(days=7)
    prev_7_days = today - timedelta(days=14)
    
    sum_last_7 = await db.execute(
        select(func.sum(Transaction.amount)).where(
            Transaction.transaction_type == TransactionType.EXPENSE,
            Transaction.date > last_7_days,
            Transaction.date <= today
        )
    )
    val_last_7 = sum_last_7.scalar() or 0.0

    sum_prev_7 = await db.execute(
        select(func.sum(Transaction.amount)).where(
            Transaction.transaction_type == TransactionType.EXPENSE,
            Transaction.date > prev_7_days,
            Transaction.date <= last_7_days
        )
    )
    val_prev_7 = sum_prev_7.scalar() or 0.0

    if val_prev_7 > 50.0 and val_last_7 > (val_prev_7 * 1.5):
        subject = "Spending Spike Detected"
        already_sent = await db.execute(
            select(Notification).where(
                Notification.subject == subject,
                Notification.metadata_json['week_start'].astext == last_7_days.isoformat()
            )
        )
        if not already_sent.first():
            await dispatch_alert(
                db, channels, subject,
                f"Your spending in the last 7 days (${val_last_7:.2f}) is significantly higher than the previous week (${val_prev_7:.2f}).",
                {"anomaly_type": "SPENDING_SPIKE", "week_start": last_7_days.isoformat()},
                telegram_chat_id
            )

    # 3. New recurring charges
    recent_non_recurring = await db.execute(
        select(Transaction).where(
            Transaction.date >= last_7_days,
            Transaction.transaction_type == TransactionType.EXPENSE,
            Transaction.is_recurring == False,
            Transaction.merchant.is_not(None)
        )
    )
    for tx in recent_non_recurring.scalars().all():
        target_date_start = tx.date - timedelta(days=35)
        target_date_end = tx.date - timedelta(days=25)
        
        similar_tx = await db.execute(
            select(Transaction).where(
                Transaction.merchant == tx.merchant,
                Transaction.amount.between(tx.amount * 0.9, tx.amount * 1.1),
                Transaction.date.between(target_date_start, target_date_end),
                Transaction.id != tx.id
            )
        )
        if similar_tx.first():
            subject = f"Possible New Recurring Charge: {tx.merchant}"
            already_sent = await db.execute(
                select(Notification).where(
                    Notification.subject == subject,
                    Notification.metadata_json['merchant'].astext == tx.merchant
                )
            )
            if not already_sent.first():
                await dispatch_alert(
                    db, channels, subject,
                    f"We noticed multiple identical charges for {tx.merchant} around ${tx.amount:.2f}. Is this a new subscription?",
                    {"anomaly_type": "NEW_RECURRING", "merchant": tx.merchant, "transaction_id": str(tx.id)},
                    telegram_chat_id
                )

async def _check_cashflow(db: AsyncSession, channels: list[str], telegram_chat_id: str | None) -> None:
    today = date.today()
    end_date = today + timedelta(days=14)

    bills_result = await db.execute(select(RecurringBill).where(RecurringBill.is_active == True))
    bills = bills_result.scalars().all()
    
    # Synchronous cashflow engine calls
    projections = project_bills(bills, today, end_date)
    forecast = aggregate_daily_forecast(projections, today, end_date)

    accounts_result = await db.execute(
        select(FinancialAccount).where(FinancialAccount.account_type.in_(["CHECKING", "SAVINGS", "CASH"]))
    )
    running_balance = sum(acc.balance for acc in accounts_result.scalars().all())
    negative_date = None

    for day in forecast:
        running_balance += day.net
        if running_balance < 0:
            negative_date = day.date
            break

    if negative_date:
        subject = "Cash Flow Warning"
        already_sent = await db.execute(
            select(Notification).where(
                Notification.subject == subject,
                Notification.metadata_json['negative_date'].astext == negative_date.isoformat()
            )
        )
        if not already_sent.first():
            await dispatch_alert(
                db, channels, subject,
                f"Warning: Your projected cash balance may drop below zero on {negative_date}.",
                {"negative_date": negative_date.isoformat()},
                telegram_chat_id
            )

async def _run_weekly_digest(db: AsyncSession) -> None:
    today = date.today()
    start_of_week = today - timedelta(days=7)
    
    expenses_result = await db.execute(
        select(func.sum(Transaction.amount)).where(
            Transaction.transaction_type == TransactionType.EXPENSE,
            Transaction.date >= start_of_week
        )
    )
    income_result = await db.execute(
        select(func.sum(Transaction.amount)).where(
            Transaction.transaction_type == TransactionType.INCOME,
            Transaction.date >= start_of_week
        )
    )
    
    summary = {
        "week_start": start_of_week.isoformat(),
        "total_expenses": expenses_result.scalar() or 0.0,
        "total_income": income_result.scalar() or 0.0
    }
    
    already_done = await db.execute(
        select(Notification).where(
            Notification.subject == "Weekly Digest Generated",
            Notification.metadata_json['week_start'].astext == start_of_week.isoformat()
        )
    )
    if not already_done.first():
        await generate_weekly_digest(db, summary)
        
        notif = Notification(
            channel="DASHBOARD",
            subject="Weekly Digest Generated",
            body="AI Insights generated successfully.",
            metadata_json={"week_start": start_of_week.isoformat()},
            sent_ok=True
        )
        db.add(notif)
```

### 3. API Router & Worker

**`src/api/routers/alerts.py`**
```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.deps import get_db
from src.models.preference import AlertPreference
from src.models.notification import Notification
from src.schemas.alerts import AlertPreferenceUpdate, AlertPreferenceResponse, NotificationResponse

router = APIRouter(prefix="/alerts", tags=["alerts"])

@router.get("/preferences", response_model=AlertPreferenceResponse)
async def get_preferences(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AlertPreference).where(AlertPreference.user_id == "default"))
    prefs = result.scalar_one_or_none()
    
    if not prefs:
        prefs = AlertPreference(user_id="default")
        db.add(prefs)
        await db.commit()
        await db.refresh(prefs)
        
    return prefs

@router.put("/preferences", response_model=AlertPreferenceResponse)
async def update_preferences(data: AlertPreferenceUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AlertPreference).where(AlertPreference.user_id == "default"))
    prefs = result.scalar_one_or_none()
    
    if not prefs:
        prefs = AlertPreference(user_id="default")
        db.add(prefs)
    
    if data.alerts_enabled is not None:
        prefs.alerts_enabled = data.alerts_enabled
    if data.telegram_chat_id is not None:
        prefs.telegram_chat_id = data.telegram_chat_id
    if data.channels is not None:
        prefs.channels = data.channels
        
    await db.commit()
    await db.refresh(prefs)
    return prefs

@router.get("/history", response_model=list[NotificationResponse])
async def get_alert_history(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Notification)
        .order_by(Notification.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()
```

**`src/worker.py`**
```python
import asyncio
import logging
from src.db.engine import async_session
from src.services.alert_engine.engine import run_alert_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RUN_INTERVAL_SECONDS = 12 * 60 * 60  # Execute every 12 hours

async def alert_worker():
    logger.info("Starting background alert worker...")
    while True:
        try:
            async with async_session() as db:
                await run_alert_engine(db)
        except Exception as e:
            logger.error(f"Alert worker encountered an error: {e}")
            
        logger.info(f"Sleeping for {RUN_INTERVAL_SECONDS} seconds...")
        await asyncio.sleep(RUN_INTERVAL_SECONDS)

if __name__ == "__main__":
    asyncio.run(alert_worker())
```

### 4. Frontend Component

**`frontend/AlertsDashboard.tsx`**
```tsx
import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from './client';

interface AlertPreference {
  id: string;
  alerts_enabled: boolean;
  telegram_chat_id: string | null;
  channels: string[];
}

interface NotificationLog {
  id: string;
  channel: string;
  subject: string;
  body: string;
  created_at: string;
  sent_ok: boolean;
}

export default function AlertsDashboard() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<'history' | 'preferences'>('history');

  const { data: prefs, isLoading: prefsLoading } = useQuery<AlertPreference>({
    queryKey: ['alertPreferences'],
    queryFn: async () => {
      const res = await apiClient.get('/alerts/preferences');
      return res.data;
    }
  });

  const { data: history, isLoading: historyLoading } = useQuery<NotificationLog[]>({
    queryKey: ['alertHistory'],
    queryFn: async () => {
      const res = await apiClient.get('/alerts/history');
      return res.data;
    }
  });

  const updatePrefsMutation = useMutation({
    mutationFn: async (newPrefs: Partial<AlertPreference>) => {
      const res = await apiClient.put('/alerts/preferences', newPrefs);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alertPreferences'] });
      alert('Preferences saved successfully!');
    }
  });

  const handleSavePrefs = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const alerts_enabled = formData.get('alerts_enabled') === 'on';
    const telegram_chat_id = formData.get('telegram_chat_id') as string;
    const channelsStr = formData.get('channels') as string;
    const channels = channelsStr.split(',');

    updatePrefsMutation.mutate({
      alerts_enabled,
      telegram_chat_id: telegram_chat_id || null,
      channels
    });
  };

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">Smart Alerts & Notifications</h1>
      </div>

      <div className="flex space-x-4 border-b border-gray-200">
        <button
          onClick={() => setActiveTab('history')}
          className={`py-2 px-4 font-medium transition-colors ${activeTab === 'history' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
        >
          Alert History
        </button>
        <button
          onClick={() => setActiveTab('preferences')}
          className={`py-2 px-4 font-medium transition-colors ${activeTab === 'preferences' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
        >
          Preferences
        </button>
      </div>

      {activeTab === 'history' && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          {historyLoading ? (
            <div className="p-6 text-gray-500">Loading history...</div>
          ) : history && history.length > 0 ? (
            <ul className="divide-y divide-gray-100">
              {history.map((log) => (
                <li key={log.id} className="p-6 hover:bg-gray-50 transition-colors">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="text-lg font-semibold text-gray-800">{log.subject}</h3>
                      <p className="text-gray-600 mt-1">{log.body}</p>
                      <div className="flex items-center gap-3 mt-3 text-sm text-gray-500">
                        <span className="bg-gray-100 px-2 py-1 rounded text-xs font-medium">
                          {log.channel}
                        </span>
                        <span>{new Date(log.created_at).toLocaleString()}</span>
                        {!log.sent_ok && (
                          <span className="text-red-500 font-medium">Delivery Failed</span>
                        )}
                      </div>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="p-12 text-center text-gray-500">
              No alerts have been generated yet.
            </div>
          )}
        </div>
      )}

      {activeTab === 'preferences' && prefs && !prefsLoading && (
        <form onSubmit={handleSavePrefs} className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 space-y-6 max-w-lg">
          <div className="flex items-center justify-between">
            <label className="font-semibold text-gray-800">Enable Smart Alerts</label>
            <input
              type="checkbox"
              name="alerts_enabled"
              defaultChecked={prefs.alerts_enabled}
              className="w-5 h-5 text-blue-600 rounded"
            />
          </div>

          <div>
            <label className="block font-semibold text-gray-800 mb-2">Delivery Channels</label>
            <select
              name="channels"
              defaultValue={prefs.channels.join(',')}
              className="w-full border border-gray-300 rounded-lg p-3 bg-gray-50 focus:ring-2 focus:ring-blue-500"
            >
              <option value="DASHBOARD">In-App Dashboard Only</option>
              <option value="DASHBOARD,TELEGRAM">Dashboard & Telegram</option>
            </select>
          </div>

          <div>
            <label className="block font-semibold text-gray-800 mb-2">Telegram Chat ID</label>
            <input
              type="text"
              name="telegram_chat_id"
              defaultValue={prefs.telegram_chat_id || ''}
              className="w-full border border-gray-300 rounded-lg p-3 bg-gray-50 focus:ring-2 focus:ring-blue-500"
              placeholder="e.g. 123456789"
            />
            <p className="text-sm text-gray-500 mt-2">
              If enabled, alerts will be sent directly to your Telegram.
            </p>
          </div>

          <button 
            type="submit" 
            disabled={updatePrefsMutation.isPending}
            className="w-full bg-blue-600 text-white font-semibold py-3 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            {updatePrefsMutation.isPending ? 'Saving...' : 'Save Preferences'}
          </button>
        </form>
      )}
    </div>
  );
}
```