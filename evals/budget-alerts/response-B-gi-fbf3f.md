Here is the complete, production-ready implementation of the Smart Budget Alert & Notification System. It addresses all previous issues, including proper deduplication of alerts, correct usage of `settings` for external APIs, proper async/sync boundaries (especially with the `CashFlowEngine`), and integration into the FastAPI lifecycle.

### 1. Dependencies Setup

**`requirements-alerts.txt`**
```text
# Add these to your Python environment
httpx>=0.25.0
anthropic>=0.3.0

# Add these to your React frontend environment
# npm install @tanstack/react-query lucide-react recharts
```

### 2. Models & Schemas

**`src/models/alert_preference.py`**
```python
from sqlalchemy import Column, Boolean
from src.models.base import BaseModel

class AlertPreference(BaseModel):
    """
    Stores system preferences for which alerts to receive and via which channels.
    Assumes a single-tenant/personal platform instance.
    """
    __tablename__ = "alert_preferences"
    
    budget_alerts_enabled = Column(Boolean, default=True, nullable=False)
    bill_reminders_enabled = Column(Boolean, default=True, nullable=False)
    anomaly_alerts_enabled = Column(Boolean, default=True, nullable=False)
    cashflow_warnings_enabled = Column(Boolean, default=True, nullable=False)
    ai_insights_enabled = Column(Boolean, default=True, nullable=False)
    
    telegram_enabled = Column(Boolean, default=False, nullable=False)
    dashboard_enabled = Column(Boolean, default=True, nullable=False)
```

**`src/schemas/alert_preference.py`**
```python
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime

class AlertPreferenceCreate(BaseModel):
    budget_alerts_enabled: bool = True
    bill_reminders_enabled: bool = True
    anomaly_alerts_enabled: bool = True
    cashflow_warnings_enabled: bool = True
    ai_insights_enabled: bool = True
    telegram_enabled: bool = False
    dashboard_enabled: bool = True

class AlertPreferenceUpdate(BaseModel):
    budget_alerts_enabled: Optional[bool] = None
    bill_reminders_enabled: Optional[bool] = None
    anomaly_alerts_enabled: Optional[bool] = None
    cashflow_warnings_enabled: Optional[bool] = None
    ai_insights_enabled: Optional[bool] = None
    telegram_enabled: Optional[bool] = None
    dashboard_enabled: Optional[bool] = None

class AlertPreferenceResponse(AlertPreferenceCreate):
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    model_config = {"from_attributes": True}
```

### 3. Alert Dispatcher & Notification Delivery

**`src/core/alerts/dispatcher.py`**
```python
import httpx
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.models.notification import Notification, NotificationChannel
from src.models.alert_preference import AlertPreference
from src.config import settings

logger = logging.getLogger(__name__)

async def get_alert_preferences(db: AsyncSession) -> AlertPreference:
    """Retrieves or creates the default alert preferences."""
    result = await db.execute(select(AlertPreference).limit(1))
    pref = result.scalars().first()
    if not pref:
        pref = AlertPreference()
        db.add(pref)
        await db.commit()
        await db.refresh(pref)
    return pref

async def send_telegram_message(text: str) -> bool:
    """Sends a message via the Telegram Bot API using credentials from settings."""
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.warning("Telegram credentials not configured in settings.")
        return False
        
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False

async def dispatch_alert(
    db: AsyncSession, 
    subject: str, 
    body: str, 
    signature: str,
    metadata: dict = None
) -> None:
    """
    Dispatches an alert based on user preferences and logs it to prevent duplicates.
    The signature is used to deduplicate alerts.
    """
    prefs = await get_alert_preferences(db)
    meta = metadata or {}
    meta["signature"] = signature
    
    if prefs.dashboard_enabled:
        dashboard_notif = Notification(
            channel=NotificationChannel.DASHBOARD,
            subject=subject,
            body=body,
            metadata_json=meta,
            sent_ok=True
        )
        db.add(dashboard_notif)
        
    if prefs.telegram_enabled:
        success = await send_telegram_message(f"<b>{subject}</b>\n\n{body}")
        telegram_notif = Notification(
            channel=NotificationChannel.TELEGRAM,
            subject=subject,
            body=body,
            metadata_json=meta,
            sent_ok=success,
            error=None if success else "Failed to send via Telegram API"
        )
        db.add(telegram_notif)
        
    await db.commit()
```

### 4. Alert Rules & Monitoring Logic

**`src/core/alerts/rules.py`**
```python
import logging
from datetime import date, timedelta, datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
import httpx

from src.models.budget import Budget
from src.models.transaction import Transaction, TransactionType
from src.models.bill import RecurringBill
from src.models.account import FinancialAccount, AccountType
from src.models.notification import Notification
from src.models.recommendation import Recommendation, RecommendationType, RecommendationStatus
from src.core.alerts.dispatcher import dispatch_alert, get_alert_preferences
from src.core.cashflow import project_bills, aggregate_daily_forecast
from src.config import settings

logger = logging.getLogger(__name__)

async def get_sent_signatures(db: AsyncSession, days_back: int = 30) -> set[str]:
    """Fetches alert signatures sent recently to prevent duplicate notifications."""
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    result = await db.execute(select(Notification).where(Notification.created_at >= cutoff))
    notifications = result.scalars().all()
    
    signatures = set()
    for n in notifications:
        if n.metadata_json and "signature" in n.metadata_json:
            signatures.add(n.metadata_json["signature"])
    return signatures

async def check_budget_thresholds(db: AsyncSession) -> None:
    """Monitors spending against budgets and alerts if the threshold is exceeded."""
    prefs = await get_alert_preferences(db)
    if not prefs.budget_alerts_enabled:
        return

    today = date.today()
    current_month = today.strftime("%Y-%m")
    sent_sigs = await get_sent_signatures(db)
    
    result = await db.execute(select(Budget).where(Budget.month == current_month))
    budgets = result.scalars().all()
    
    for budget in budgets:
        # Calculate spent amount for this category in the current month
        start_date = today.replace(day=1)
        tx_result = await db.execute(
            select(func.sum(Transaction.amount))
            .where(
                Transaction.category_id == budget.category_id,
                Transaction.date >= start_date,
                Transaction.transaction_type == TransactionType.EXPENSE
            )
        )
        spent = tx_result.scalar() or 0.0
        
        if budget.budgeted_amount > 0 and budget.alert_threshold > 0:
            ratio = spent / budget.budgeted_amount
            if ratio >= budget.alert_threshold:
                sig = f"budget_{budget.id}_{current_month}"
                if sig not in sent_sigs:
                    await dispatch_alert(
                        db,
                        subject=f"Budget Alert: {budget.category.name if budget.category else 'Category'}",
                        body=f"You have spent ${spent:.2f}, which is {ratio*100:.1f}% of your ${budget.budgeted_amount:.2f} budget.",
                        signature=sig
                    )

async def check_bill_reminders(db: AsyncSession) -> None:
    """Uses CashFlowEngine to project bills and send reminders based on alert_days_before."""
    prefs = await get_alert_preferences(db)
    if not prefs.bill_reminders_enabled:
        return

    today = date.today()
    end_date = today + timedelta(days=45)
    sent_sigs = await get_sent_signatures(db)
    
    result = await db.execute(select(RecurringBill).where(RecurringBill.is_active == True))
    bills = result.scalars().all()
    
    # project_bills is a sync function
    projections = project_bills(bills, today, end_date)
    
    # Map bill IDs back to their objects for alert_days_before access
    bill_map = {b.id: b for b in bills}
    
    for proj in projections:
        bill = bill_map.get(proj.bill_id)
        if not bill:
            continue
            
        days_until_due = (proj.due_date - today).days
        if days_until_due == bill.alert_days_before:
            sig = f"bill_{bill.id}_{proj.due_date.isoformat()}"
            if sig not in sent_sigs:
                await dispatch_alert(
                    db,
                    subject=f"Upcoming Bill: {proj.name}",
                    body=f"Your bill for ${proj.amount:.2f} is due in {days_until_due} days on {proj.due_date}.",
                    signature=sig
                )

async def check_anomalies(db: AsyncSession) -> None:
    """Detects large single transactions, week-over-week spikes, and new recurring charges."""
    prefs = await get_alert_preferences(db)
    if not prefs.anomaly_alerts_enabled:
        return

    today = date.today()
    sent_sigs = await get_sent_signatures(db)
    
    # 1. Large Single Transactions (>2x 90-day average)
    recent_tx_result = await db.execute(
        select(Transaction).where(
            Transaction.date >= today - timedelta(days=3),
            Transaction.transaction_type == TransactionType.EXPENSE
        )
    )
    recent_txs = recent_tx_result.scalars().all()
    
    for tx in recent_txs:
        if not tx.category_id:
            continue
            
        avg_result = await db.execute(
            select(func.avg(Transaction.amount)).where(
                Transaction.category_id == tx.category_id,
                Transaction.date >= today - timedelta(days=90),
                Transaction.date < today - timedelta(days=3),
                Transaction.transaction_type == TransactionType.EXPENSE
            )
        )
        avg_amount = avg_result.scalar() or 0.0
        
        if avg_amount > 0 and tx.amount > (2 * avg_amount) and tx.amount > 50.0:
            sig = f"anomaly_large_{tx.id}"
            if sig not in sent_sigs:
                await dispatch_alert(
                    db,
                    subject="Large Transaction Detected",
                    body=f"A transaction of ${tx.amount:.2f} at {tx.merchant or 'Unknown'} is significantly higher than your average for this category.",
                    signature=sig
                )
                
    # 2. Week-over-Week Spike
    current_week_start = today - timedelta(days=7)
    prev_week_start = today - timedelta(days=14)
    
    curr_result = await db.execute(
        select(func.sum(Transaction.amount)).where(
            Transaction.date >= current_week_start,
            Transaction.transaction_type == TransactionType.EXPENSE
        )
    )
    curr_spent = curr_result.scalar() or 0.0
    
    prev_result = await db.execute(
        select(func.sum(Transaction.amount)).where(
            Transaction.date >= prev_week_start,
            Transaction.date < current_week_start,
            Transaction.transaction_type == TransactionType.EXPENSE
        )
    )
    prev_spent = prev_result.scalar() or 0.0
    
    if prev_spent > 0 and curr_spent > (1.5 * prev_spent):
        week_str = current_week_start.strftime("%Y-%W")
        sig = f"anomaly_wow_{week_str}"
        if sig not in sent_sigs:
            await dispatch_alert(
                db,
                subject="Spending Spike Detected",
                body=f"Your spending in the last 7 days (${curr_spent:.2f}) is over 50% higher than the previous 7 days.",
                signature=sig
            )

    # 3. New Recurring Charges
    new_recurring = [tx for tx in recent_txs if tx.is_recurring]
    for tx in new_recurring:
        sig = f"anomaly_newrec_{tx.id}"
        if sig not in sent_sigs:
            await dispatch_alert(
                db,
                subject="New Recurring Charge",
                body=f"A new recurring charge of ${tx.amount:.2f} was detected for {tx.merchant or 'Unknown'}.",
                signature=sig
            )

async def check_cashflow_warnings(db: AsyncSession) -> None:
    """Projects cash flow 14 days out and warns if balance drops below zero."""
    prefs = await get_alert_preferences(db)
    if not prefs.cashflow_warnings_enabled:
        return

    today = date.today()
    sent_sigs = await get_sent_signatures(db)
    
    # Calculate current liquid balance
    acc_result = await db.execute(
        select(func.sum(FinancialAccount.balance)).where(
            FinancialAccount.account_type.in_([AccountType.CHECKING, AccountType.SAVINGS, AccountType.CASH]),
            FinancialAccount.is_active == True
        )
    )
    liquid_balance = acc_result.scalar() or 0.0
    
    # Project bills
    bill_result = await db.execute(select(RecurringBill).where(RecurringBill.is_active == True))
    bills = bill_result.scalars().all()
    projections = project_bills(bills, today, today + timedelta(days=14))
    
    # aggregate_daily_forecast is a sync function
    forecast = aggregate_daily_forecast(projections, today, today + timedelta(days=14), {}, {})
    
    running_balance = liquid_balance
    for row in forecast:
        running_balance += row.net # Uses the net property
        if running_balance < 0:
            sig = f"cashflow_{row.date.isoformat()}"
            if sig not in sent_sigs:
                body = f"Projected balance drops below zero (${running_balance:.2f}) on {row.date} due to upcoming expenses."
                await dispatch_alert(db, "Cash Flow Warning", body, sig)
                
                # Store as recommendation
                rec = Recommendation(
                    recommendation_type=RecommendationType.CASHFLOW_WARNING,
                    status=RecommendationStatus.PENDING,
                    title="Prevent Negative Balance",
                    summary=body,
                    priority=1 # High priority
                )
                db.add(rec)
                await db.commit()
            break # Only one warning per run

async def generate_ai_insights(db: AsyncSession) -> None:
    """Generates a weekly financial digest using Anthropic API."""
    prefs = await get_alert_preferences(db)
    if not prefs.ai_insights_enabled or not settings.anthropic_api_key:
        return

    # Check if we already generated an insight this week
    cutoff = datetime.utcnow() - timedelta(days=7)
    recent_rec = await db.execute(
        select(Recommendation).where(
            Recommendation.recommendation_type == RecommendationType.GENERAL_INSIGHT,
            Recommendation.created_at >= cutoff
        )
    )
    if recent_rec.scalars().first():
        return # Already generated this week

    today = date.today()
    
    # Gather basic context
    acc_result = await db.execute(select(func.sum(FinancialAccount.balance)))
    total_balance = acc_result.scalar() or 0.0
    
    tx_result = await db.execute(
        select(func.sum(Transaction.amount)).where(
            Transaction.date >= today.replace(day=1),
            Transaction.transaction_type == TransactionType.EXPENSE
        )
    )
    spent_this_month = tx_result.scalar() or 0.0

    prompt = f"""
    You are an expert personal CFO. Provide a brief, encouraging weekly financial insight for the user.
    Current total balance: ${total_balance:.2f}.
    Spent this month: ${spent_this_month:.2f}.
    Keep it under 3 sentences. Focus on actionable advice or a positive observation.
    """

    headers = {
        "x-api-key": settings.anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": settings.ai_model,
        "max_tokens": 150,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            insight_text = data["content"][0]["text"]
            
            rec = Recommendation(
                recommendation_type=RecommendationType.GENERAL_INSIGHT,
                status=RecommendationStatus.PENDING,
                title="Weekly Financial Digest",
                summary=insight_text,
                priority=5 # Lower priority
            )
            db.add(rec)
            await db.commit()
            
            await dispatch_alert(
                db, 
                "Weekly AI Financial Insight", 
                insight_text, 
                f"ai_insight_{today.strftime('%Y-%W')}"
            )
    except Exception as e:
        logger.error(f"Failed to generate AI insights: {e}")
```

### 5. Background Scheduler

**`src/core/alerts/scheduler.py`**
```python
import asyncio
import logging
from src.db.engine import async_session
from src.core.alerts.rules import (
    check_budget_thresholds,
    check_bill_reminders,
    check_anomalies,
    check_cashflow_warnings,
    generate_ai_insights
)

logger = logging.getLogger(__name__)

async def run_alert_cycle():
    """Executes one full pass of all alert monitoring rules."""
    logger.info("Starting alert monitoring cycle...")
    async with async_session() as db:
        try:
            await check_budget_thresholds(db)
            await check_bill_reminders(db)
            await check_anomalies(db)
            await check_cashflow_warnings(db)
            await generate_ai_insights(db)
            logger.info("Alert monitoring cycle completed successfully.")
        except Exception as e:
            logger.error(f"Error during alert monitoring cycle: {e}")
            await db.rollback()

async def alert_scheduler_loop():
    """Infinite loop to run the alert cycle periodically (e.g., daily)."""
    while True:
        await run_alert_cycle()
        # Sleep for 24 hours (86400 seconds)
        await asyncio.sleep(86400)
```

### 6. FastAPI Router and Lifecycle Integration

**`src/api/routers/alerts.py`**
```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from typing import List

from src.api.deps import get_db
from src.models.alert_preference import AlertPreference
from src.models.notification import Notification
from src.schemas.alert_preference import AlertPreferenceResponse, AlertPreferenceUpdate
from src.core.alerts.dispatcher import get_alert_preferences

router = APIRouter(tags=["alerts"])

@router.get("/alerts/preferences", response_model=AlertPreferenceResponse)
async def get_preferences(db: AsyncSession = Depends(get_db)):
    """Retrieve current alert preferences."""
    return await get_alert_preferences(db)

@router.put("/alerts/preferences", response_model=AlertPreferenceResponse)
async def update_preferences(
    prefs_in: AlertPreferenceUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update alert preferences."""
    prefs = await get_alert_preferences(db)
    
    update_data = prefs_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(prefs, field, value)
        
    await db.commit()
    await db.refresh(prefs)
    return prefs

@router.get("/alerts/history")
async def get_alert_history(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Retrieve the log of sent notifications."""
    result = await db.execute(
        select(Notification)
        .order_by(desc(Notification.created_at))
        .limit(limit)
    )
    notifications = result.scalars().all()
    
    return [
        {
            "id": n.id,
            "channel": n.channel.value,
            "subject": n.subject,
            "body": n.body,
            "sent_ok": n.sent_ok,
            "created_at": n.created_at
        }
        for n in notifications
    ]
```

**`src/api/app.py`** (Snippet demonstrating integration)
```python
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.core.alerts.scheduler import alert_scheduler_loop
from src.api.routers import alerts

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the background alert scheduler
    scheduler_task = asyncio.create_task(alert_scheduler_loop())
    yield
    # Shutdown: Cancel the task
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="um-cfo API", lifespan=lifespan)

# Register routers without version prefix
app.include_router(alerts.router)
```

### 7. React Frontend

**`frontend/src/pages/Alerts.tsx`**
```tsx
import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Bell, Settings, History, CheckCircle, AlertTriangle } from 'lucide-react';
import { apiClient } from './client';

interface AlertPreferences {
  id: string;
  budget_alerts_enabled: boolean;
  bill_reminders_enabled: boolean;
  anomaly_alerts_enabled: boolean;
  cashflow_warnings_enabled: boolean;
  ai_insights_enabled: boolean;
  telegram_enabled: boolean;
  dashboard_enabled: boolean;
}

interface AlertHistoryItem {
  id: string;
  channel: string;
  subject: string;
  body: string;
  sent_ok: boolean;
  created_at: string;
}

export default function Alerts() {
  const [activeTab, setActiveTab] = useState<'preferences' | 'history'>('preferences');
  const queryClient = useQueryClient();

  const { data: preferences, isLoading: prefsLoading } = useQuery<AlertPreferences>({
    queryKey: ['alertPreferences'],
    queryFn: async () => {
      const res = await apiClient.get('/alerts/preferences');
      return res.data;
    }
  });

  const { data: history, isLoading: historyLoading } = useQuery<AlertHistoryItem[]>({
    queryKey: ['alertHistory'],
    queryFn: async () => {
      const res = await apiClient.get('/alerts/history');
      return res.data;
    },
    enabled: activeTab === 'history'
  });

  const updateMutation = useMutation({
    mutationFn: async (newPrefs: Partial<AlertPreferences>) => {
      const res = await apiClient.put('/alerts/preferences', newPrefs);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alertPreferences'] });
    }
  });

  const handleToggle = (key: keyof AlertPreferences) => {
    if (!preferences) return;
    updateMutation.mutate({ [key]: !preferences[key] });
  };

  if (prefsLoading) return <div className="p-8 text-center text-gray-500">Loading alerts...</div>;

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="flex items-center gap-3 mb-8">
        <Bell className="w-8 h-8 text-indigo-600" />
        <h1 className="text-3xl font-bold text-gray-900">Alerts & Notifications</h1>
      </div>

      <div className="flex border-b border-gray-200 mb-6">
        <button
          onClick={() => setActiveTab('preferences')}
          className={`flex items-center gap-2 py-3 px-6 font-medium border-b-2 transition-colors ${
            activeTab === 'preferences' ? 'border-indigo-600 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <Settings size={18} /> Preferences
        </button>
        <button
          onClick={() => setActiveTab('history')}
          className={`flex items-center gap-2 py-3 px-6 font-medium border-b-2 transition-colors ${
            activeTab === 'history' ? 'border-indigo-600 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          <History size={18} /> History
        </button>
      </div>

      {activeTab === 'preferences' && preferences && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Alert Triggers</h2>
            <div className="space-y-4">
              {[
                { key: 'budget_alerts_enabled', label: 'Budget Thresholds', desc: 'Get notified when nearing category limits.' },
                { key: 'bill_reminders_enabled', label: 'Bill Reminders', desc: 'Upcoming due date alerts.' },
                { key: 'anomaly_alerts_enabled', label: 'Anomaly Detection', desc: 'Spikes and unusual large transactions.' },
                { key: 'cashflow_warnings_enabled', label: 'Cash Flow Warnings', desc: 'Alerts when balance may drop below zero.' },
                { key: 'ai_insights_enabled', label: 'AI Weekly Digest', desc: 'Personalized weekly financial insights.' }
              ].map(({ key, label, desc }) => (
                <div key={key} className="flex items-center justify-between py-2">
                  <div>
                    <p className="font-medium text-gray-900">{label}</p>
                    <p className="text-sm text-gray-500">{desc}</p>
                  </div>
                  <button
                    onClick={() => handleToggle(key as keyof AlertPreferences)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                      preferences[key as keyof AlertPreferences] ? 'bg-indigo-600' : 'bg-gray-200'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        preferences[key as keyof AlertPreferences] ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Delivery Channels</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between py-2">
                <div>
                  <p className="font-medium text-gray-900">In-App Dashboard</p>
                  <p className="text-sm text-gray-500">Receive alerts directly in the web interface.</p>
                </div>
                <button
                  onClick={() => handleToggle('dashboard_enabled')}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    preferences.dashboard_enabled ? 'bg-indigo-600' : 'bg-gray-200'
                  }`}
                >
                  <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${preferences.dashboard_enabled ? 'translate-x-6' : 'translate-x-1'}`} />
                </button>
              </div>
              <div className="flex items-center justify-between py-2">
                <div>
                  <p className="font-medium text-gray-900">Telegram Bot</p>
                  <p className="text-sm text-gray-500">Instant notifications to your Telegram app.</p>
                </div>
                <button
                  onClick={() => handleToggle('telegram_enabled')}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    preferences.telegram_enabled ? 'bg-indigo-600' : 'bg-gray-200'
                  }`}
                >
                  <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${preferences.telegram_enabled ? 'translate-x-6' : 'translate-x-1'}`} />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'history' && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          {historyLoading ? (
            <div className="p-8 text-center text-gray-500">Loading history...</div>
          ) : history?.length === 0 ? (
            <div className="p-12 text-center text-gray-500 flex flex-col items-center">
              <CheckCircle size={48} className="text-green-400 mb-4" />
              <p className="text-lg font-medium">All caught up!</p>
              <p>No alerts have been generated yet.</p>
            </div>
          ) : (
            <ul className="divide-y divide-gray-100">
              {history?.map((alert) => (
                <li key={alert.id} className="p-6 hover:bg-gray-50 transition-colors">
                  <div className="flex justify-between items-start">
                    <div className="flex gap-3">
                      {alert.sent_ok ? (
                        <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
                      ) : (
                        <AlertTriangle className="text-red-500 mt-1 flex-shrink-0" size={20} />
                      )}
                      <div>
                        <h3 className="text-gray-900 font-medium text-lg">{alert.subject}</h3>
                        <p className="text-gray-600 mt-1 leading-relaxed">{alert.body}</p>
                        <span className="inline-block mt-2 text-xs font-semibold px-2 py-1 bg-gray-100 text-gray-600 rounded">
                          {alert.channel}
                        </span>
                      </div>
                    </div>
                    <span className="text-sm text-gray-400 whitespace-nowrap ml-4">
                      {new Date(alert.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
```