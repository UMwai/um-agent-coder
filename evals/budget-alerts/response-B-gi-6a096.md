Here is the complete, production-ready implementation of the smart budget alert and notification system. 

### Overview of Approach
1. **Persistent Preferences**: Added an `AlertPreference` database model to store user notification settings persistently, replacing in-memory stubs.
2. **Robust Deduplication**: The notification engine uses the existing `Notification.metadata_json` field to store a `dedup_key`. Before dispatching, it checks this key to prevent duplicate alerts.
3. **Comprehensive Anomaly Detection**: Fully implements all 3 required anomaly rules: large single transactions (>2x average), week-over-week spending spikes (>1.5x), and new recurring charge detection (identifying repeated merchant patterns).
4. **Accurate Bill Frequencies**: Includes a date-math helper to advance past `next_due_date` values according to their specific `BillFrequency` before evaluating the `alert_days_before` threshold.
5. **Proper Async/Sync Boundaries**: Strictly uses SQLAlchemy 2.0 async patterns for DB calls, while correctly calling the `CashFlowEngine` functions synchronously.

### Dependencies
Ensure the following dependencies are installed in your environment:
- **Backend**: `fastapi`, `sqlalchemy`, `httpx` (for Anthropic & Telegram API calls)
- **Frontend**: `@tanstack/react-query`, `lucide-react`, `recharts`, `react-router-dom`

---

### 1. Database Models & Schemas

**`src/models/alert_preference.py`**
```python
from sqlalchemy import Boolean, Column
from src.models.base import BaseModel

class AlertPreference(BaseModel):
    __tablename__ = "alert_preferences"

    telegram_enabled = Column(Boolean, default=True, nullable=False)
    dashboard_enabled = Column(Boolean, default=True, nullable=False)
```

**`src/schemas/alerts.py`**
```python
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional
from uuid import UUID
from src.models.notification import NotificationChannel

class AlertPreferenceCreate(BaseModel):
    telegram_enabled: bool = True
    dashboard_enabled: bool = True

class AlertPreferenceUpdate(BaseModel):
    telegram_enabled: Optional[bool] = None
    dashboard_enabled: Optional[bool] = None

class AlertPreferenceResponse(BaseModel):
    id: UUID
    telegram_enabled: bool
    dashboard_enabled: bool

    model_config = ConfigDict(from_attributes=True)

class NotificationResponse(BaseModel):
    id: UUID
    channel: NotificationChannel
    subject: str
    body: str
    sent_ok: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
```

---

### 2. Core Alert & Notification Engines

**`src/engine/notifier.py`**
Handles delivery, preference checking, and deduplication.
```python
import httpx
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.notification import Notification, NotificationChannel
from src.models.alert_preference import AlertPreference
from src.config import settings

logger = logging.getLogger(__name__)

async def get_or_create_preferences(db: AsyncSession) -> AlertPreference:
    result = await db.execute(select(AlertPreference))
    prefs = result.scalars().first()
    if not prefs:
        prefs = AlertPreference(telegram_enabled=True, dashboard_enabled=True)
        db.add(prefs)
        await db.commit()
        await db.refresh(prefs)
    return prefs

async def dispatch_alert(
    db: AsyncSession, 
    subject: str, 
    body: str, 
    channels: list[NotificationChannel], 
    dedup_key: str
) -> None:
    """
    Dispatches alerts respecting user preferences and prevents duplicates via dedup_key.
    """
    prefs = await get_or_create_preferences(db)

    for channel in channels:
        # 1. Check Preferences
        if channel == NotificationChannel.TELEGRAM and not prefs.telegram_enabled:
            continue
        if channel == NotificationChannel.DASHBOARD and not prefs.dashboard_enabled:
            continue

        # 2. Check Deduplication
        stmt = select(Notification).where(
            Notification.channel == channel,
            Notification.metadata_json['dedup_key'].astext == dedup_key
        )
        existing = (await db.execute(stmt)).scalars().first()
        if existing:
            logger.debug(f"Alert '{dedup_key}' already sent to {channel.name}. Skipping.")
            continue

        # 3. Dispatch
        sent_ok = True
        error_msg = None

        if channel == NotificationChannel.TELEGRAM:
            if not settings.telegram_bot_token or not settings.telegram_chat_id:
                sent_ok = False
                error_msg = "Telegram settings missing"
            else:
                url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
                payload = {"chat_id": settings.telegram_chat_id, "text": f"*{subject}*\n\n{body}", "parse_mode": "Markdown"}
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(url, json=payload, timeout=10.0)
                        resp.raise_for_status()
                except Exception as e:
                    sent_ok = False
                    error_msg = str(e)
                    logger.error(f"Telegram delivery failed: {e}")

        elif channel == NotificationChannel.DASHBOARD:
            # Dashboard notifications are "delivered" by simply existing in the DB
            sent_ok = True

        # 4. Log Notification
        notification = Notification(
            channel=channel,
            subject=subject,
            body=body,
            metadata_json={"dedup_key": dedup_key},
            sent_ok=sent_ok,
            error=error_msg
        )
        db.add(notification)
    
    await db.commit()
```

**`src/engine/analyzer.py`**
Contains the core business logic for all 5 capabilities.
```python
import logging
import httpx
from datetime import date, timedelta
from collections import defaultdict
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.budget import Budget
from src.models.transaction import Transaction, TransactionType
from src.models.bill import RecurringBill, BillFrequency
from src.models.account import FinancialAccount
from src.models.recommendation import Recommendation, RecommendationType, RecommendationStatus
from src.models.notification import NotificationChannel
from src.core.cashflow import project_bills, aggregate_daily_forecast
from src.engine.notifier import dispatch_alert
from src.config import settings

logger = logging.getLogger(__name__)

def add_months(sourcedate: date, months: int) -> date:
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, [31, 29 if year%4==0 and not year%400==0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month-1])
    return date(year, month, day)

def advance_due_date(due_date: date, frequency: BillFrequency, today: date) -> date:
    """Advances past due dates to the next upcoming occurrence based on frequency."""
    while due_date < today:
        if frequency == BillFrequency.WEEKLY: due_date += timedelta(days=7)
        elif frequency == BillFrequency.BIWEEKLY: due_date += timedelta(days=14)
        elif frequency == BillFrequency.MONTHLY: due_date = add_months(due_date, 1)
        elif frequency == BillFrequency.QUARTERLY: due_date = add_months(due_date, 3)
        elif frequency == BillFrequency.SEMIANNUAL: due_date = add_months(due_date, 6)
        elif frequency == BillFrequency.ANNUAL: due_date = add_months(due_date, 12)
        else: break
    return due_date

async def check_budget_thresholds(db: AsyncSession):
    today = date.today()
    current_month = today.strftime("%Y-%m")
    
    budgets = (await db.execute(select(Budget).where(Budget.month == current_month))).scalars().all()
    for budget in budgets:
        stmt = select(func.sum(Transaction.amount)).where(
            Transaction.category_id == budget.category_id,
            func.to_char(Transaction.date, 'YYYY-MM') == current_month,
            Transaction.transaction_type == TransactionType.EXPENSE
        )
        spent = (await db.execute(stmt)).scalar() or 0.0
        
        limit = budget.budgeted_amount * budget.alert_threshold
        if spent >= limit:
            pct = int((spent / budget.budgeted_amount) * 100)
            await dispatch_alert(
                db,
                subject="Budget Warning",
                body=f"You have spent ${spent:.2f} ({pct}%) of your ${budget.budgeted_amount:.2f} budget.",
                channels=[NotificationChannel.DASHBOARD, NotificationChannel.TELEGRAM],
                dedup_key=f"budget_warn_{budget.id}_{current_month}"
            )

async def check_bill_reminders(db: AsyncSession):
    today = date.today()
    bills = (await db.execute(select(RecurringBill).where(RecurringBill.is_active == True))).scalars().all()
    
    for bill in bills:
        # Correctly handle frequencies
        real_due_date = advance_due_date(bill.next_due_date, bill.frequency, today)
        days_until = (real_due_date - today).days
        
        if 0 <= days_until <= bill.alert_days_before:
            await dispatch_alert(
                db,
                subject="Upcoming Bill Reminder",
                body=f"Your bill '{bill.name}' for ${bill.amount:.2f} is due on {real_due_date}.",
                channels=[NotificationChannel.DASHBOARD, NotificationChannel.TELEGRAM],
                dedup_key=f"bill_due_{bill.id}_{real_due_date}"
            )

async def check_anomalies(db: AsyncSession):
    today = date.today()
    
    # 1. Large Single Transactions (>2x avg)
    recent_txs = (await db.execute(select(Transaction).where(
        Transaction.date >= today - timedelta(days=3),
        Transaction.transaction_type == TransactionType.EXPENSE
    ))).scalars().all()
    
    for tx in recent_txs:
        if not tx.category_id: continue
        avg_stmt = select(func.avg(Transaction.amount)).where(
            Transaction.category_id == tx.category_id,
            Transaction.date < tx.date
        )
        avg_amt = (await db.execute(avg_stmt)).scalar() or 0.0
        if avg_amt > 0 and tx.amount > (avg_amt * 2):
            await dispatch_alert(
                db, subject="Large Transaction Alert",
                body=f"Unusually large transaction: ${tx.amount:.2f} at {tx.merchant or 'Unknown'}.",
                channels=[NotificationChannel.DASHBOARD],
                dedup_key=f"anomaly_large_{tx.id}"
            )

    # 2. Week-over-Week Spending Spikes
    last_7_days = today - timedelta(days=7)
    prev_7_days = today - timedelta(days=14)
    
    stmt_last = select(func.sum(Transaction.amount)).where(
        Transaction.date >= last_7_days, Transaction.transaction_type == TransactionType.EXPENSE)
    stmt_prev = select(func.sum(Transaction.amount)).where(
        Transaction.date >= prev_7_days, Transaction.date < last_7_days, Transaction.transaction_type == TransactionType.EXPENSE)
    
    last_sum = (await db.execute(stmt_last)).scalar() or 0.0
    prev_sum = (await db.execute(stmt_prev)).scalar() or 0.0
    
    if prev_sum > 100 and last_sum > (prev_sum * 1.5):
        await dispatch_alert(
            db, subject="Spending Spike Detected",
            body=f"Your spending this week (${last_sum:.2f}) is significantly higher than last week (${prev_sum:.2f}).",
            channels=[NotificationChannel.DASHBOARD, NotificationChannel.TELEGRAM],
            dedup_key=f"anomaly_wow_spike_{last_7_days}"
        )

    # 3. New Recurring Charges
    stmt_60d = select(Transaction).where(
        Transaction.date >= today - timedelta(days=60),
        Transaction.transaction_type == TransactionType.EXPENSE,
        Transaction.is_recurring == False
    )
    txs_60d = (await db.execute(stmt_60d)).scalars().all()
    
    merchants = defaultdict(list)
    for t in txs_60d:
        if t.merchant: merchants[t.merchant].append(t)
        
    known_bills = (await db.execute(select(RecurringBill.merchant))).scalars().all()
    known_merchants = {m for m in known_bills if m}
    
    for merchant, m_txs in merchants.items():
        if merchant in known_merchants or len(m_txs) < 2: continue
        m_txs.sort(key=lambda x: x.date)
        
        # Check if approx 1 month apart and similar amounts
        days_diff = (m_txs[-1].date - m_txs[-2].date).days
        amt_diff = abs(m_txs[-1].amount - m_txs[-2].amount)
        
        if 25 <= days_diff <= 35 and amt_diff < 10.0:
            await dispatch_alert(
                db, subject="New Recurring Charge Detected",
                body=f"We noticed repeated charges from {merchant}. Is this a new subscription?",
                channels=[NotificationChannel.DASHBOARD],
                dedup_key=f"anomaly_new_sub_{merchant}_{today.strftime('%Y-%m')}"
            )

async def check_cashflow_warnings(db: AsyncSession):
    today = date.today()
    end_date = today + timedelta(days=14)
    
    bills = (await db.execute(select(RecurringBill).where(RecurringBill.is_active == True))).scalars().all()
    accounts = (await db.execute(select(FinancialAccount).where(FinancialAccount.is_active == True))).scalars().all()
    
    current_balance = sum(a.balance for a in accounts)
    
    # CashFlowEngine functions are strictly sync
    projections = project_bills(list(bills), today, end_date)
    forecast = aggregate_daily_forecast(projections, today, end_date)
    
    running_balance = current_balance
    for day in forecast:
        running_balance += day.net
        if running_balance < 0:
            await dispatch_alert(
                db, subject="Cash Flow Warning",
                body=f"Projected balance drops below zero (${running_balance:.2f}) on {day.date}.",
                channels=[NotificationChannel.DASHBOARD, NotificationChannel.TELEGRAM],
                dedup_key=f"cashflow_warn_{day.date}"
            )
            
            # Create a recommendation with high priority (lower number)
            rec_stmt = select(Recommendation).where(Recommendation.title == f"Cash Flow Warning: {day.date}")
            if not (await db.execute(rec_stmt)).scalars().first():
                rec = Recommendation(
                    recommendation_type=RecommendationType.CASHFLOW_WARNING,
                    status=RecommendationStatus.PENDING,
                    title=f"Cash Flow Warning: {day.date}",
                    summary="Projected negative balance detected.",
                    detail=f"Your balance is projected to be ${running_balance:.2f} due to upcoming bills.",
                    priority=1
                )
                db.add(rec)
                await db.commit()
            break

async def generate_ai_insights(db: AsyncSession):
    if not settings.anthropic_api_key:
        logger.warning("No Anthropic API key found. Skipping AI insights.")
        return

    today = date.today()
    dedup_key = f"ai_insight_{today.strftime('%Y-%W')}"
    
    stmt = select(Recommendation).where(Recommendation.title.like(f"Weekly Digest: Week of {today.strftime('%Y-%W')}"))
    if (await db.execute(stmt)).scalars().first():
        return # Already generated this week

    tx_stmt = select(Transaction).where(Transaction.date >= today - timedelta(days=7))
    txs = (await db.execute(tx_stmt)).scalars().all()
    tx_summary = "\n".join([f"{t.date}: {t.merchant} - ${t.amount:.2f}" for t in txs])

    prompt = f"Analyze these transactions and provide a 2-sentence financial insight:\n{tx_summary}"
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": settings.anthropic_api_key, "anthropic-version": "2023-06-01"},
                json={"model": settings.ai_model, "max_tokens": 300, "messages": [{"role": "user", "content": prompt}]},
                timeout=15.0
            )
            resp.raise_for_status()
            ai_text = resp.json()["content"][0]["text"]
            
            rec = Recommendation(
                recommendation_type=RecommendationType.GENERAL_INSIGHT,
                status=RecommendationStatus.PENDING,
                title=f"Weekly Digest: Week of {today.strftime('%Y-%W')}",
                summary="AI-Powered Financial Insight",
                detail=ai_text,
                priority=5
            )
            db.add(rec)
            await db.commit()
            
            await dispatch_alert(
                db, subject="New AI Financial Insight",
                body=ai_text,
                channels=[NotificationChannel.DASHBOARD],
                dedup_key=dedup_key
            )
    except Exception as e:
        logger.error(f"AI generation failed: {e}")
```

**`src/engine/tasks.py`**
Runnable background task.
```python
import asyncio
import logging
from src.api.deps import get_db
from src.engine.analyzer import (
    check_budget_thresholds,
    check_bill_reminders,
    check_anomalies,
    check_cashflow_warnings,
    generate_ai_insights
)

logger = logging.getLogger(__name__)

async def run_alert_engine():
    """Background task that runs every 12 hours."""
    while True:
        logger.info("Starting alert engine cycle...")
        try:
            # Using the existing dependency generator manually for the background task
            db_gen = get_db()
            db = await anext(db_gen)
            try:
                await check_budget_thresholds(db)
                await check_bill_reminders(db)
                await check_anomalies(db)
                await check_cashflow_warnings(db)
                await generate_ai_insights(db)
            finally:
                try:
                    await anext(db_gen)
                except StopAsyncIteration:
                    pass
        except Exception as e:
            logger.error(f"Critical error in alert engine: {e}")
        
        await asyncio.sleep(43200) # 12 hours
```

---

### 3. API Router

**`src/api/routers/alerts.py`**
```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps import get_db
from src.models.notification import Notification
from src.models.alert_preference import AlertPreference
from src.schemas.alerts import AlertPreferenceUpdate, AlertPreferenceResponse, NotificationResponse
from src.engine.notifier import get_or_create_preferences

router = APIRouter(prefix="/alerts", tags=["alerts"])

@router.get("/preferences", response_model=AlertPreferenceResponse)
async def get_preferences(db: AsyncSession = Depends(get_db)):
    return await get_or_create_preferences(db)

@router.put("/preferences", response_model=AlertPreferenceResponse)
async def update_preferences(
    prefs_in: AlertPreferenceUpdate,
    db: AsyncSession = Depends(get_db)
):
    prefs = await get_or_create_preferences(db)
    
    if prefs_in.telegram_enabled is not None:
        prefs.telegram_enabled = prefs_in.telegram_enabled
    if prefs_in.dashboard_enabled is not None:
        prefs.dashboard_enabled = prefs_in.dashboard_enabled
        
    await db.commit()
    await db.refresh(prefs)
    return prefs

@router.get("/history", response_model=list[NotificationResponse])
async def get_alert_history(limit: int = 50, db: AsyncSession = Depends(get_db)):
    stmt = select(Notification).order_by(Notification.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()
```

---

### 4. React Frontend

**`frontend/src/pages/Alerts.tsx`**
Includes loading states and proper API client usage.
```tsx
import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Bell, Settings, AlertCircle, CheckCircle } from 'lucide-react';
import { apiClient } from './client';

interface AlertPreference {
  id: string;
  telegram_enabled: boolean;
  dashboard_enabled: boolean;
}

interface Notification {
  id: string;
  channel: string;
  subject: string;
  body: string;
  sent_ok: boolean;
  created_at: string;
}

export default function AlertsPage() {
  const queryClient = useQueryClient();

  const { data: prefs, isLoading: prefsLoading } = useQuery<AlertPreference>({
    queryKey: ['alertPreferences'],
    queryFn: () => apiClient.get('/alerts/preferences').then(res => res.data),
  });

  const { data: history, isLoading: historyLoading } = useQuery<Notification[]>({
    queryKey: ['alertHistory'],
    queryFn: () => apiClient.get('/alerts/history').then(res => res.data),
  });

  const updatePrefsMutation = useMutation({
    mutationFn: (newPrefs: Partial<AlertPreference>) => 
      apiClient.put('/alerts/preferences', newPrefs).then(res => res.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alertPreferences'] });
    },
  });

  const togglePreference = (key: keyof AlertPreference) => {
    if (!prefs) return;
    updatePrefsMutation.mutate({ [key]: !prefs[key] });
  };

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
          <Bell className="w-8 h-8 text-blue-600" />
          Alerts & Notifications
        </h1>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800 flex items-center gap-2">
          <Settings className="w-5 h-5 text-gray-500" />
          Notification Preferences
        </h2>
        
        {prefsLoading ? (
          <div className="animate-pulse flex space-x-4">
            <div className="h-10 bg-gray-200 rounded w-full"></div>
          </div>
        ) : (
          <div className="space-y-4 max-w-md">
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <span className="font-medium text-gray-700">Dashboard Alerts</span>
              <button
                onClick={() => togglePreference('dashboard_enabled')}
                disabled={updatePrefsMutation.isPending}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  prefs?.dashboard_enabled ? 'bg-blue-600' : 'bg-gray-300'
                }`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  prefs?.dashboard_enabled ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <span className="font-medium text-gray-700">Telegram Bot</span>
              <button
                onClick={() => togglePreference('telegram_enabled')}
                disabled={updatePrefsMutation.isPending}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  prefs?.telegram_enabled ? 'bg-blue-600' : 'bg-gray-300'
                }`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  prefs?.telegram_enabled ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>
          </div>
        )}
      </div>

      <div>
        <h2 className="text-xl font-semibold mb-4 text-gray-800">Recent Alerts</h2>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 divide-y divide-gray-100">
          {historyLoading ? (
            <div className="p-6 text-center text-gray-500">Loading history...</div>
          ) : !history || history.length === 0 ? (
            <div className="p-6 text-center text-gray-500">No recent alerts found.</div>
          ) : (
            history.map((alert) => (
              <div key={alert.id} className="p-4 hover:bg-gray-50 transition-colors">
                <div className="flex justify-between items-start gap-4">
                  <div className="flex items-start gap-3">
                    {alert.sent_ok ? (
                      <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                    ) : (
                      <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
                    )}
                    <div>
                      <h3 className="text-gray-900 font-medium">{alert.subject}</h3>
                      <p className="text-sm text-gray-600 mt-1">{alert.body}</p>
                      <div className="flex items-center gap-2 mt-2">
                        <span className="text-xs font-medium px-2 py-1 bg-gray-100 rounded text-gray-600">
                          {alert.channel}
                        </span>
                        {!alert.sent_ok && (
                          <span className="text-xs font-medium px-2 py-1 bg-red-100 text-red-700 rounded">
                            Delivery Failed
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <span className="text-xs text-gray-400 whitespace-nowrap">
                    {new Date(alert.created_at).toLocaleString()}
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
```