Here is the complete, production-ready implementation of the smart budget alert and notification system. It strictly adheres to your existing SQLAlchemy 2.0 patterns, correctly utilizes the `CashFlowEngine` as synchronous functions, and properly integrates with your existing models and configuration.

### 1. Database Model for Preferences

**`src/models/alert_preference.py`**
```python
from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column
from src.models.base import BaseModel

class AlertPreference(BaseModel):
    """Stores user notification and alert preferences."""
    __tablename__ = "alert_preferences"

    user_id: Mapped[str] = mapped_column(String(50), default="default", unique=True)
    telegram_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    budget_alerts: Mapped[bool] = mapped_column(Boolean, default=True)
    bill_reminders: Mapped[bool] = mapped_column(Boolean, default=True)
    cashflow_warnings: Mapped[bool] = mapped_column(Boolean, default=True)
    anomaly_detection: Mapped[bool] = mapped_column(Boolean, default=True)
    ai_insights: Mapped[bool] = mapped_column(Boolean, default=True)
```

### 2. Notification Dispatcher

**`src/services/alerts/dispatcher.py`**
```python
import httpx
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.notification import Notification, NotificationChannel
from src.config import settings

logger = logging.getLogger(__name__)

async def send_notification(
    db: AsyncSession,
    channel: NotificationChannel,
    subject: str,
    body: str,
    metadata_json: dict | None = None
):
    """Dispatches notifications and logs them using the existing Notification model."""
    sent_ok = False
    error_msg = None

    if channel == NotificationChannel.TELEGRAM:
        if getattr(settings, "telegram_bot_token", None) and getattr(settings, "telegram_chat_id", None):
            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
                    payload = {
                        "chat_id": settings.telegram_chat_id,
                        "text": f"*{subject}*\n\n{body}",
                        "parse_mode": "Markdown"
                    }
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    sent_ok = True
            except Exception as e:
                logger.error(f"Telegram send failed: {e}")
                error_msg = str(e)
        else:
            error_msg = "Telegram credentials not configured"
    elif channel == NotificationChannel.DASHBOARD:
        sent_ok = True  # Dashboard alerts are just stored in DB

    notification = Notification(
        channel=channel,
        subject=subject,
        body=body,
        metadata_json=metadata_json or {},
        sent_ok=sent_ok,
        error=error_msg
    )
    db.add(notification)
    await db.commit()
```

### 3. Alert Engine Logic

**`src/services/alerts/engine.py`**
```python
import asyncio
import logging
from datetime import date, timedelta, datetime
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import anthropic

from src.db.engine import async_session
from src.models.alert_preference import AlertPreference
from src.models.budget import Budget
from src.models.transaction import Transaction, TransactionType
from src.models.bill import RecurringBill
from src.models.account import FinancialAccount
from src.models.notification import Notification, NotificationChannel
from src.models.recommendation import Recommendation, RecommendationType
from src.core.cashflow import project_bills, aggregate_daily_forecast
from src.services.alerts.dispatcher import send_notification
from src.config import settings

logger = logging.getLogger(__name__)

async def get_preferences(db: AsyncSession) -> AlertPreference:
    result = await db.execute(select(AlertPreference).where(AlertPreference.user_id == "default"))
    pref = result.scalars().first()
    if not pref:
        pref = AlertPreference(user_id="default")
        db.add(pref)
        await db.commit()
        await db.refresh(pref)
    return pref

async def check_budget_thresholds(db: AsyncSession, prefs: AlertPreference):
    if not prefs.budget_alerts:
        return

    current_month = date.today().strftime("%Y-%m")
    start_date = date.today().replace(day=1)
    next_month = start_date.replace(day=28) + timedelta(days=4)
    end_date = next_month - timedelta(days=next_month.day)
    start_of_month_dt = datetime.combine(start_date, datetime.min.time())

    result = await db.execute(
        select(Budget).options(selectinload(Budget.category)).where(Budget.month == current_month)
    )
    budgets = result.scalars().all()

    for budget in budgets:
        subject = f"Budget Alert: {budget.category.name}"
        # Prevent spamming the same alert multiple times in a month
        existing_alert = await db.execute(
            select(Notification).where(
                Notification.subject == subject,
                Notification.created_at >= start_of_month_dt
            )
        )
        if existing_alert.scalars().first():
            continue

        tx_result = await db.execute(
            select(func.sum(Transaction.amount))
            .where(
                Transaction.category_id == budget.category_id,
                Transaction.date >= start_date,
                Transaction.date <= end_date,
                Transaction.transaction_type == TransactionType.EXPENSE
            )
        )
        spent = tx_result.scalar() or 0.0

        if budget.budgeted_amount > 0:
            ratio = spent / budget.budgeted_amount
            if ratio >= budget.alert_threshold:
                body = f"You have spent ${spent:.2f} of your ${budget.budgeted_amount:.2f} budget for {budget.category.name} ({ratio*100:.1f}%)."
                await send_notification(db, NotificationChannel.DASHBOARD, subject, body, {"budget_id": str(budget.id)})
                if prefs.telegram_enabled:
                    await send_notification(db, NotificationChannel.TELEGRAM, subject, body, {"budget_id": str(budget.id)})

async def check_bill_reminders(db: AsyncSession, prefs: AlertPreference):
    if not prefs.bill_reminders:
        return

    today = date.today()
    result = await db.execute(select(RecurringBill).where(RecurringBill.is_active == True))
    bills = result.scalars().all()

    for bill in bills:
        if bill.next_due_date:
            days_until_due = (bill.next_due_date - today).days
            if days_until_due == bill.alert_days_before:
                subject = f"Bill Reminder: {bill.name}"
                body = f"Your bill '{bill.name}' for ${bill.amount:.2f} is due on {bill.next_due_date} ({days_until_due} days)."
                
                await send_notification(db, NotificationChannel.DASHBOARD, subject, body, {"bill_id": str(bill.id)})
                if prefs.telegram_enabled:
                    await send_notification(db, NotificationChannel.TELEGRAM, subject, body, {"bill_id": str(bill.id)})

async def check_cashflow_warnings(db: AsyncSession, prefs: AlertPreference):
    if not prefs.cashflow_warnings:
        return

    today = date.today()
    today_dt = datetime.combine(today, datetime.min.time())

    # Limit to one warning per day
    existing_warning = await db.execute(
        select(Recommendation).where(
            Recommendation.recommendation_type == RecommendationType.CASHFLOW_WARNING,
            Recommendation.created_at >= today_dt
        )
    )
    if existing_warning.scalars().first():
        return

    end_date = today + timedelta(days=14)
    acc_result = await db.execute(select(func.sum(FinancialAccount.balance)).where(FinancialAccount.is_active == True))
    current_balance = acc_result.scalar() or 0.0

    bills_result = await db.execute(select(RecurringBill).where(RecurringBill.is_active == True))
    bills = bills_result.scalars().all()

    # CashFlowEngine functions are plain defs, NOT async
    projections = project_bills(list(bills), today, end_date)
    forecast = aggregate_daily_forecast(projections, today, end_date)

    running_balance = current_balance
    for day in forecast:
        running_balance += day.actual_income - day.actual_expenses - day.projected_expenses
        if running_balance < 0:
            subject = "Cash Flow Warning"
            body = f"Your projected balance may drop below zero on {day.date}."
            
            rec = Recommendation(
                recommendation_type=RecommendationType.CASHFLOW_WARNING,
                title=subject,
                summary=body,
                priority=1
            )
            db.add(rec)
            
            await send_notification(db, NotificationChannel.DASHBOARD, subject, body)
            if prefs.telegram_enabled:
                await send_notification(db, NotificationChannel.TELEGRAM, subject, body)
            break

async def check_anomalies(db: AsyncSession, prefs: AlertPreference):
    if not prefs.anomaly_detection:
        return

    today = date.today()
    start_of_week = today - timedelta(days=today.weekday())
    start_of_week_dt = datetime.combine(start_of_week, datetime.min.time())
    one_day_ago = datetime.utcnow() - timedelta(days=1)

    # 1. Large single transactions (>2x average)
    thirty_days_ago = today - timedelta(days=30)
    recent_txs = await db.execute(
        select(Transaction).where(
            Transaction.created_at >= one_day_ago,
            Transaction.transaction_type == TransactionType.EXPENSE,
            Transaction.category_id.isnot(None)
        )
    )
    for tx in recent_txs.scalars().all():
        avg_result = await db.execute(
            select(func.avg(Transaction.amount)).where(
                Transaction.category_id == tx.category_id,
                Transaction.date >= thirty_days_ago,
                Transaction.date < tx.date,
                Transaction.transaction_type == TransactionType.EXPENSE
            )
        )
        avg_amount = avg_result.scalar() or 0.0
        if avg_amount > 0 and tx.amount > (avg_amount * 2):
            subject = "Large Transaction Detected"
            body = f"A transaction of ${tx.amount:.2f} at {tx.merchant or 'Unknown'} is more than twice your average for this category."
            
            rec = Recommendation(
                recommendation_type=RecommendationType.SPENDING_ALERT,
                title=subject,
                summary=body,
                priority=2,
                impact_amount=tx.amount
            )
            db.add(rec)
            await send_notification(db, NotificationChannel.DASHBOARD, subject, body)
            if prefs.telegram_enabled:
                await send_notification(db, NotificationChannel.TELEGRAM, subject, body)

    # 2. Sudden spending spikes (week-over-week)
    existing_spike = await db.execute(
        select(Recommendation).where(
            Recommendation.recommendation_type == RecommendationType.SPENDING_ALERT,
            Recommendation.title == "Spending Spike Detected",
            Recommendation.created_at >= start_of_week_dt
        )
    )
    if not existing_spike.scalars().first():
        one_week_ago = today - timedelta(days=7)
        two_weeks_ago = today - timedelta(days=14)

        this_week_res = await db.execute(
            select(func.sum(Transaction.amount)).where(
                Transaction.date >= one_week_ago,
                Transaction.transaction_type == TransactionType.EXPENSE
            )
        )
        this_week_spent = this_week_res.scalar() or 0.0

        last_week_res = await db.execute(
            select(func.sum(Transaction.amount)).where(
                Transaction.date >= two_weeks_ago,
                Transaction.date < one_week_ago,
                Transaction.transaction_type == TransactionType.EXPENSE
            )
        )
        last_week_spent = last_week_res.scalar() or 0.0

        if last_week_spent > 0 and this_week_spent > (last_week_spent * 1.5):
            subject = "Spending Spike Detected"
            body = f"Your spending this week (${this_week_spent:.2f}) is significantly higher than last week (${last_week_spent:.2f})."
            
            rec = Recommendation(
                recommendation_type=RecommendationType.SPENDING_ALERT,
                title=subject,
                summary=body,
                priority=2
            )
            db.add(rec)
            await send_notification(db, NotificationChannel.DASHBOARD, subject, body)
            if prefs.telegram_enabled:
                await send_notification(db, NotificationChannel.TELEGRAM, subject, body)

    # 3. New recurring charges
    recent_recurring = await db.execute(
        select(Transaction).where(
            Transaction.created_at >= one_day_ago,
            Transaction.is_recurring == True,
            Transaction.transaction_type == TransactionType.EXPENSE
        )
    )
    for tx in recent_recurring.scalars().all():
        bill_res = await db.execute(
            select(RecurringBill).where(
                (RecurringBill.name.ilike(f"%{tx.merchant or tx.description}%")) | 
                (RecurringBill.merchant.ilike(f"%{tx.merchant or tx.description}%"))
            )
        )
        if not bill_res.scalars().first():
            subject = "New Recurring Charge Detected"
            body = f"We noticed a new recurring charge of ${tx.amount:.2f} for {tx.merchant or tx.description}."
            
            rec = Recommendation(
                recommendation_type=RecommendationType.BILL_ALERT,
                title=subject,
                summary=body,
                priority=2
            )
            db.add(rec)
            await send_notification(db, NotificationChannel.DASHBOARD, subject, body)
            if prefs.telegram_enabled:
                await send_notification(db, NotificationChannel.TELEGRAM, subject, body)

async def generate_ai_insights(db: AsyncSession, prefs: AlertPreference):
    if not prefs.ai_insights or not getattr(settings, "anthropic_api_key", None):
        return

    today = date.today()
    start_of_week = today - timedelta(days=today.weekday())
    start_of_week_dt = datetime.combine(start_of_week, datetime.min.time())

    existing = await db.execute(
        select(Recommendation).where(
            Recommendation.recommendation_type == RecommendationType.GENERAL_INSIGHT,
            Recommendation.created_at >= start_of_week_dt
        )
    )
    if existing.scalars().first():
        return

    thirty_days_ago = today - timedelta(days=30)
    tx_result = await db.execute(
        select(Transaction).where(Transaction.date >= thirty_days_ago)
    )
    transactions = tx_result.scalars().all()
    
    total_spent = sum(t.amount for t in transactions if t.transaction_type == TransactionType.EXPENSE)
    total_income = sum(t.amount for t in transactions if t.transaction_type == TransactionType.INCOME)

    prompt = f"""
    Analyze the following financial summary for the last 30 days and provide a short, actionable insight.
    Total Income: ${total_income:.2f}
    Total Expenses: ${total_spent:.2f}
    
    Format your response exactly like this:
    Title: [Your Title]
    Summary: [Your 2-sentence Summary]
    """

    try:
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        response = await client.messages.create(
            model=settings.ai_model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        ai_text = response.content[0].text
        
        title = "Weekly AI Financial Insight"
        summary = ai_text
        for line in ai_text.split('\n'):
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("Summary:"):
                summary = line.replace("Summary:", "").strip()

        rec = Recommendation(
            recommendation_type=RecommendationType.GENERAL_INSIGHT,
            title=title,
            summary=summary,
            priority=3
        )
        db.add(rec)
        await send_notification(db, NotificationChannel.DASHBOARD, title, summary)
        if prefs.telegram_enabled:
            await send_notification(db, NotificationChannel.TELEGRAM, title, summary)

    except Exception as e:
        logger.error(f"AI Insight generation failed: {e}")

async def run_alert_checks():
    async with async_session() as db:
        try:
            prefs = await get_preferences(db)
            await check_budget_thresholds(db, prefs)
            await check_bill_reminders(db, prefs)
            await check_cashflow_warnings(db, prefs)
            await check_anomalies(db, prefs)
            await generate_ai_insights(db, prefs)
        except Exception as e:
            logger.error(f"Error running alert checks: {e}")

async def start_alert_engine():
    """Background task to run checks periodically. Add this to FastAPI lifespan."""
    while True:
        try:
            await run_alert_checks()
        except Exception as e:
            logger.error(f"Alert engine crashed: {e}")
        await asyncio.sleep(86400) # Run daily
```

### 4. API Layer

**`src/api/routers/alerts.py`**
```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel as PydanticBaseModel

from src.api.deps import get_db
from src.models.alert_preference import AlertPreference
from src.models.notification import Notification, NotificationChannel

router = APIRouter(prefix="/alerts", tags=["alerts"])

class AlertPreferenceUpdate(PydanticBaseModel):
    telegram_enabled: bool | None = None
    budget_alerts: bool | None = None
    bill_reminders: bool | None = None
    cashflow_warnings: bool | None = None
    anomaly_detection: bool | None = None
    ai_insights: bool | None = None

class AlertPreferenceResponse(PydanticBaseModel):
    id: UUID
    user_id: str
    telegram_enabled: bool
    budget_alerts: bool
    bill_reminders: bool
    cashflow_warnings: bool
    anomaly_detection: bool
    ai_insights: bool

    model_config = {"from_attributes": True}

class NotificationResponse(PydanticBaseModel):
    id: UUID
    channel: NotificationChannel
    subject: str
    body: str
    sent_ok: bool
    created_at: datetime

    model_config = {"from_attributes": True}

@router.get("/preferences", response_model=AlertPreferenceResponse)
async def get_alert_preferences(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AlertPreference).where(AlertPreference.user_id == "default"))
    pref = result.scalars().first()
    if not pref:
        pref = AlertPreference(user_id="default")
        db.add(pref)
        await db.commit()
        await db.refresh(pref)
    return pref

@router.put("/preferences", response_model=AlertPreferenceResponse)
async def update_alert_preferences(
    prefs_in: AlertPreferenceUpdate,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(AlertPreference).where(AlertPreference.user_id == "default"))
    pref = result.scalars().first()
    if not pref:
        pref = AlertPreference(user_id="default")
        db.add(pref)

    update_data = prefs_in.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(pref, key, value)
    
    await db.commit()
    await db.refresh(pref)
    return pref

@router.get("/history", response_model=List[NotificationResponse])
async def get_alert_history(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Notification)
        .order_by(Notification.created_at.desc())
        .limit(50)
    )
    return result.scalars().all()
```

### 5. Frontend UI

**`src/frontend/src/pages/Alerts.tsx`**
```tsx
import React, { useState, useEffect } from 'react';
import { apiClient } from '../client';

interface AlertPrefs {
  id: string;
  telegram_enabled: boolean;
  budget_alerts: boolean;
  bill_reminders: boolean;
  cashflow_warnings: boolean;
  anomaly_detection: boolean;
  ai_insights: boolean;
}

interface NotificationLog {
  id: string;
  channel: string;
  subject: string;
  body: string;
  sent_ok: boolean;
  created_at: string;
}

export const AlertsPage: React.FC = () => {
  const [prefs, setPrefs] = useState<AlertPrefs | null>(null);
  const [history, setHistory] = useState<NotificationLog[]>([]);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const prefsRes = await apiClient.get('/alerts/preferences');
      setPrefs(prefsRes.data);
      const histRes = await apiClient.get('/alerts/history');
      setHistory(histRes.data);
    } catch (error) {
      console.error('Failed to fetch alerts data', error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prefs) return;
    setSaving(true);
    setMessage('');
    try {
      const updateData = {
        telegram_enabled: prefs.telegram_enabled,
        budget_alerts: prefs.budget_alerts,
        bill_reminders: prefs.bill_reminders,
        cashflow_warnings: prefs.cashflow_warnings,
        anomaly_detection: prefs.anomaly_detection,
        ai_insights: prefs.ai_insights,
      };
      await apiClient.put('/alerts/preferences', updateData);
      setMessage('Preferences saved successfully!');
    } catch (error) {
      setMessage('Failed to save preferences.');
    } finally {
      setSaving(false);
    }
  };

  if (!prefs) return <div className="p-6 text-gray-600">Loading...</div>;

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <h1 className="text-2xl font-bold text-gray-900">Alerts & Notifications</h1>

      {message && (
        <div className={`p-4 rounded ${message.includes('success') ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
          {message}
        </div>
      )}

      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Notification Preferences</h2>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-4 border-b pb-6">
            <h3 className="text-lg font-medium text-gray-700">Delivery Channels</h3>
            <label className="flex items-center space-x-3">
              <input 
                type="checkbox" 
                checked={prefs.telegram_enabled}
                onChange={e => setPrefs({...prefs, telegram_enabled: e.target.checked})}
                className="h-4 w-4 text-blue-600 rounded"
              />
              <span className="text-gray-700">Enable Telegram Alerts</span>
            </label>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-700">Alert Types</h3>
            {[
              { key: 'budget_alerts', label: 'Budget Threshold Alerts' },
              { key: 'bill_reminders', label: 'Bill Due Date Reminders' },
              { key: 'cashflow_warnings', label: '14-Day Cash Flow Warnings' },
              { key: 'anomaly_detection', label: 'Anomaly Detection (Spikes, Large & New Charges)' },
              { key: 'ai_insights', label: 'Weekly AI-Powered Financial Insights' },
            ].map(({ key, label }) => (
              <label key={key} className="flex items-center space-x-3">
                <input 
                  type="checkbox" 
                  checked={prefs[key as keyof AlertPrefs] as boolean}
                  onChange={e => setPrefs({...prefs, [key]: e.target.checked})}
                  className="h-4 w-4 text-blue-600 rounded"
                />
                <span className="text-gray-700">{label}</span>
              </label>
            ))}
          </div>

          <button 
            type="submit" 
            disabled={saving}
            className="w-full sm:w-auto bg-blue-600 text-white font-bold py-2 px-6 rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save Preferences'}
          </button>
        </form>
      </div>

      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Recent Alerts</h2>
        {history.length === 0 ? (
          <p className="text-gray-500">No alerts found.</p>
        ) : (
          <div className="space-y-4">
            {history.map(log => (
              <div key={log.id} className="border p-4 rounded-lg flex flex-col sm:flex-row sm:justify-between sm:items-center">
                <div>
                  <h4 className="font-semibold text-gray-800">{log.subject}</h4>
                  <p className="text-sm text-gray-600 mt-1">{log.body}</p>
                </div>
                <div className="mt-2 sm:mt-0 text-sm text-gray-500 flex flex-col items-end">
                  <span>{new Date(log.created_at).toLocaleString()}</span>
                  <span className={`mt-1 px-2 py-1 rounded text-xs ${log.sent_ok ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                    {log.channel} {log.sent_ok ? 'Sent' : 'Failed'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
```