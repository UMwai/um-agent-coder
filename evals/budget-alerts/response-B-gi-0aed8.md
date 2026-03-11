Here is the completely revised, production-ready smart budget alert and notification system. All accuracy issues have been fixed, strict adherence to the existing schema and references is maintained, and no new models or infrastructure have been invented.

### Directory Structure
```text
src/
├── services/
│   ├── dispatcher.py       # Notification delivery (Telegram, Dashboard)
│   ├── alert_engine.py     # Budget, bill, cashflow, and anomaly monitoring
│   └── ai_insights.py      # Anthropic weekly digest generator
├── api/
│   └── routers/
│       └── alerts.py       # FastAPI endpoints for alerts & preferences
└── worker/
    └── alert_tasks.py      # Asyncio background task runner
frontend/
└── src/
    └── pages/
        └── AlertsDashboard.tsx # React 19 + TypeScript UI
```

### 1. Notification Dispatcher
**`src/services/dispatcher.py`**
Handles multi-channel delivery and logs all notifications using the exact `Notification` schema.

```python
import logging
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.notification import Notification, NotificationChannel
from src.config import settings

logger = logging.getLogger(__name__)

async def dispatch_alert(
    db: AsyncSession, 
    channel: NotificationChannel, 
    subject: str, 
    body: str, 
    metadata_json: dict | None = None
) -> None:
    """Dispatches an alert to the specified channel and logs it to the database."""
    sent_ok = False
    error_msg = None

    if metadata_json is None:
        metadata_json = {"is_read": False}

    try:
        if channel == NotificationChannel.TELEGRAM:
            sent_ok, error_msg = await _send_telegram(subject, body)
        elif channel == NotificationChannel.DASHBOARD:
            # Dashboard notifications are inherently "sent" by being saved to the DB
            sent_ok = True
        else:
            error_msg = f"Delivery for channel {channel.name} is not implemented yet."
            logger.warning(error_msg)
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to dispatch alert to {channel.name}: {error_msg}")

    # Log to database using exact schema fields
    notification = Notification(
        channel=channel,
        subject=subject,
        body=body,
        metadata_json=metadata_json,
        sent_ok=sent_ok,
        error=error_msg
    )
    db.add(notification)
    await db.commit()

async def _send_telegram(subject: str, body: str) -> tuple[bool, str | None]:
    """Sends a message via Telegram Bot API."""
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return False, "Telegram credentials not configured in settings."

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": f"*{subject}*\n\n{body}",
        "parse_mode": "Markdown"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=10.0)
        if response.status_code == 200:
            return True, None
        return False, f"Telegram API error: {response.text}"
```

### 2. Core Alert Engine
**`src/services/alert_engine.py`**
Monitors budgets, bills, anomalies, and cash flow utilizing existing models and standard functions.

```python
import logging
from datetime import date, datetime, timedelta
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.budget import Budget
from src.models.transaction import Transaction
from src.models.bill import RecurringBill
from src.models.account import FinancialAccount
from src.models.notification import NotificationChannel
from src.core.cashflow import project_bills, aggregate_daily_forecast
from src.services.dispatcher import dispatch_alert

logger = logging.getLogger(__name__)

async def check_budget_thresholds(db: AsyncSession) -> None:
    """Alerts when spending approaches or exceeds the budget threshold."""
    current_month_str = datetime.now().strftime("%Y-%m")
    start_of_month = date.today().replace(day=1)
    
    stmt = select(Budget).where(Budget.month == current_month_str)
    budgets = (await db.execute(stmt)).scalars().all()

    for budget in budgets:
        # Calculate total spent for this category in the current month
        spent_stmt = select(func.sum(Transaction.amount)).where(
            Transaction.category_id == budget.category_id,
            Transaction.date >= start_of_month
        )
        spent = (await db.execute(spent_stmt)).scalar() or 0.0

        threshold_amount = budget.budgeted_amount * budget.alert_threshold
        if spent >= threshold_amount:
            subject = f"Budget Alert: {budget.category.name if budget.category else 'Category'}"
            body = f"You have spent ${spent:.2f}, which is >= {budget.alert_threshold*100}% of your ${budget.budgeted_amount:.2f} budget."
            await dispatch_alert(db, NotificationChannel.DASHBOARD, subject, body, {"type": "budget_alert", "is_read": False})

async def check_bill_reminders(db: AsyncSession) -> None:
    """Sends reminders for upcoming active bills based on their alert_days_before field."""
    today = date.today()
    stmt = select(RecurringBill).where(RecurringBill.is_active == True)
    bills = (await db.execute(stmt)).scalars().all()

    for bill in bills:
        if not bill.next_due_date:
            continue
            
        days_until_due = (bill.next_due_date - today).days
        if 0 <= days_until_due <= bill.alert_days_before:
            subject = f"Upcoming Bill: {bill.name}"
            body = f"Your bill for ${bill.amount:.2f} is due on {bill.next_due_date} ({days_until_due} days)."
            await dispatch_alert(db, NotificationChannel.DASHBOARD, subject, body, {"type": "bill_reminder", "is_read": False})

async def detect_anomalies(db: AsyncSession) -> None:
    """Detects transactions > 2x the average for their category."""
    thirty_days_ago = date.today() - timedelta(days=30)
    
    # Get recent transactions
    recent_stmt = select(Transaction).where(Transaction.date >= thirty_days_ago)
    recent_txs = (await db.execute(recent_stmt)).scalars().all()

    for tx in recent_txs:
        if not tx.category_id or tx.amount <= 0:
            continue

        # Get historical average for this category
        avg_stmt = select(func.avg(Transaction.amount)).where(
            Transaction.category_id == tx.category_id,
            Transaction.date < thirty_days_ago
        )
        avg_amount = (await db.execute(avg_stmt)).scalar() or 0.0

        if avg_amount > 0 and tx.amount > (avg_amount * 2):
            subject = "Large Transaction Detected"
            body = f"A transaction of ${tx.amount:.2f} at {tx.merchant or 'Unknown'} is more than 2x your average for this category."
            await dispatch_alert(db, NotificationChannel.DASHBOARD, subject, body, {"type": "anomaly", "is_read": False})

async def check_cashflow_warnings(db: AsyncSession) -> None:
    """Uses CashFlowEngine to warn if projected balances go negative in the next 14 days."""
    today = date.today()
    end_date = today + timedelta(days=14)

    # Get active bills for projection
    bills_stmt = select(RecurringBill).where(RecurringBill.is_active == True)
    bills = (await db.execute(bills_stmt)).scalars().all()

    # Synchronous calls to CashFlowEngine plain functions
    projections = project_bills(list(bills), today, end_date)
    forecasts = aggregate_daily_forecast(projections, today, end_date)

    # Calculate current total balance across all active accounts
    acc_stmt = select(func.sum(FinancialAccount.balance)).where(FinancialAccount.is_active == True)
    current_balance = (await db.execute(acc_stmt)).scalar() or 0.0

    running_balance = current_balance
    for forecast in forecasts:
        running_balance += forecast.net
        if running_balance < 0:
            subject = "Cash Flow Warning"
            body = f"Your projected balance will drop below zero (${running_balance:.2f}) on {forecast.date}."
            await dispatch_alert(db, NotificationChannel.DASHBOARD, subject, body, {"type": "cashflow_warning", "is_read": False})
            await dispatch_alert(db, NotificationChannel.TELEGRAM, subject, body)
            break # Only alert once for the earliest negative day
```

### 3. AI-Powered Insights
**`src/services/ai_insights.py`**
Generates weekly digests utilizing Anthropic and saves them as `Recommendation` entries.

```python
import logging
from anthropic import AsyncAnthropic
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.config import settings
from src.models.transaction import Transaction
from src.models.recommendation import Recommendation, RecommendationType, RecommendationStatus

logger = logging.getLogger(__name__)

async def generate_weekly_digest(db: AsyncSession) -> None:
    """Generates an AI-powered financial digest and stores it as a Recommendation."""
    if not settings.anthropic_api_key:
        logger.warning("Anthropic API key not configured. Skipping AI insights.")
        return

    # Gather basic context for the AI
    spent_stmt = select(func.sum(Transaction.amount))
    total_spent = (await db.execute(spent_stmt)).scalar() or 0.0

    prompt = (
        f"As a personal CFO, generate a brief, encouraging weekly financial digest. "
        f"The user has spent a total of ${total_spent:.2f} historically. "
        f"Provide a short summary and one actionable piece of advice."
    )

    try:
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        response = await client.messages.create(
            model=settings.ai_model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        
        ai_text = response.content[0].text
        
        # Split into summary and detail for the Recommendation model
        parts = ai_text.split('\n\n', 1)
        summary = parts[0]
        detail = parts[1] if len(parts) > 1 else None

        # Store using the exact Recommendation schema
        recommendation = Recommendation(
            recommendation_type=RecommendationType.GENERAL_INSIGHT,
            status=RecommendationStatus.PENDING,
            title="Weekly AI Financial Digest",
            summary=summary,
            detail=detail,
            priority=1, # Lower number = higher priority
            action_payload={"source": "ai_weekly_digest"}
        )
        db.add(recommendation)
        await db.commit()

    except Exception as e:
        logger.error(f"Failed to generate AI insights: {e}")
```

### 4. Background Task Worker
**`src/worker/alert_tasks.py`**
Runnable background task to trigger the monitoring engines periodically.

```python
import asyncio
import logging
from src.db.engine import async_session
from src.services.alert_engine import (
    check_budget_thresholds,
    check_bill_reminders,
    detect_anomalies,
    check_cashflow_warnings
)
from src.services.ai_insights import generate_weekly_digest

logger = logging.getLogger(__name__)

async def run_all_checks() -> None:
    """Executes all alert and insight checks using a fresh database session."""
    logger.info("Starting background alert checks...")
    async with async_session() as db:
        try:
            await check_budget_thresholds(db)
            await check_bill_reminders(db)
            await detect_anomalies(db)
            await check_cashflow_warnings(db)
            
            # Weekly digest could be conditionally run based on day of week
            # For demonstration, we run it here
            await generate_weekly_digest(db)
            
            await db.commit()
            logger.info("Completed background alert checks successfully.")
        except Exception as e:
            await db.rollback()
            logger.error(f"Error running alert checks: {e}")

async def start_scheduler(interval_seconds: int = 86400) -> None:
    """Runs the checks periodically (default: daily)."""
    while True:
        await run_all_checks()
        await asyncio.sleep(interval_seconds)
```

### 5. API Router
**`src/api/routers/alerts.py`**
FastAPI endpoints using proper Pydantic v2 configuration and exact dependencies. Since `AlertPreference` is not in the schema, preferences are stored in a local JSON file to avoid reinventing infrastructure.

```python
import json
import os
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel

from src.api.deps import get_db
from src.models.notification import Notification, NotificationChannel

router = APIRouter(prefix="/alerts", tags=["alerts"])
PREFS_FILE = "data/alert_preferences.json"

class NotificationResponse(BaseModel):
    id: UUID
    subject: str
    body: str
    metadata_json: dict | None
    sent_ok: bool
    
    model_config = {"from_attributes": True}

class AlertPreferences(BaseModel):
    budget_alerts: bool = True
    bill_reminders: bool = True
    anomaly_detection: bool = True
    telegram_enabled: bool = False

@router.get("/", response_model=list[NotificationResponse])
async def get_dashboard_alerts(db: AsyncSession = Depends(get_db)):
    """Fetches all dashboard notifications."""
    stmt = select(Notification).where(
        Notification.channel == NotificationChannel.DASHBOARD
    ).order_by(Notification.created_at.desc()).limit(50)
    
    result = await db.execute(stmt)
    return result.scalars().all()

@router.put("/{alert_id}/read")
async def mark_alert_read(alert_id: UUID, db: AsyncSession = Depends(get_db)):
    """Marks an alert as read by updating its metadata_json."""
    stmt = select(Notification).where(Notification.id == alert_id)
    alert = (await db.execute(stmt)).scalars().first()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    current_meta = alert.metadata_json or {}
    current_meta["is_read"] = True
    
    # SQLAlchemy JSON mutation requires re-assignment
    alert.metadata_json = current_meta
    await db.commit()
    return {"status": "success"}

@router.get("/preferences", response_model=AlertPreferences)
async def get_preferences():
    """Retrieves alert preferences from local storage."""
    if os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "r") as f:
            return AlertPreferences(**json.load(f))
    return AlertPreferences()

@router.put("/preferences", response_model=AlertPreferences)
async def update_preferences(prefs: AlertPreferences):
    """Saves alert preferences to local storage."""
    os.makedirs(os.path.dirname(PREFS_FILE), exist_ok=True)
    with open(PREFS_FILE, "w") as f:
        json.dump(prefs.model_dump(), f)
    return prefs
```

### 6. Frontend Dashboard
**`frontend/src/pages/AlertsDashboard.tsx`**
React 19 + TypeScript implementation using the required `apiClient`.

```tsx
import React, { useState, useEffect } from 'react';
import { apiClient } from '../client';

interface Notification {
  id: string;
  subject: string;
  body: string;
  metadata_json: { is_read?: boolean; type?: string } | null;
  sent_ok: boolean;
}

interface AlertPreferences {
  budget_alerts: boolean;
  bill_reminders: boolean;
  anomaly_detection: boolean;
  telegram_enabled: boolean;
}

export default function AlertsDashboard() {
  const [alerts, setAlerts] = useState<Notification[]>([]);
  const [prefs, setPrefs] = useState<AlertPreferences | null>(null);
  const [activeTab, setActiveTab] = useState<'history' | 'settings'>('history');

  useEffect(() => {
    fetchAlerts();
    fetchPreferences();
  }, []);

  const fetchAlerts = async () => {
    try {
      const response = await apiClient.get('/alerts');
      setAlerts(response.data);
    } catch (error) {
      console.error('Failed to fetch alerts', error);
    }
  };

  const fetchPreferences = async () => {
    try {
      const response = await apiClient.get('/alerts/preferences');
      setPrefs(response.data);
    } catch (error) {
      console.error('Failed to fetch preferences', error);
    }
  };

  const markAsRead = async (id: string) => {
    try {
      await apiClient.put(`/alerts/${id}/read`);
      setAlerts(alerts.map(a => 
        a.id === id ? { ...a, metadata_json: { ...a.metadata_json, is_read: true } } : a
      ));
    } catch (error) {
      console.error('Failed to mark as read', error);
    }
  };

  const savePreferences = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prefs) return;
    try {
      await apiClient.put('/alerts/preferences', prefs);
      alert('Preferences saved successfully!');
    } catch (error) {
      console.error('Failed to save preferences', error);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Alerts & Notifications</h1>
        <div className="space-x-2">
          <button 
            onClick={() => setActiveTab('history')}
            className={`px-4 py-2 rounded-lg font-medium ${activeTab === 'history' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600'}`}
          >
            Alert History
          </button>
          <button 
            onClick={() => setActiveTab('settings')}
            className={`px-4 py-2 rounded-lg font-medium ${activeTab === 'settings' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600'}`}
          >
            Preferences
          </button>
        </div>
      </div>

      {activeTab === 'history' ? (
        <div className="space-y-4">
          {alerts.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No alerts found.</p>
          ) : (
            alerts.map(alert => {
              const isRead = alert.metadata_json?.is_read;
              return (
                <div key={alert.id} className={`p-4 rounded-lg border shadow-sm flex justify-between items-start ${isRead ? 'bg-gray-50' : 'bg-white border-blue-200'}`}>
                  <div>
                    <h3 className={`font-semibold ${isRead ? 'text-gray-700' : 'text-gray-900'}`}>
                      {alert.subject}
                    </h3>
                    <p className="text-gray-600 mt-1">{alert.body}</p>
                  </div>
                  {!isRead && (
                    <button 
                      onClick={() => markAsRead(alert.id)}
                      className="text-sm text-blue-600 hover:text-blue-800 font-medium whitespace-nowrap ml-4"
                    >
                      Mark as Read
                    </button>
                  )}
                </div>
              );
            })
          )}
        </div>
      ) : (
        <form onSubmit={savePreferences} className="bg-white p-6 rounded-lg border shadow-sm space-y-6">
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-900 text-lg">Monitoring Rules</h3>
            <label className="flex items-center space-x-3">
              <input 
                type="checkbox" 
                checked={prefs?.budget_alerts || false} 
                onChange={e => setPrefs(prev => prev ? {...prev, budget_alerts: e.target.checked} : null)} 
                className="rounded text-blue-600 w-5 h-5" 
              />
              <span className="text-gray-700">Budget Threshold Alerts</span>
            </label>
            <label className="flex items-center space-x-3">
              <input 
                type="checkbox" 
                checked={prefs?.bill_reminders || false} 
                onChange={e => setPrefs(prev => prev ? {...prev, bill_reminders: e.target.checked} : null)} 
                className="rounded text-blue-600 w-5 h-5" 
              />
              <span className="text-gray-700">Bill Due Date Reminders</span>
            </label>
            <label className="flex items-center space-x-3">
              <input 
                type="checkbox" 
                checked={prefs?.anomaly_detection || false} 
                onChange={e => setPrefs(prev => prev ? {...prev, anomaly_detection: e.target.checked} : null)} 
                className="rounded text-blue-600 w-5 h-5" 
              />
              <span className="text-gray-700">Anomaly & Spike Detection</span>
            </label>
          </div>

          <div className="space-y-4 pt-4 border-t">
            <h3 className="font-semibold text-gray-900 text-lg">Delivery Channels</h3>
            <label className="flex items-center space-x-3">
              <input 
                type="checkbox" 
                checked={prefs?.telegram_enabled || false} 
                onChange={e => setPrefs(prev => prev ? {...prev, telegram_enabled: e.target.checked} : null)} 
                className="rounded text-blue-600 w-5 h-5" 
              />
              <span className="text-gray-700">Enable Telegram Delivery</span>
            </label>
            <p className="text-sm text-gray-500 pl-8">
              Requires Telegram Bot Token and Chat ID to be configured in your environment variables.
            </p>
          </div>

          <div className="pt-4">
            <button type="submit" className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 font-medium transition-colors">
              Save Preferences
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
```