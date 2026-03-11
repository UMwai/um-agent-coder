Here is the complete, production-ready implementation of the smart budget alert and notification system. 

It strictly adheres to your SQLAlchemy 2.0 async patterns, uses the existing models, avoids N+1 queries by aggregating data in memory or via grouped SQL statements, and fully implements all requested features including the previously missing new recurring charge detection.

### 1. Dependencies Setup

**`requirements.txt`** (Append these to your existing file)
```text
fastapi>=0.100.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
httpx>=0.24.1
anthropic>=0.3.0
```

**`frontend/package.json`** (Dependencies snippet)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-router-dom": "^6.14.0",
    "recharts": "^2.7.2",
    "@tanstack/react-query": "^4.29.19",
    "tailwindcss": "^3.3.2",
    "axios": "^1.4.0",
    "lucide-react": "^0.263.1"
  }
}
```

### 2. Database Models & Schemas

**`src/models/preferences.py`**
```python
from sqlalchemy import Column, Boolean, String
from src.models.base import BaseModel

class AlertPreference(BaseModel):
    """
    Stores user preferences for which alerts to receive and through which channels.
    Assumes a single-tenant environment as per existing models (no user_id on base models).
    """
    __tablename__ = "alert_preferences"
    
    profile_id = Column(String(50), unique=True, default="default", index=True)
    
    telegram_enabled = Column(Boolean, default=False)
    dashboard_enabled = Column(Boolean, default=True)
    
    budget_alerts = Column(Boolean, default=True)
    bill_reminders = Column(Boolean, default=True)
    anomaly_detection = Column(Boolean, default=True)
    cash_flow_warnings = Column(Boolean, default=True)
```

**`src/schemas/alerts.py`**
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Any
from src.models.notification import NotificationChannel

class AlertPreferenceUpdate(BaseModel):
    telegram_enabled: Optional[bool] = None
    dashboard_enabled: Optional[bool] = None
    budget_alerts: Optional[bool] = None
    bill_reminders: Optional[bool] = None
    anomaly_detection: Optional[bool] = None
    cash_flow_warnings: Optional[bool] = None

class AlertPreferenceResponse(BaseModel):
    id: str
    telegram_enabled: bool
    dashboard_enabled: bool
    budget_alerts: bool
    bill_reminders: bool
    anomaly_detection: bool
    cash_flow_warnings: bool

    model_config = {"from_attributes": True}

class NotificationResponse(BaseModel):
    id: str
    channel: NotificationChannel
    subject: str
    body: str
    metadata_json: dict[str, Any]
    sent_ok: bool
    error: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}
```

### 3. Core Services

**`src/services/notifications.py`**
```python
import logging
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.notification import Notification, NotificationChannel
from src.config import settings

logger = logging.getLogger(__name__)

class NotificationService:
    """Handles the dispatching and logging of notifications across multiple channels."""
    
    @staticmethod
    async def dispatch(
        db: AsyncSession, 
        subject: str, 
        body: str, 
        channel: NotificationChannel, 
        metadata: dict = None
    ) -> Notification:
        """
        Dispatches an alert to the requested channel and logs it in the database.
        """
        metadata = metadata or {}
        sent_ok = False
        error_msg = None

        try:
            if channel == NotificationChannel.TELEGRAM:
                sent_ok, error_msg = await NotificationService._send_telegram(subject, body)
            elif channel == NotificationChannel.DASHBOARD:
                # Dashboard alerts are purely stored in DB for frontend retrieval
                sent_ok = True
            else:
                error_msg = f"Delivery for channel {channel.name} not implemented."
                logger.warning(error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to send notification via {channel.name}: {e}")

        # Log notification to database
        notification = Notification(
            channel=channel,
            subject=subject,
            body=body,
            metadata_json=metadata,
            sent_ok=sent_ok,
            error=error_msg
        )
        db.add(notification)
        await db.commit()
        await db.refresh(notification)
        
        return notification

    @staticmethod
    async def _send_telegram(subject: str, body: str) -> tuple[bool, str | None]:
        """Sends a message via the Telegram Bot API."""
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            return False, "Telegram credentials not configured in settings."

        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": settings.telegram_chat_id,
            "text": f"🚨 *{subject}*\n\n{body}",
            "parse_mode": "Markdown"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10.0)
            if response.status_code == 200:
                return True, None
            return False, f"Telegram API Error: {response.text}"
```

**`src/services/alert_engine.py`**
```python
import logging
from datetime import date, datetime, timedelta
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from src.models.budget import Budget
from src.models.bill import RecurringBill
from src.models.transaction import Transaction, TransactionType
from src.models.account import FinancialAccount
from src.models.notification import Notification, NotificationChannel
from src.models.preferences import AlertPreference
from src.services.notifications import NotificationService
from src.core.cashflow import project_bills, aggregate_daily_forecast

logger = logging.getLogger(__name__)

# Constants for anomaly detection and alert throttling
LARGE_TX_MULTIPLIER = 2.0
WOW_SPIKE_THRESHOLD_RATIO = 1.5
WOW_MINIMUM_SPEND = 100.0
ALERT_COOLDOWN_DAYS = 7

class AlertEngine:
    """Proactively monitors financial state and triggers alerts based on user preferences."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.recent_notifications: List[Notification] = []

    async def _load_recent_notifications(self):
        """Pre-loads recent notifications to avoid N+1 queries when checking for duplicates."""
        cutoff = datetime.utcnow() - timedelta(days=ALERT_COOLDOWN_DAYS)
        stmt = select(Notification).where(
            Notification.created_at >= cutoff,
            Notification.sent_ok == True
        )
        result = await self.db.execute(stmt)
        self.recent_notifications = list(result.scalars().all())

    def _has_recent_alert(self, key: str, value: str) -> bool:
        """Checks if a specific alert was already sent recently (idempotency check)."""
        for n in self.recent_notifications:
            if n.metadata_json and n.metadata_json.get(key) == value:
                return True
        return False

    async def get_preferences(self) -> AlertPreference:
        """Fetches or creates the default alert preferences."""
        stmt = select(AlertPreference).where(AlertPreference.profile_id == "default")
        result = await self.db.execute(stmt)
        prefs = result.scalars().first()
        if not prefs:
            prefs = AlertPreference(profile_id="default")
            self.db.add(prefs)
            await self.db.commit()
        return prefs

    async def run_all_checks(self):
        """Executes all enabled monitoring checks."""
        prefs = await self.get_preferences()
        await self._load_recent_notifications()

        channels = []
        if prefs.dashboard_enabled:
            channels.append(NotificationChannel.DASHBOARD)
        if prefs.telegram_enabled:
            channels.append(NotificationChannel.TELEGRAM)

        if not channels:
            logger.info("No notification channels enabled. Skipping checks.")
            return

        if prefs.budget_alerts:
            await self.check_budgets(channels)
        if prefs.bill_reminders:
            await self.check_bills(channels)
        if prefs.anomaly_detection:
            await self.check_anomalies(channels)
        if prefs.cash_flow_warnings:
            await self.check_cashflow(channels)

    async def check_budgets(self, channels: list[NotificationChannel]):
        """Monitors spending against budgets and alerts if exceeding thresholds."""
        current_month = date.today().strftime("%Y-%m")
        month_start = date.today().replace(day=1)
        
        # Get all budgets for the month
        budgets = (await self.db.execute(select(Budget).where(Budget.month == current_month))).scalars().all()
        if not budgets:
            return

        # Avoid N+1: Get all category spending for the current month in one grouped query
        stmt = select(
            Transaction.category_id, 
            func.sum(Transaction.amount).label("spent")
        ).where(
            Transaction.date >= month_start,
            Transaction.transaction_type == TransactionType.EXPENSE
        ).group_by(Transaction.category_id)
        
        category_spending = dict((await self.db.execute(stmt)).all())

        for budget in budgets:
            spent = category_spending.get(budget.category_id, 0.0)
            threshold_amount = budget.budgeted_amount * budget.alert_threshold

            if spent >= threshold_amount:
                alert_key = f"budget_{budget.id}_{current_month}"
                if self._has_recent_alert("alert_key", alert_key):
                    continue

                subject = "Budget Threshold Reached"
                body = f"You have spent ${spent:.2f} of your ${budget.budgeted_amount:.2f} budget for this category."
                
                for channel in channels:
                    await NotificationService.dispatch(
                        self.db, subject, body, channel, {"alert_key": alert_key, "budget_id": str(budget.id)}
                    )

    async def check_bills(self, channels: list[NotificationChannel]):
        """Sends reminders for bills due soon, respecting the alert_days_before setting."""
        bills = (await self.db.execute(select(RecurringBill).where(RecurringBill.is_active == True))).scalars().all()
        today = date.today()

        for bill in bills:
            if not bill.next_due_date:
                continue
                
            days_until = (bill.next_due_date - today).days
            
            # Using <= handles missed background jobs, but idempotency check prevents spam
            if 0 <= days_until <= bill.alert_days_before:
                alert_key = f"bill_{bill.id}_{bill.next_due_date.isoformat()}"
                if self._has_recent_alert("alert_key", alert_key):
                    continue

                subject = f"Upcoming Bill: {bill.name}"
                body = f"Your bill for {bill.name} (${bill.amount:.2f}) is due on {bill.next_due_date.strftime('%b %d')} ({days_until} days)."
                
                for channel in channels:
                    await NotificationService.dispatch(
                        self.db, subject, body, channel, {"alert_key": alert_key, "bill_id": str(bill.id)}
                    )

    async def check_anomalies(self, channels: list[NotificationChannel]):
        """Detects unusual spending patterns: large transactions, spikes, and new recurring charges."""
        today = date.today()
        seven_days_ago = today - timedelta(days=7)
        fourteen_days_ago = today - timedelta(days=14)

        # 1. Large Transactions (>2x category average)
        recent_txs = (await self.db.execute(
            select(Transaction).where(
                Transaction.date >= seven_days_ago,
                Transaction.transaction_type == TransactionType.EXPENSE
            )
        )).scalars().all()

        if recent_txs:
            # Avoid N+1: Get historical averages for all categories in one query
            cat_ids = list({t.category_id for t in recent_txs if t.category_id})
            avg_stmt = select(
                Transaction.category_id, 
                func.avg(Transaction.amount)
            ).where(
                Transaction.category_id.in_(cat_ids),
                Transaction.transaction_type == TransactionType.EXPENSE,
                Transaction.date < seven_days_ago
            ).group_by(Transaction.category_id)
            
            averages = dict((await self.db.execute(avg_stmt)).all())

            for tx in recent_txs:
                avg = averages.get(tx.category_id, 0)
                if avg > 0 and tx.amount > (avg * LARGE_TX_MULTIPLIER):
                    alert_key = f"large_tx_{tx.id}"
                    if not self._has_recent_alert("alert_key", alert_key):
                        for c in channels:
                            await NotificationService.dispatch(
                                self.db, "Large Transaction Detected",
                                f"Unusually large transaction: ${tx.amount:.2f} at {tx.merchant or 'Unknown'}.",
                                c, {"alert_key": alert_key}
                            )

        # 2. Sudden Spending Spikes (WoW)
        current_week_stmt = select(func.sum(Transaction.amount)).where(
            Transaction.date >= seven_days_ago,
            Transaction.transaction_type == TransactionType.EXPENSE
        )
        prev_week_stmt = select(func.sum(Transaction.amount)).where(
            Transaction.date >= fourteen_days_ago,
            Transaction.date < seven_days_ago,
            Transaction.transaction_type == TransactionType.EXPENSE
        )
        
        current_spend = (await self.db.execute(current_week_stmt)).scalar() or 0.0
        prev_spend = (await self.db.execute(prev_week_stmt)).scalar() or 0.0

        if prev_spend > 0 and current_spend > WOW_MINIMUM_SPEND and current_spend > (prev_spend * WOW_SPIKE_THRESHOLD_RATIO):
            alert_key = f"spike_wow_{today.isocalendar()[1]}"
            if not self._has_recent_alert("alert_key", alert_key):
                for c in channels:
                    await NotificationService.dispatch(
                        self.db, "Spending Spike Alert",
                        f"You've spent ${current_spend:.2f} in the last 7 days, significantly higher than last week (${prev_spend:.2f}).",
                        c, {"alert_key": alert_key}
                    )

        # 3. New Recurring Charges
        recent_recurring = (await self.db.execute(
            select(Transaction).where(
                Transaction.is_recurring == True,
                Transaction.date >= fourteen_days_ago
            )
        )).scalars().all()

        if recent_recurring:
            # Fetch known active bills to compare
            bills = (await self.db.execute(select(RecurringBill).where(RecurringBill.is_active == True))).scalars().all()
            known_merchants = {b.merchant.lower() for b in bills if b.merchant}
            known_names = {b.name.lower() for b in bills if b.name}

            for tx in recent_recurring:
                tx_desc = (tx.merchant or tx.description or "").lower()
                
                # Check if this transaction matches any known bill
                is_known = any(m in tx_desc for m in known_merchants) or any(n in tx_desc for n in known_names)
                
                if not is_known:
                    alert_key = f"new_recurring_{tx.id}"
                    if not self._has_recent_alert("alert_key", alert_key):
                        for c in channels:
                            await NotificationService.dispatch(
                                self.db, "New Recurring Charge Detected",
                                f"We noticed a new recurring charge: ${tx.amount:.2f} at {tx.merchant or tx.description}. Consider adding it to your Bills.",
                                c, {"alert_key": alert_key}
                            )

    async def check_cashflow(self, channels: list[NotificationChannel]):
        """Integrates with CashFlowEngine to warn if projected balances go negative in 14 days."""
        accounts = (await self.db.execute(select(FinancialAccount).where(FinancialAccount.is_active == True))).scalars().all()
        total_balance = sum(acc.balance for acc in accounts)

        bills = (await self.db.execute(select(RecurringBill).where(RecurringBill.is_active == True))).scalars().all()
        
        start_date = date.today()
        end_date = start_date + timedelta(days=14)
        
        # CashFlowEngine functions are synchronous per requirements
        projections = project_bills(bills, start_date, end_date)
        forecast = aggregate_daily_forecast(projections, start_date, end_date, {}, {})

        running_balance = total_balance
        negative_date = None

        for day in forecast:
            running_balance += day.net
            if running_balance < 0:
                negative_date = day.date
                break

        if negative_date:
            alert_key = f"cashflow_warn_{negative_date.isoformat()}"
            if not self._has_recent_alert("alert_key", alert_key):
                subject = "⚠️ Cash Flow Warning"
                body = f"Based on upcoming bills, your total balance is projected to drop below $0.00 on {negative_date.strftime('%b %d')}."
                for c in channels:
                    await NotificationService.dispatch(self.db, subject, body, c, {"alert_key": alert_key})
```

**`src/services/ai_insights.py`**
```python
import logging
from datetime import date, timedelta
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import anthropic

from src.config import settings
from src.models.recommendation import Recommendation, RecommendationType, RecommendationStatus
from src.models.transaction import Transaction, TransactionType

logger = logging.getLogger(__name__)

async def generate_weekly_digest(db: AsyncSession):
    """Generates an AI-powered weekly digest and stores it as a Recommendation."""
    if not settings.anthropic_api_key:
        logger.warning("Anthropic API key not configured. Skipping AI insights.")
        return

    # Check if we already generated one this week
    week_start = date.today() - timedelta(days=date.today().weekday())
    stmt = select(Recommendation).where(
        Recommendation.recommendation_type == RecommendationType.GENERAL_INSIGHT,
        func.date(Recommendation.created_at) >= week_start
    )
    existing = (await db.execute(stmt)).scalars().first()
    if existing:
        return

    # Gather data for the prompt
    month_start = date.today().replace(day=1)
    tx_stmt = select(Transaction).where(
        Transaction.date >= month_start,
        Transaction.transaction_type == TransactionType.EXPENSE
    ).order_by(Transaction.date.desc()).limit(50)
    
    recent_expenses = (await db.execute(tx_stmt)).scalars().all()
    total_spent = sum(t.amount for t in recent_expenses)

    prompt = (
        f"As an expert financial advisor, analyze this user's recent spending.\n"
        f"Total spent this month: ${total_spent:.2f}\n"
        f"Recent transactions: {[f'{t.date}: ${t.amount} at {t.merchant or t.description}' for t in recent_expenses[:10]]}\n"
        "Provide a short, encouraging 2-paragraph weekly financial digest with 1 concrete recommendation. "
        "Do not use markdown formatting, just plain text."
    )

    try:
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        response = await client.messages.create(
            model=settings.ai_model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        insight_text = response.content[0].text

        rec = Recommendation(
            recommendation_type=RecommendationType.GENERAL_INSIGHT,
            status=RecommendationStatus.PENDING,
            title=f"Weekly Financial Digest - {date.today().strftime('%b %d')}",
            summary="Your personalized AI financial review for the week.",
            detail=insight_text,
            priority=5, # Lower number = higher priority, 5 is standard for weekly insights
            confidence=0.9
        )
        db.add(rec)
        await db.commit()
        
    except Exception as e:
        logger.error(f"Failed to generate AI insight: {e}")
```

### 4. API & Background Tasks

**`src/api/alerts.py`**
```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from src.api.deps import get_db
from src.models.preferences import AlertPreference
from src.models.notification import Notification
from src.schemas.alerts import AlertPreferenceUpdate, AlertPreferenceResponse, NotificationResponse
from src.services.alert_engine import AlertEngine

router = APIRouter(prefix="/alerts", tags=["alerts"])

@router.get("/preferences", response_model=AlertPreferenceResponse)
async def get_preferences(db: AsyncSession = Depends(get_db)):
    """Retrieve the user's current alert preferences."""
    engine = AlertEngine(db)
    return await engine.get_preferences()

@router.patch("/preferences", response_model=AlertPreferenceResponse)
async def update_preferences(prefs_in: AlertPreferenceUpdate, db: AsyncSession = Depends(get_db)):
    """Update alert and notification preferences."""
    engine = AlertEngine(db)
    prefs = await engine.get_preferences()
    
    update_data = prefs_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(prefs, field, value)
        
    await db.commit()
    await db.refresh(prefs)
    return prefs

@router.get("/history", response_model=List[NotificationResponse])
async def get_alert_history(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """View the history of sent notifications."""
    stmt = select(Notification).order_by(Notification.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

@router.post("/trigger-run", status_code=202)
async def trigger_manual_run(db: AsyncSession = Depends(get_db)):
    """Manually triggers the background alert engine (useful for testing)."""
    engine = AlertEngine(db)
    await engine.run_all_checks()
    return {"status": "Alert engine run triggered successfully"}
```

**`src/tasks/scheduler.py`**
```python
import asyncio
import logging
from src.db.engine import async_session
from src.services.alert_engine import AlertEngine
from src.services.ai_insights import generate_weekly_digest

logger = logging.getLogger(__name__)

async def run_alert_engine_task():
    """
    Background task meant to be run periodically (e.g., via APScheduler or a simple asyncio loop).
    Executes core monitoring and AI insight generation.
    """
    logger.info("Starting background alert engine task...")
    try:
        async with async_session() as db:
            # 1. Run Core Alert Checks
            engine = AlertEngine(db)
            await engine.run_all_checks()
            
            # 2. Run Weekly AI Insights generator
            await generate_weekly_digest(db)
            
        logger.info("Alert engine task completed successfully.")
    except Exception as e:
        logger.error(f"Error in background alert engine task: {e}")

# Example of a simple periodic runner if an external scheduler isn't used
async def start_periodic_scheduler(interval_seconds: int = 3600):
    while True:
        await run_alert_engine_task()
        await asyncio.sleep(interval_seconds)
```

### 5. Frontend Implementation

**`frontend/src/client.ts`**
```typescript
import axios from 'axios';

// Assuming standard setup matching the project's React pattern
export const apiClient = axios.create({
  baseURL: '/api', // Adjust depending on proxy setup
  headers: {
    'Content-Type': 'application/json',
  },
});
```

**`frontend/src/pages/AlertsPage.tsx`**
```tsx
import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Bell, Settings, AlertTriangle, CheckCircle, Activity, Smartphone } from 'lucide-react';
import { apiClient } from '../client';

// Types matching the Pydantic schemas
interface AlertPreferences {
  id: string;
  telegram_enabled: boolean;
  dashboard_enabled: boolean;
  budget_alerts: boolean;
  bill_reminders: boolean;
  anomaly_detection: boolean;
  cash_flow_warnings: boolean;
}

interface NotificationLog {
  id: string;
  channel: string;
  subject: string;
  body: string;
  created_at: string;
  sent_ok: boolean;
}

export default function AlertsPage() {
  const queryClient = useQueryClient();

  // Fetch Preferences
  const { data: prefs, isLoading: prefsLoading } = useQuery<AlertPreferences>({
    queryKey: ['alert-preferences'],
    queryFn: async () => {
      const { data } = await apiClient.get('/alerts/preferences');
      return data;
    },
  });

  // Fetch History
  const { data: history, isLoading: historyLoading } = useQuery<NotificationLog[]>({
    queryKey: ['alert-history'],
    queryFn: async () => {
      const { data } = await apiClient.get('/alerts/history');
      return data;
    },
  });

  // Update Preferences Mutation
  const updatePrefs = useMutation({
    mutationFn: async (updates: Partial<AlertPreferences>) => {
      await apiClient.patch('/alerts/preferences', updates);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alert-preferences'] });
    },
  });

  const handleToggle = (key: keyof AlertPreferences) => {
    if (prefs) {
      updatePrefs.mutate({ [key]: !prefs[key] });
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      <header className="flex items-center justify-between border-b pb-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-2">
            <Bell className="w-8 h-8 text-blue-600" />
            Alerts & Notifications
          </h1>
          <p className="text-gray-500 mt-1">Manage your proactive financial monitoring.</p>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Preferences */}
        <div className="lg:col-span-1 bg-white border rounded-xl shadow-sm p-6 space-y-6">
          <div className="flex items-center gap-2 text-lg font-semibold text-gray-800 border-b pb-2">
            <Settings className="w-5 h-5 text-gray-500" />
            Configuration
          </div>

          {prefsLoading ? (
            <div className="animate-pulse space-y-4">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="h-6 bg-gray-200 rounded w-full"></div>
              ))}
            </div>
          ) : (
            prefs && (
              <div className="space-y-6">
                <div className="space-y-4">
                  <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider">Delivery Channels</h3>
                  <ToggleRow label="In-App Dashboard" checked={prefs.dashboard_enabled} onChange={() => handleToggle('dashboard_enabled')} icon={<Activity className="w-4 h-4 text-blue-500" />} />
                  <ToggleRow label="Telegram Bot" checked={prefs.telegram_enabled} onChange={() => handleToggle('telegram_enabled')} icon={<Smartphone className="w-4 h-4 text-blue-500" />} />
                </div>

                <div className="space-y-4">
                  <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider">Monitoring Rules</h3>
                  <ToggleRow label="Budget Thresholds" checked={prefs.budget_alerts} onChange={() => handleToggle('budget_alerts')} />
                  <ToggleRow label="Bill Reminders" checked={prefs.bill_reminders} onChange={() => handleToggle('bill_reminders')} />
                  <ToggleRow label="Anomaly Detection" checked={prefs.anomaly_detection} onChange={() => handleToggle('anomaly_detection')} />
                  <ToggleRow label="Cash Flow Warnings" checked={prefs.cash_flow_warnings} onChange={() => handleToggle('cash_flow_warnings')} />
                </div>
              </div>
            )
          )}
        </div>

        {/* Right Column: Alert History */}
        <div className="lg:col-span-2 bg-white border rounded-xl shadow-sm p-6">
          <div className="flex items-center gap-2 text-lg font-semibold text-gray-800 border-b pb-4 mb-4">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            Recent Activity
          </div>

          {historyLoading ? (
            <div className="flex justify-center p-8 text-gray-400">Loading history...</div>
          ) : history && history.length > 0 ? (
            <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
              {history.map((log) => (
                <div key={log.id} className="p-4 border rounded-lg hover:bg-gray-50 transition-colors flex items-start gap-4">
                  <div className="mt-1">
                    {log.sent_ok ? (
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    ) : (
                      <AlertTriangle className="w-5 h-5 text-red-500" />
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-start">
                      <h4 className="font-semibold text-gray-900">{log.subject}</h4>
                      <span className="text-xs text-gray-400 font-medium bg-gray-100 px-2 py-1 rounded">
                        {log.channel}
                      </span>
                    </div>
                    <p className="text-gray-600 text-sm mt-1">{log.body}</p>
                    <p className="text-xs text-gray-400 mt-2">
                      {new Date(log.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              No recent alerts found. You're all caught up!
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Reusable Toggle Component
function ToggleRow({ label, checked, onChange, icon }: { label: string, checked: boolean, onChange: () => void, icon?: React.ReactNode }) {
  return (
    <label className="flex items-center justify-between cursor-pointer group">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-gray-700 font-medium group-hover:text-gray-900 transition-colors text-sm">{label}</span>
      </div>
      <div className="relative inline-flex items-center">
        <input type="checkbox" className="sr-only peer" checked={checked} onChange={onChange} />
        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
      </div>
    </label>
  );
}
```