Here is the complete, production-ready implementation of the Smart Budget Alert and Notification System for the um-cfo platform. 

This implementation addresses all previous feedback: schemas use the correct `Optional` update pattern, anomaly detection strictly filters by `EXPENSE` and includes the missing >2x category average check, error handling/logging is robust across all layers, and the frontend properly handles React Query error states.

### 1. Database Models (`src/models/alerts.py`)
Stores user preferences and deduplication states to prevent alert spam.

```python
import uuid
from sqlalchemy import Column, Boolean, String, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from src.models.base import BaseModel

class AlertPreference(BaseModel):
    __tablename__ = "alert_preferences"

    # In a multi-user system, this links to the User. For um-cfo, 
    # we use a UUID to represent the primary user profile.
    user_id = Column(UUID(as_uuid=True), nullable=False, unique=True, index=True)
    
    # Alert rules toggles
    enable_budget_alerts = Column(Boolean, default=True, nullable=False)
    enable_bill_reminders = Column(Boolean, default=True, nullable=False)
    enable_anomaly_detection = Column(Boolean, default=True, nullable=False)
    enable_cashflow_warnings = Column(Boolean, default=True, nullable=False)
    enable_ai_insights = Column(Boolean, default=True, nullable=False)
    
    # Delivery channels
    notify_via_dashboard = Column(Boolean, default=True, nullable=False)
    notify_via_telegram = Column(Boolean, default=False, nullable=False)

class AlertState(BaseModel):
    """Tracks triggered alerts to prevent duplicate notifications."""
    __tablename__ = "alert_states"

    alert_type = Column(String(50), nullable=False, index=True)
    reference_id = Column(String(100), nullable=False, index=True)
    last_triggered_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

```

### 2. Pydantic Schemas (`src/schemas/alerts.py`)
Follows the standard CRUD pattern, explicitly making update fields optional.

```python
from pydantic import BaseModel
from typing import Optional
from uuid import UUID

class AlertPreferenceCreate(BaseModel):
    enable_budget_alerts: bool = True
    enable_bill_reminders: bool = True
    enable_anomaly_detection: bool = True
    enable_cashflow_warnings: bool = True
    enable_ai_insights: bool = True
    notify_via_dashboard: bool = True
    notify_via_telegram: bool = False

class AlertPreferenceUpdate(BaseModel):
    enable_budget_alerts: Optional[bool] = None
    enable_bill_reminders: Optional[bool] = None
    enable_anomaly_detection: Optional[bool] = None
    enable_cashflow_warnings: Optional[bool] = None
    enable_ai_insights: Optional[bool] = None
    notify_via_dashboard: Optional[bool] = None
    notify_via_telegram: Optional[bool] = None

class AlertPreferenceResponse(BaseModel):
    id: UUID
    user_id: UUID
    enable_budget_alerts: bool
    enable_bill_reminders: bool
    enable_anomaly_detection: bool
    enable_cashflow_warnings: bool
    enable_ai_insights: bool
    notify_via_dashboard: bool
    notify_via_telegram: bool

    model_config = {"from_attributes": True}
```

### 3. Notification Dispatcher (`src/core/alerts/dispatcher.py`)
Handles multi-channel delivery and logging to the existing `Notification` model.

```python
import logging
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.notification import Notification, NotificationChannel
from src.config import settings

logger = logging.getLogger(__name__)

async def dispatch_notification(
    db: AsyncSession,
    subject: str,
    body: str,
    channels: list[NotificationChannel],
    metadata: dict | None = None
) -> None:
    """Dispatches alerts to configured channels and logs them."""
    for channel in channels:
        sent_ok = False
        error_msg = None

        try:
            if channel == NotificationChannel.TELEGRAM:
                sent_ok, error_msg = await _send_telegram(body)
            elif channel == NotificationChannel.DASHBOARD:
                sent_ok = True  # Dashboard is just stored in DB
            else:
                error_msg = f"Delivery not implemented for channel: {channel}"
                logger.warning(error_msg)
                
        except Exception as e:
            logger.exception(f"Failed to send notification via {channel}")
            error_msg = str(e)

        # Log to existing Notification model
        notification_log = Notification(
            channel=channel,
            subject=subject,
            body=body,
            metadata_json=metadata or {},
            sent_ok=sent_ok,
            error=error_msg
        )
        db.add(notification_log)
    
    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to commit notification logs: {e}")

async def _send_telegram(message: str) -> tuple[bool, str | None]:
    token = getattr(settings, "telegram_bot_token", None)
    chat_id = getattr(settings, "telegram_chat_id", None)
    
    if not token or not chat_id:
        return False, "Telegram credentials not configured in settings."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=10.0)
        if response.status_code == 200:
            return True, None
        return False, f"Telegram API error: {response.text}"
```

### 4. Alert Rules Engine (`src/core/alerts/rules.py`)
Contains the heavy lifting for all 5 core capabilities.

```python
import logging
import datetime
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from anthropic import AsyncAnthropic

from src.models.budget import Budget
from src.models.transaction import Transaction, TransactionType
from src.models.category import Category
from src.models.bill import RecurringBill
from src.models.recommendation import Recommendation, RecommendationType, RecommendationStatus
from src.models.notification import NotificationChannel
from src.models.alerts import AlertPreference, AlertState
from src.core.alerts.dispatcher import dispatch_notification
from src.core.cashflow import project_bills, aggregate_daily_forecast
from src.config import settings

logger = logging.getLogger(__name__)

async def get_active_channels(db: AsyncSession) -> list[NotificationChannel]:
    """Helper to get user's configured notification channels."""
    try:
        result = await db.execute(select(AlertPreference).limit(1))
        pref = result.scalars().first()
        channels = []
        if pref:
            if pref.notify_via_dashboard:
                channels.append(NotificationChannel.DASHBOARD)
            if pref.notify_via_telegram:
                channels.append(NotificationChannel.TELEGRAM)
        else:
            channels.append(NotificationChannel.DASHBOARD) # Default fallback
        return channels
    except Exception as e:
        logger.error(f"Error fetching active channels: {e}")
        return [NotificationChannel.DASHBOARD]

async def check_alert_state(db: AsyncSession, alert_type: str, ref_id: str, cooldown_days: int = 1) -> bool:
    """Returns True if the alert should be sent (not in cooldown)."""
    try:
        result = await db.execute(
            select(AlertState)
            .where(AlertState.alert_type == alert_type, AlertState.reference_id == ref_id)
        )
        state = result.scalars().first()
        
        now = datetime.datetime.now(datetime.timezone.utc)
        if state:
            # Check if cooldown has passed
            if (now - state.last_triggered_at).days < cooldown_days:
                return False
            state.last_triggered_at = now
        else:
            new_state = AlertState(alert_type=alert_type, reference_id=ref_id, last_triggered_at=now)
            db.add(new_state)
        
        await db.commit()
        return True
    except Exception as e:
        await db.rollback()
        logger.error(f"Error checking alert state: {e}")
        return False

async def run_budget_alerts(db: AsyncSession):
    """1. Monitor spending against budgets and trigger thresholds."""
    try:
        current_month = datetime.date.today().strftime("%Y-%m")
        channels = await get_active_channels(db)

        # Get all budgets for current month
        budgets_res = await db.execute(
            select(Budget, Category).join(Category).where(Budget.month == current_month)
        )
        budgets = budgets_res.all()

        for budget, category in budgets:
            # Sum expenses for this category in current month
            spent_res = await db.execute(
                select(func.sum(Transaction.amount))
                .where(
                    Transaction.category_id == category.id,
                    Transaction.transaction_type == TransactionType.EXPENSE,
                    func.to_char(Transaction.date, 'YYYY-MM') == current_month
                )
            )
            spent = spent_res.scalar() or 0.0
            
            threshold_amt = budget.budgeted_amount * budget.alert_threshold
            if spent >= threshold_amt:
                ref_id = f"{budget.id}-{current_month}-threshold"
                if await check_alert_state(db, "BUDGET_THRESHOLD", ref_id, cooldown_days=7):
                    pct = (spent / budget.budgeted_amount) * 100
                    subject = f"Budget Alert: {category.name}"
                    body = f"You have spent ${spent:.2f} ({pct:.1f}%) of your ${budget.budgeted_amount:.2f} budget for {category.name}."
                    await dispatch_notification(db, subject, body, channels)
    except Exception as e:
        logger.exception(f"Error in run_budget_alerts: {e}")

async def run_bill_reminders(db: AsyncSession):
    """2. Send reminders before bills are due based on alert_days_before."""
    try:
        today = datetime.date.today()
        channels = await get_active_channels(db)

        bills_res = await db.execute(select(RecurringBill).where(RecurringBill.is_active == True))
        bills = bills_res.scalars().all()

        # Project bills out 30 days to handle all frequencies accurately
        end_date = today + datetime.timedelta(days=30)
        projections = project_bills(bills, today, end_date)

        for proj in projections:
            # Find the original bill to get alert_days_before
            original_bill = next((b for b in bills if str(b.id) == str(proj.bill_id)), None)
            if not original_bill:
                continue
            
            target_alert_date = proj.due_date - datetime.timedelta(days=original_bill.alert_days_before)
            
            if target_alert_date == today:
                ref_id = f"{proj.bill_id}-{proj.due_date}"
                if await check_alert_state(db, "BILL_REMINDER", ref_id, cooldown_days=30):
                    subject = f"Upcoming Bill: {proj.name}"
                    body = f"Your bill for {proj.name} (${proj.amount:.2f}) is due on {proj.due_date.strftime('%b %d')}."
                    await dispatch_notification(db, subject, body, channels)
    except Exception as e:
        logger.exception(f"Error in run_bill_reminders: {e}")

async def run_anomaly_detection(db: AsyncSession):
    """3. Detect unusual spending: large single txns, WoW spikes, new recurring."""
    try:
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        channels = await get_active_channels(db)

        # A. Large Single Transactions (>2x category average over last 90 days)
        recent_txns_res = await db.execute(
            select(Transaction, Category)
            .join(Category, isouter=True)
            .where(Transaction.date >= yesterday, Transaction.transaction_type == TransactionType.EXPENSE, Transaction.amount > 0)
        )
        recent_txns = recent_txns_res.all()

        for txn, cat in recent_txns:
            if not cat:
                continue
            # Get 90-day avg for this category
            avg_res = await db.execute(
                select(func.avg(Transaction.amount))
                .where(
                    Transaction.category_id == cat.id,
                    Transaction.transaction_type == TransactionType.EXPENSE,
                    Transaction.date >= today - datetime.timedelta(days=90),
                    Transaction.date < yesterday
                )
            )
            cat_avg = avg_res.scalar() or 0.0
            
            if cat_avg > 0 and txn.amount > (2 * cat_avg):
                if await check_alert_state(db, "LARGE_TRANSACTION", str(txn.id)):
                    subject = "Large Transaction Detected"
                    body = f"A transaction of ${txn.amount:.2f} at {txn.merchant or 'Unknown'} is more than 2x your usual spending in {cat.name}."
                    await dispatch_notification(db, subject, body, channels)

        # B. Week-over-Week Spending Spikes (Filtered by EXPENSE)
        week1_start = today - datetime.timedelta(days=7)
        week2_start = today - datetime.timedelta(days=14)

        w1_res = await db.execute(
            select(func.sum(Transaction.amount))
            .where(Transaction.date >= week1_start, Transaction.transaction_type == TransactionType.EXPENSE)
        )
        w1_spend = w1_res.scalar() or 0.0

        w2_res = await db.execute(
            select(func.sum(Transaction.amount))
            .where(Transaction.date >= week2_start, Transaction.date < week1_start, Transaction.transaction_type == TransactionType.EXPENSE)
        )
        w2_spend = w2_res.scalar() or 0.0

        if w2_spend > 0 and w1_spend > (w2_spend * 1.5):
            ref_id = f"wow-spike-{week1_start}"
            if await check_alert_state(db, "WOW_SPIKE", ref_id, cooldown_days=7):
                subject = "Spending Spike Detected"
                body = f"Your spending this past week (${w1_spend:.2f}) was significantly higher than the previous week (${w2_spend:.2f})."
                await dispatch_notification(db, subject, body, channels)

        # C. New Recurring Charges
        for txn, cat in recent_txns:
            if txn.is_recurring and txn.merchant:
                # Check if merchant existed before 30 days ago
                past_res = await db.execute(
                    select(Transaction.id)
                    .where(Transaction.merchant == txn.merchant, Transaction.date < today - datetime.timedelta(days=30))
                    .limit(1)
                )
                if not past_res.first():
                    if await check_alert_state(db, "NEW_RECURRING", f"{txn.merchant}-{today.month}"):
                        subject = "New Recurring Charge"
                        body = f"We detected a new recurring charge for {txn.merchant} (${txn.amount:.2f})."
                        await dispatch_notification(db, subject, body, channels)
    except Exception as e:
        logger.exception(f"Error in run_anomaly_detection: {e}")

async def run_cashflow_warnings(db: AsyncSession):
    """4. Warn when projected balances go negative within 14 days."""
    try:
        today = datetime.date.today()
        end_date = today + datetime.timedelta(days=14)
        channels = await get_active_channels(db)

        bills_res = await db.execute(select(RecurringBill).where(RecurringBill.is_active == True))
        bills = bills_res.scalars().all()

        # Synchronous calls to CashFlowEngine
        projections = project_bills(bills, today, end_date)
        forecast = aggregate_daily_forecast(projections, today, end_date)

        negative_date = next((day for day in forecast if day.net < 0), None)
        
        if negative_date:
            ref_id = f"negative-cashflow-{negative_date.date}"
            if await check_alert_state(db, "CASHFLOW_WARNING", ref_id, cooldown_days=7):
                subject = "Cash Flow Warning"
                body = f"Your projected balance may drop below zero on {negative_date.date.strftime('%b %d')}."
                await dispatch_notification(db, subject, body, channels)
    except Exception as e:
        logger.exception(f"Error in run_cashflow_warnings: {e}")

async def run_ai_insights(db: AsyncSession):
    """5. Use Anthropic API to generate weekly insights and store as Recommendation."""
    try:
        if not settings.anthropic_api_key:
            logger.warning("Anthropic API key not set. Skipping AI insights.")
            return

        today = datetime.date.today()
        last_week = today - datetime.timedelta(days=7)

        # Check if we already generated an insight this week
        ref_id = f"ai-insight-{last_week.strftime('%Y-%W')}"
        if not await check_alert_state(db, "AI_INSIGHT", ref_id, cooldown_days=6):
            return

        # Gather data for prompt
        txns_res = await db.execute(
            select(Transaction.merchant, Transaction.amount, Category.name)
            .join(Category, isouter=True)
            .where(Transaction.date >= last_week, Transaction.transaction_type == TransactionType.EXPENSE)
        )
        txns = txns_res.all()
        
        if not txns:
            return

        txn_text = "\n".join([f"- {t.merchant or 'Unknown'}: ${t.amount:.2f} ({t.name or 'Uncategorized'})" for t in txns])
        prompt = f"Analyze these expenses from the last 7 days and provide 1 brief, actionable financial insight:\n{txn_text}"

        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        response = await client.messages.create(
            model=settings.ai_model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        insight_text = response.content[0].text

        # Store as Recommendation
        rec = Recommendation(
            recommendation_type=RecommendationType.GENERAL_INSIGHT,
            status=RecommendationStatus.PENDING,
            title="Weekly AI Spending Insight",
            summary=insight_text[:200] + "..." if len(insight_text) > 200 else insight_text,
            detail=insight_text,
            priority=1  # Lower number = higher priority
        )
        db.add(rec)
        await db.commit()

        # Also notify via channels
        channels = await get_active_channels(db)
        await dispatch_notification(db, "Weekly AI Insight Ready", "A new AI-generated insight has been added to your recommendations.", channels)

    except Exception as e:
        await db.rollback()
        logger.exception(f"Error in run_ai_insights: {e}")
```

### 5. Background Scheduler (`src/core/alerts/scheduler.py`)
Provides the async loop to run the engine in the background.

```python
import asyncio
import logging
from src.db.engine import async_session
from src.core.alerts.rules import (
    run_budget_alerts,
    run_bill_reminders,
    run_anomaly_detection,
    run_cashflow_warnings,
    run_ai_insights
)
from src.models.alerts import AlertPreference
from sqlalchemy import select

logger = logging.getLogger(__name__)

async def alert_engine_task():
    """Background task that runs periodically to check alerts."""
    logger.info("Alert engine started.")
    while True:
        try:
            async with async_session() as db:
                # Check if system is active for user
                pref_res = await db.execute(select(AlertPreference).limit(1))
                pref = pref_res.scalars().first()

                if pref:
                    if pref.enable_budget_alerts:
                        await run_budget_alerts(db)
                    if pref.enable_bill_reminders:
                        await run_bill_reminders(db)
                    if pref.enable_anomaly_detection:
                        await run_anomaly_detection(db)
                    if pref.enable_cashflow_warnings:
                        await run_cashflow_warnings(db)
                    if pref.enable_ai_insights:
                        await run_ai_insights(db)
        except Exception as e:
            logger.exception(f"Critical error in alert engine loop: {e}")
        
        # Run every 6 hours
        await asyncio.sleep(60 * 60 * 6)
```

### 6. API Router (`src/api/routers/alerts.py`)
Endpoints for preferences and history, using proper error handling.

```python
import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.deps import get_db
from src.models.alerts import AlertPreference
from src.models.notification import Notification
from src.schemas.alerts import AlertPreferenceUpdate, AlertPreferenceResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/alerts", tags=["alerts"])

# Dummy user ID for single-user platform context
DEFAULT_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")

@router.get("/preferences", response_model=AlertPreferenceResponse)
async def get_preferences(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(AlertPreference).where(AlertPreference.user_id == DEFAULT_USER_ID))
        pref = result.scalars().first()
        
        if not pref:
            pref = AlertPreference(user_id=DEFAULT_USER_ID)
            db.add(pref)
            await db.commit()
            await db.refresh(pref)
            
        return pref
    except Exception as e:
        logger.exception("Failed to fetch alert preferences")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/preferences", response_model=AlertPreferenceResponse)
async def update_preferences(
    payload: AlertPreferenceUpdate,
    db: AsyncSession = Depends(get_db)
):
    try:
        result = await db.execute(select(AlertPreference).where(AlertPreference.user_id == DEFAULT_USER_ID))
        pref = result.scalars().first()
        
        if not pref:
            raise HTTPException(status_code=404, detail="Preferences not found")

        update_data = payload.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(pref, key, value)
            
        await db.commit()
        await db.refresh(pref)
        return pref
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.exception("Failed to update alert preferences")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/history")
async def get_alert_history(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(
            select(Notification)
            .order_by(Notification.created_at.desc())
            .limit(50)
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
    except Exception as e:
        logger.exception("Failed to fetch alert history")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 7. Frontend Interface (`src/pages/Alerts.tsx`)
React page handling state, mutations, and API errors robustly.

```tsx
import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from './client';

interface AlertPreferences {
  enable_budget_alerts: boolean;
  enable_bill_reminders: boolean;
  enable_anomaly_detection: boolean;
  enable_cashflow_warnings: boolean;
  enable_ai_insights: boolean;
  notify_via_dashboard: boolean;
  notify_via_telegram: boolean;
}

interface NotificationHistory {
  id: string;
  channel: string;
  subject: string;
  body: string;
  sent_ok: boolean;
  created_at: string;
}

export default function Alerts() {
  const queryClient = useQueryClient();

  const { data: preferences, isError: isPrefError, error: prefError, isLoading: isPrefLoading } = useQuery<AlertPreferences>({
    queryKey: ['alert-preferences'],
    queryFn: async () => {
      const res = await apiClient.get('/alerts/preferences');
      return res.data;
    }
  });

  const { data: history, isError: isHistError, error: histError, isLoading: isHistLoading } = useQuery<NotificationHistory[]>({
    queryKey: ['alert-history'],
    queryFn: async () => {
      const res = await apiClient.get('/alerts/history');
      return res.data;
    }
  });

  const updatePrefs = useMutation({
    mutationFn: async (newPrefs: Partial<AlertPreferences>) => {
      await apiClient.put('/alerts/preferences', newPrefs);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alert-preferences'] });
    }
  });

  const handleToggle = (key: keyof AlertPreferences, currentValue: boolean) => {
    updatePrefs.mutate({ [key]: !currentValue });
  };

  if (isPrefLoading || isHistLoading) {
    return <div className="p-8 text-center text-gray-500">Loading alerts system...</div>;
  }

  if (isPrefError) {
    return <div className="p-8 text-red-500">Error loading preferences: {(prefError as Error).message}</div>;
  }

  if (isHistError) {
    return <div className="p-8 text-red-500">Error loading history: {(histError as Error).message}</div>;
  }

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-8">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Alerts & Notifications</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Settings Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Detection Rules</h2>
            <div className="space-y-4">
              {[
                { key: 'enable_budget_alerts', label: 'Budget Thresholds', desc: 'Warn when approaching limits' },
                { key: 'enable_bill_reminders', label: 'Bill Reminders', desc: 'Alert before bills are due' },
                { key: 'enable_anomaly_detection', label: 'Anomaly Detection', desc: 'Spikes and large transactions' },
                { key: 'enable_cashflow_warnings', label: 'Cash Flow Warnings', desc: 'Projected negative balances' },
                { key: 'enable_ai_insights', label: 'AI Weekly Insights', desc: 'Smart spending analysis' }
              ].map((item) => (
                <div key={item.key} className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-900">{item.label}</p>
                    <p className="text-xs text-gray-500">{item.desc}</p>
                  </div>
                  <button
                    onClick={() => handleToggle(item.key as keyof AlertPreferences, preferences![item.key as keyof AlertPreferences])}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      preferences![item.key as keyof AlertPreferences] ? 'bg-blue-600' : 'bg-gray-200'
                    }`}
                  >
                    <span className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                      preferences![item.key as keyof AlertPreferences] ? 'translate-x-5' : 'translate-x-0'
                    }`} />
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Delivery Channels</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-900">In-App Dashboard</p>
                  <p className="text-xs text-gray-500">Show alerts on screen</p>
                </div>
                <button
                  onClick={() => handleToggle('notify_via_dashboard', preferences!.notify_via_dashboard)}
                  className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                    preferences!.notify_via_dashboard ? 'bg-blue-600' : 'bg-gray-200'
                  }`}
                >
                  <span className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                    preferences!.notify_via_dashboard ? 'translate-x-5' : 'translate-x-0'
                  }`} />
                </button>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-900">Telegram Bot</p>
                  <p className="text-xs text-gray-500">Instant push messages</p>
                </div>
                <button
                  onClick={() => handleToggle('notify_via_telegram', preferences!.notify_via_telegram)}
                  className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                    preferences!.notify_via_telegram ? 'bg-blue-600' : 'bg-gray-200'
                  }`}
                >
                  <span className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                    preferences!.notify_via_telegram ? 'translate-x-5' : 'translate-x-0'
                  }`} />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* History Panel */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 h-[800px] flex flex-col">
            <div className="p-6 border-b border-gray-200 bg-gray-50 flex-shrink-0">
              <h2 className="text-xl font-semibold text-gray-800">Recent Notifications</h2>
            </div>
            
            <div className="p-6 overflow-y-auto flex-1 space-y-4">
              {!history || history.length === 0 ? (
                <p className="text-gray-500 text-sm text-center mt-10">No recent alerts found.</p>
              ) : (
                history.map((notification) => (
                  <div key={notification.id} className="p-4 rounded-lg border bg-white border-gray-100 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-semibold text-gray-900">{notification.subject}</h4>
                      <span className="text-xs text-gray-400">
                        {new Date(notification.created_at).toLocaleString()}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 whitespace-pre-wrap">{notification.body}</p>
                    <div className="mt-3 flex items-center gap-2">
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800">
                        {notification.channel}
                      </span>
                      {!notification.sent_ok && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800">
                          Delivery Failed
                        </span>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
```