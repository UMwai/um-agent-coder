# Eval Context: um-cfo Budget Alert System

## Actual API Signatures & Models (from um-cfo source code)

### Base Model (src/models/base.py):
- All models inherit from `BaseModel` which provides: `id` (UUID, default=uuid.uuid4), `created_at` (datetime, server_default=func.now()), `updated_at` (datetime, server_default=func.now(), onupdate=func.now())
- The declarative `Base` class lives in `src/db/engine.py`

### FinancialAccount (src/models/account.py):
- Table: `financial_accounts`
- Fields: name (String200), account_type (AccountType enum), institution (String200, nullable), balance (Float, default=0.0), currency (String3, default="USD"), is_active (bool, default=True)
- AccountType enum: CHECKING, SAVINGS, CREDIT, INVESTMENT, BROKERAGE, CRYPTO, MORTGAGE, LOAN, CASH, OTHER
- Relationship: transactions (list[Transaction], back_populates="account")

### Transaction (src/models/transaction.py):
- Table: `transactions`
- Fields: date (Date, index), amount (Float), description (String500), transaction_type (TransactionType enum), account_id (FK->financial_accounts.id), category_id (FK->categories.id, nullable), fingerprint (String64, unique), merchant (String200, nullable), notes (Text, nullable), is_pending (bool), is_recurring (bool)
- TransactionType enum: INCOME, EXPENSE, TRANSFER
- Relationships: account (FinancialAccount), category (Category, nullable)

### Category (src/models/category.py):
- Table: `categories`
- Fields: name (String100), category_type (CategoryType enum), parent_id (FK->categories.id, nullable), icon (String50, nullable), color (String7 hex, nullable), is_system (bool, default=False)
- CategoryType enum: INCOME, EXPENSE, TRANSFER
- Relationships: parent (Category, nullable), transactions (list[Transaction]), budgets (list[Budget])

### Budget (src/models/budget.py):
- Table: `budgets`
- Unique constraint: (month, category_id)
- Fields: month (String7, format "YYYY-MM", index), category_id (FK->categories.id), budgeted_amount (Float), alert_threshold (Float, default=0.9), notes (String500, nullable)
- Relationship: category (Category, back_populates="budgets")

### RecurringBill (src/models/bill.py):
- Table: `recurring_bills`
- Fields: name (String200), amount (Float), frequency (BillFrequency enum), next_due_date (Date, index), category_id (FK, nullable), account_id (FK, nullable), is_auto_detected (bool, default=False), is_active (bool, default=True), merchant (String200, nullable), alert_days_before (int, default=3)
- BillFrequency enum: WEEKLY, BIWEEKLY, MONTHLY, QUARTERLY, SEMIANNUAL, ANNUAL

### Recommendation (src/models/recommendation.py):
- Table: `recommendations`
- Fields: recommendation_type (RecommendationType enum), status (RecommendationStatus enum, default=PENDING), title (String300), summary (Text), detail (Text, nullable), impact_amount (Float, nullable), confidence (Float 0.0-1.0, nullable), priority (int, default=0, lower=higher priority), action_payload (JSONB, nullable), rejection_reason (Text, nullable)
- RecommendationType enum: BUDGET_ADJUSTMENT, DEBT_PAYMENT, INVESTMENT_REBALANCE, BILL_ALERT, SAVINGS_OPPORTUNITY, TAX_OPTIMIZATION, SPENDING_ALERT, CASHFLOW_WARNING, GENERAL_INSIGHT
- RecommendationStatus enum: PENDING, APPROVED, REJECTED, EXPIRED, EXECUTED

### Notification (src/models/notification.py):
- Table: `notification_log`
- NotificationChannel enum: TELEGRAM, DISCORD, SLACK, EMAIL, DASHBOARD
- Fields: channel, subject (String300), body (Text), metadata_json (JSONB), sent_ok (bool), error (Text)

### Database Session Pattern (src/api/deps.py):
```python
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.engine import async_session

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```
- Usage in routes: `db: AsyncSession = Depends(get_db)`
- Queries: `result = await db.execute(select(Model).where(...))`, `items = result.scalars().all()`

### Router Registration Pattern (src/api/app.py):
- No API version prefix — all routers at root level
- Pattern: `app.include_router(module.router)` where router has `prefix="/name"` and `tags=["name"]`
- Existing prefixes: /accounts, /categories, /transactions, /budgets, /dashboard, /investments, /debts, /bills, /recommendations, /sync

### Config/Settings (src/config.py):
- `from src.config import settings`
- Key fields: anthropic_api_key (str), ai_model (str, default="claude-sonnet-4-20250514"), database_url, debug, app_mode, telegram_bot_token (str, optional), telegram_chat_id (str, optional)
- Singleton: `settings = Settings()` imported everywhere

### CashFlowEngine (src/core/cashflow/__init__.py):
```python
def project_bills(bills: list, start_date: date, end_date: date) -> list[ProjectedBillPayment]
def aggregate_daily_forecast(projections: list[ProjectedBillPayment], start_date: date, end_date: date, actual_income_by_date: dict[date, float] | None = None, actual_expenses_by_date: dict[date, float] | None = None) -> list[DailyForecastRow]
```
- These are plain `def` functions, NOT async, NOT class methods
- ProjectedBillPayment: bill_id (UUID), name, amount, due_date, frequency (BillFrequency)
- DailyForecastRow: date, actual_income (float, default=0.0), actual_expenses (float, default=0.0), projected_expenses (float, default=0.0), net (property)

### Frontend Patterns:
- React 19 + Vite + TypeScript + Tailwind CSS + Recharts + @tanstack/react-query
- API client: `import { apiClient } from './client'` then `apiClient.get('/endpoint')`, `apiClient.post('/endpoint', data)`
- Charts: recharts with ResponsiveContainer
- Routing: react-router-dom v6
- Components: Tailwind utility classes

### Schema Pattern for CRUD:
```python
class XCreate(BaseModel): ...   # input fields
class XUpdate(BaseModel): ...   # all fields Optional
class XResponse(BaseModel):     # full model output
    model_config = {"from_attributes": True}
```

## CRITICAL CHECKS:
1. Models MUST inherit from project's BaseModel, NOT SQLAlchemy's DeclarativeBase directly
2. DB session MUST use `Depends(get_db)` from `src/api/deps`
3. CashFlowEngine functions are plain `def`, NOT async — do NOT call with `await`
4. Budget month format is "YYYY-MM" string
5. Router prefix: no API version prefix
6. AI must use Anthropic via `settings.anthropic_api_key` and `settings.ai_model`
7. Frontend must import apiClient from `./client`
8. Notification logging uses existing Notification model — do NOT create new tables for it
9. Recommendation entries use existing RecommendationType enum values
10. priority field: lower number = higher priority
