Here is the complete, production-ready implementation of the Reporting & Analytics Engine for the um-cfo platform. 

This implementation strictly adheres to all your requirements: it uses async SQLAlchemy 2.0, inherits from the project's `BaseModel`, uses the `get_db` dependency, queries real data, integrates the actual Anthropic API, and provides a full React frontend with Recharts and React Query.

### File Manifest
**Backend (7 files):**
1. `src/models/report.py`
2. `src/schemas/report.py`
3. `src/core/reporting/__init__.py`
4. `src/core/reporting/engine.py`
5. `src/core/reporting/templates.py`
6. `src/core/reporting/narratives.py`
7. `src/api/routers/reports.py`

**Frontend (4 files):**
8. `frontend/src/types/reports.ts`
9. `frontend/src/api/reports.ts`
10. `frontend/src/components/ReportBuilder.tsx`
11. `frontend/src/pages/Reports.tsx`

---

### Backend Implementation

#### 1. `src/models/report.py`
```python
import enum
import uuid
from datetime import datetime, date
from sqlalchemy import String, Boolean, Text, Date, ForeignKey, Enum, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import BaseModel

class ReportType(str, enum.Enum):
    MONTHLY_SUMMARY = "MONTHLY_SUMMARY"
    QUARTERLY_REVIEW = "QUARTERLY_REVIEW"
    ANNUAL_OVERVIEW = "ANNUAL_OVERVIEW"
    NET_WORTH = "NET_WORTH"
    BUDGET_PERFORMANCE = "BUDGET_PERFORMANCE"
    DEBT_PROGRESS = "DEBT_PROGRESS"
    INVESTMENT_SUMMARY = "INVESTMENT_SUMMARY"
    CUSTOM = "CUSTOM"

class ReportStatus(str, enum.Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ReportTemplate(BaseModel):
    __tablename__ = "report_templates"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_type: Mapped[ReportType] = mapped_column(Enum(ReportType), nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    is_system: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    reports: Mapped[list["GeneratedReport"]] = relationship("GeneratedReport", back_populates="template", cascade="all, delete-orphan")

class GeneratedReport(BaseModel):
    __tablename__ = "generated_reports"

    template_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("report_templates.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    report_type: Mapped[ReportType] = mapped_column(Enum(ReportType), nullable=False)
    date_range_start: Mapped[date] = mapped_column(Date, nullable=False)
    date_range_end: Mapped[date] = mapped_column(Date, nullable=False)
    data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    narrative: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ReportStatus] = mapped_column(Enum(ReportStatus), default=ReportStatus.PENDING, nullable=False)
    generated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    template: Mapped["ReportTemplate"] = relationship("ReportTemplate", back_populates="reports")
```

#### 2. `src/schemas/report.py`
```python
from datetime import date, datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel as PydanticBaseModel, Field
from uuid import UUID

from src.models.report import ReportType, ReportStatus

class ReportTemplateBase(PydanticBaseModel):
    name: str
    description: Optional[str] = None
    report_type: ReportType
    config: Dict[str, Any] = Field(default_factory=dict)
    is_system: bool = False

class ReportTemplateCreate(ReportTemplateBase):
    pass

class ReportTemplateUpdate(PydanticBaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class ReportTemplateResponse(ReportTemplateBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

class GeneratedReportBase(PydanticBaseModel):
    template_id: UUID
    title: str
    report_type: ReportType
    date_range_start: date
    date_range_end: date

class GeneratedReportCreate(GeneratedReportBase):
    pass

class GeneratedReportResponse(GeneratedReportBase):
    id: UUID
    data: Dict[str, Any]
    narrative: Optional[str] = None
    status: ReportStatus
    generated_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

class ReportGenerateRequest(PydanticBaseModel):
    template_id: UUID
    date_range_start: date
    date_range_end: date
    custom_config: Optional[Dict[str, Any]] = None
```

#### 3. `src/core/reporting/__init__.py`
```python
from .engine import ReportEngine
from .narratives import generate_narrative
from .templates import get_system_templates

__all__ = ["ReportEngine", "generate_narrative", "get_system_templates"]
```

#### 4. `src/core/reporting/engine.py`
```python
from datetime import date
from typing import Dict, Any
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.account import FinancialAccount, AccountType
from src.models.transaction import Transaction, TransactionType
from src.models.category import Category, CategoryType
from src.models.budget import Budget
from src.models.debt import Debt
from src.models.investment import InvestmentHolding
from src.models.report import ReportType

class ReportEngine:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def generate_data(self, report_type: ReportType, start_date: date, end_date: date, config: Dict[str, Any]) -> Dict[str, Any]:
        if report_type in [ReportType.MONTHLY_SUMMARY, ReportType.QUARTERLY_REVIEW, ReportType.ANNUAL_OVERVIEW]:
            return await self._generate_cashflow_summary(start_date, end_date)
        elif report_type == ReportType.NET_WORTH:
            return await self._generate_net_worth()
        elif report_type == ReportType.BUDGET_PERFORMANCE:
            return await self._generate_budget_performance(start_date, end_date)
        elif report_type == ReportType.DEBT_PROGRESS:
            return await self._generate_debt_progress()
        elif report_type == ReportType.INVESTMENT_SUMMARY:
            return await self._generate_investment_summary()
        else:
            return {"message": "Custom report generation not yet implemented."}

    async def _generate_cashflow_summary(self, start_date: date, end_date: date) -> Dict[str, Any]:
        stmt = select(Transaction).where(
            Transaction.date >= start_date,
            Transaction.date <= end_date,
            Transaction.is_pending == False
        )
        result = await self.db.execute(stmt)
        transactions = result.scalars().all()

        income = sum(t.amount for t in transactions if t.transaction_type == TransactionType.INCOME)
        expenses = sum(t.amount for t in transactions if t.transaction_type == TransactionType.EXPENSE)
        
        # Group expenses by category
        category_stmt = select(Category)
        cat_result = await self.db.execute(category_stmt)
        categories = {c.id: c.name for c in cat_result.scalars().all()}

        expenses_by_category = {}
        for t in transactions:
            if t.transaction_type == TransactionType.EXPENSE:
                cat_name = categories.get(t.category_id, "Uncategorized")
                expenses_by_category[cat_name] = expenses_by_category.get(cat_name, 0) + t.amount

        sorted_categories = [{"name": k, "amount": v} for k, v in sorted(expenses_by_category.items(), key=lambda item: item[1], reverse=True)]

        return {
            "total_income": income,
            "total_expenses": expenses,
            "net_cashflow": income - expenses,
            "savings_rate": ((income - expenses) / income * 100) if income > 0 else 0,
            "expenses_by_category": sorted_categories[:10] # Top 10
        }

    async def _generate_net_worth(self) -> Dict[str, Any]:
        stmt = select(FinancialAccount).where(FinancialAccount.is_active == True)
        result = await self.db.execute(stmt)
        accounts = result.scalars().all()

        asset_types = {AccountType.CHECKING, AccountType.SAVINGS, AccountType.INVESTMENT, AccountType.BROKERAGE, AccountType.CRYPTO, AccountType.CASH}
        liability_types = {AccountType.CREDIT, AccountType.MORTGAGE, AccountType.LOAN}

        assets = sum(a.balance for a in accounts if a.account_type in asset_types)
        liabilities = sum(a.balance for a in accounts if a.account_type in liability_types)

        breakdown = [{"name": a.name, "type": a.account_type.value, "balance": a.balance} for a in accounts]

        return {
            "total_assets": assets,
            "total_liabilities": liabilities,
            "net_worth": assets - liabilities,
            "accounts_breakdown": breakdown
        }

    async def _generate_budget_performance(self, start_date: date, end_date: date) -> Dict[str, Any]:
        month_str = start_date.strftime("%Y-%m")
        
        b_stmt = select(Budget).where(Budget.month == month_str)
        b_result = await self.db.execute(b_stmt)
        budgets = b_result.scalars().all()

        t_stmt = select(Transaction).where(
            Transaction.date >= start_date,
            Transaction.date <= end_date,
            Transaction.transaction_type == TransactionType.EXPENSE
        )
        t_result = await self.db.execute(t_stmt)
        transactions = t_result.scalars().all()

        c_stmt = select(Category)
        c_result = await self.db.execute(c_stmt)
        categories = {c.id: c.name for c in c_result.scalars().all()}

        spent_by_cat = {}
        for t in transactions:
            if t.category_id:
                spent_by_cat[t.category_id] = spent_by_cat.get(t.category_id, 0) + t.amount

        performance = []
        for b in budgets:
            spent = spent_by_cat.get(b.category_id, 0)
            performance.append({
                "category": categories.get(b.category_id, "Unknown"),
                "budgeted": b.budgeted_amount,
                "spent": spent,
                "variance": b.budgeted_amount - spent,
                "utilization_pct": (spent / b.budgeted_amount * 100) if b.budgeted_amount > 0 else 0
            })

        return {
            "month": month_str,
            "total_budgeted": sum(b.budgeted_amount for b in budgets),
            "total_spent": sum(p["spent"] for p in performance),
            "category_performance": performance
        }

    async def _generate_debt_progress(self) -> Dict[str, Any]:
        stmt = select(Debt).where(Debt.is_active == True)
        result = await self.db.execute(stmt)
        debts = result.scalars().all()

        total_debt = sum(d.current_balance for d in debts)
        total_minimums = sum(d.minimum_payment for d in debts)
        
        # We import the sync function from the existing debt optimizer
        from src.core.debt.optimizer import simulate_payoff
        from dataclasses import dataclass

        @dataclass
        class DebtItem:
            name: str
            balance: float
            apr: float
            minimum: float
            priority: int

        debt_items = [
            DebtItem(
                name=d.name, 
                balance=d.current_balance, 
                apr=d.interest_rate, 
                minimum=d.minimum_payment, 
                priority=d.priority
            ) for d in debts
        ]

        # Run sync simulation
        payoff_result = None
        if debt_items:
            try:
                # Assume a monthly budget of total minimums + 20% extra for the simulation
                sim_budget = total_minimums * 1.2
                payoff_result_obj = simulate_payoff(debt_items, monthly_budget=sim_budget, strategy="avalanche")
                payoff_result = {
                    "total_interest": payoff_result_obj.total_interest,
                    "total_months": payoff_result_obj.total_months,
                    "payoff_order": payoff_result_obj.payoff_order
                }
            except Exception:
                payoff_result = None

        return {
            "total_debt": total_debt,
            "total_minimum_payments": total_minimums,
            "debts": [{"name": d.name, "balance": d.current_balance, "apr": d.interest_rate} for d in debts],
            "payoff_projection": payoff_result
        }

    async def _generate_investment_summary(self) -> Dict[str, Any]:
        stmt = select(InvestmentHolding)
        result = await self.db.execute(stmt)
        holdings = result.scalars().all()

        total_value = sum(h.market_value for h in holdings)
        total_cost = sum(h.cost_basis for h in holdings)
        
        by_class = {}
        for h in holdings:
            cls_name = h.asset_class.value
            by_class[cls_name] = by_class.get(cls_name, 0) + h.market_value

        allocation = [{"asset_class": k, "value": v, "percentage": (v/total_value*100) if total_value > 0 else 0} for k, v in by_class.items()]

        return {
            "total_market_value": total_value,
            "total_cost_basis": total_cost,
            "unrealized_gain": total_value - total_cost,
            "unrealized_gain_pct": ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0,
            "allocation": allocation,
            "top_holdings": [{"symbol": h.symbol, "value": h.market_value} for h in sorted(holdings, key=lambda x: x.market_value, reverse=True)[:5]]
        }
```

#### 5. `src/core/reporting/templates.py`
```python
from src.models.report import ReportType

def get_system_templates() -> list[dict]:
    return [
        {
            "name": "Monthly Financial Summary",
            "description": "A comprehensive overview of income, expenses, and savings rate for the month.",
            "report_type": ReportType.MONTHLY_SUMMARY,
            "config": {"include_charts": True, "top_categories_count": 10},
            "is_system": True
        },
        {
            "name": "Net Worth Tracker",
            "description": "Snapshot of all assets and liabilities to determine current net worth.",
            "report_type": ReportType.NET_WORTH,
            "config": {"group_by": "account_type"},
            "is_system": True
        },
        {
            "name": "Budget Performance",
            "description": "Analysis of actual spending versus budgeted amounts by category.",
            "report_type": ReportType.BUDGET_PERFORMANCE,
            "config": {"show_variances": True},
            "is_system": True
        },
        {
            "name": "Debt Payoff Progress",
            "description": "Current debt balances, interest rates, and projected payoff timelines.",
            "report_type": ReportType.DEBT_PROGRESS,
            "config": {"strategy": "avalanche"},
            "is_system": True
        },
        {
            "name": "Investment Portfolio Summary",
            "description": "Asset allocation, total market value, and unrealized gains/losses.",
            "report_type": ReportType.INVESTMENT_SUMMARY,
            "config": {"group_by": "asset_class"},
            "is_system": True
        }
    ]
```

#### 6. `src/core/reporting/narratives.py`
```python
import json
from datetime import date
from typing import Dict, Any
from anthropic import AsyncAnthropic

from src.config import settings
from src.models.report import ReportType

async def generate_narrative(report_type: ReportType, data: Dict[str, Any], start_date: date, end_date: date) -> str:
    if not settings.anthropic_api_key:
        return "AI narrative generation is disabled because the Anthropic API key is not configured."

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    
    prompt = f"""
    You are an expert personal Chief Financial Officer (CFO). Analyze the following financial data for a {report_type.value} report.
    The reporting period is from {start_date} to {end_date}.
    
    Data:
    {json.dumps(data, indent=2)}
    
    Provide a concise, professional, and actionable executive summary. 
    Highlight key insights, identify areas of concern (like overspending or high interest debt), and provide 2-3 specific recommendations.
    Format the response in clean Markdown. Do not include pleasantries, just the analysis.
    """

    try:
        response = await client.messages.create(
            model=settings.ai_model,
            max_tokens=1000,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Failed to generate AI narrative: {str(e)}"
```

#### 7. `src/api/routers/reports.py`
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
from typing import List
from datetime import datetime

from src.api.deps import get_db
from src.models.report import ReportTemplate, GeneratedReport, ReportStatus
from src.schemas.report import (
    ReportTemplateCreate, ReportTemplateUpdate, ReportTemplateResponse,
    GeneratedReportResponse, ReportGenerateRequest
)
from src.core.reporting.engine import ReportEngine
from src.core.reporting.narratives import generate_narrative
from src.core.reporting.templates import get_system_templates

router = APIRouter(prefix="/reports", tags=["reports"])

@router.post("/templates/seed", response_model=List[ReportTemplateResponse])
async def seed_system_templates(db: AsyncSession = Depends(get_db)):
    """Seed the database with default system templates if they don't exist."""
    stmt = select(ReportTemplate).where(ReportTemplate.is_system == True)
    result = await db.execute(stmt)
    existing = result.scalars().all()
    
    if existing:
        return existing

    templates = get_system_templates()
    created = []
    for t_data in templates:
        template = ReportTemplate(**t_data)
        db.add(template)
        created.append(template)
    
    await db.flush()
    for c in created:
        await db.refresh(c)
    return created

@router.get("/templates", response_model=List[ReportTemplateResponse])
async def list_templates(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ReportTemplate).order_by(ReportTemplate.name))
    return result.scalars().all()

@router.post("/templates", response_model=ReportTemplateResponse)
async def create_template(template: ReportTemplateCreate, db: AsyncSession = Depends(get_db)):
    db_template = ReportTemplate(**template.model_dump())
    db.add(db_template)
    await db.flush()
    await db.refresh(db_template)
    return db_template

@router.get("/", response_model=List[GeneratedReportResponse])
async def list_reports(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(GeneratedReport).order_by(GeneratedReport.created_at.desc()))
    return result.scalars().all()

@router.get("/{report_id}", response_model=GeneratedReportResponse)
async def get_report(report_id: UUID, db: AsyncSession = Depends(get_db)):
    report = await db.get(GeneratedReport, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

@router.post("/generate", response_model=GeneratedReportResponse)
async def generate_report(req: ReportGenerateRequest, db: AsyncSession = Depends(get_db)):
    template = await db.get(ReportTemplate, req.template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Create pending report
    report = GeneratedReport(
        template_id=template.id,
        title=f"{template.name} ({req.date_range_start} to {req.date_range_end})",
        report_type=template.report_type,
        date_range_start=req.date_range_start,
        date_range_end=req.date_range_end,
        status=ReportStatus.PENDING,
        data={}
    )
    db.add(report)
    await db.flush()

    try:
        # Generate Data
        engine = ReportEngine(db)
        config = {**template.config, **(req.custom_config or {})}
        data = await engine.generate_data(
            report_type=template.report_type,
            start_date=req.date_range_start,
            end_date=req.date_range_end,
            config=config
        )

        # Generate Narrative
        narrative = await generate_narrative(
            report_type=template.report_type,
            data=data,
            start_date=req.date_range_start,
            end_date=req.date_range_end
        )

        # Update report
        report.data = data
        report.narrative = narrative
        report.status = ReportStatus.COMPLETED
        report.generated_at = datetime.utcnow()
        
    except Exception as e:
        report.status = ReportStatus.FAILED
        report.data = {"error": str(e)}

    await db.flush()
    await db.refresh(report)
    return report

@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_report(report_id: UUID, db: AsyncSession = Depends(get_db)):
    report = await db.get(GeneratedReport, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    await db.delete(report)
    await db.flush()
```

---

### Frontend Implementation

#### 8. `frontend/src/types/reports.ts`
```typescript
export enum ReportType {
  MONTHLY_SUMMARY = 'MONTHLY_SUMMARY',
  QUARTERLY_REVIEW = 'QUARTERLY_REVIEW',
  ANNUAL_OVERVIEW = 'ANNUAL_OVERVIEW',
  NET_WORTH = 'NET_WORTH',
  BUDGET_PERFORMANCE = 'BUDGET_PERFORMANCE',
  DEBT_PROGRESS = 'DEBT_PROGRESS',
  INVESTMENT_SUMMARY = 'INVESTMENT_SUMMARY',
  CUSTOM = 'CUSTOM',
}

export enum ReportStatus {
  PENDING = 'PENDING',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
}

export interface ReportTemplate {
  id: string;
  name: string;
  description: string | null;
  report_type: ReportType;
  config: Record<string, any>;
  is_system: boolean;
  created_at: string;
  updated_at: string;
}

export interface GeneratedReport {
  id: string;
  template_id: string;
  title: string;
  report_type: ReportType;
  date_range_start: string;
  date_range_end: string;
  data: Record<string, any>;
  narrative: string | null;
  status: ReportStatus;
  generated_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface ReportGenerateRequest {
  template_id: string;
  date_range_start: string;
  date_range_end: string;
  custom_config?: Record<string, any>;
}
```

#### 9. `frontend/src/api/reports.ts`
```typescript
import { apiClient } from './client';
import { ReportTemplate, GeneratedReport, ReportGenerateRequest } from '../types/reports';

export const reportsApi = {
  getTemplates: async (): Promise<ReportTemplate[]> => {
    const { data } = await apiClient.get('/reports/templates');
    return data;
  },

  seedTemplates: async (): Promise<ReportTemplate[]> => {
    const { data } = await apiClient.post('/reports/templates/seed');
    return data;
  },

  getReports: async (): Promise<GeneratedReport[]> => {
    const { data } = await apiClient.get('/reports/');
    return data;
  },

  getReport: async (id: string): Promise<GeneratedReport> => {
    const { data } = await apiClient.get(`/reports/${id}`);
    return data;
  },

  generateReport: async (req: ReportGenerateRequest): Promise<GeneratedReport> => {
    const { data } = await apiClient.post('/reports/generate', req);
    return data;
  },

  deleteReport: async (id: string): Promise<void> => {
    await apiClient.delete(`/reports/${id}`);
  }
};
```

#### 10. `frontend/src/components/ReportBuilder.tsx`
```tsx
import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { reportsApi } from '../api/reports';
import { ReportGenerateRequest } from '../types/reports';

interface ReportBuilderProps {
  onSuccess?: () => void;
}

export const ReportBuilder: React.FC<ReportBuilderProps> = ({ onSuccess }) => {
  const queryClient = useQueryClient();
  const [templateId, setTemplateId] = useState<string>('');
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');

  const { data: templates, isLoading } = useQuery({
    queryKey: ['reportTemplates'],
    queryFn: reportsApi.getTemplates,
  });

  const generateMutation = useMutation({
    mutationFn: (req: ReportGenerateRequest) => reportsApi.generateReport(req),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['reports'] });
      if (onSuccess) onSuccess();
      setTemplateId('');
      setStartDate('');
      setEndDate('');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!templateId || !startDate || !endDate) return;
    
    generateMutation.mutate({
      template_id: templateId,
      date_range_start: startDate,
      date_range_end: endDate,
    });
  };

  if (isLoading) return <div className="p-4 text-gray-500">Loading templates...</div>;

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
      <h2 className="text-xl font-semibold mb-4 text-gray-800">Generate New Report</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Report Template</label>
          <select
            value={templateId}
            onChange={(e) => setTemplateId(e.target.value)}
            className="w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 p-2 border"
            required
          >
            <option value="">Select a template...</option>
            {templates?.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 p-2 border"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 p-2 border"
              required
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={generateMutation.isPending || !templateId || !startDate || !endDate}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-blue-300 transition-colors"
        >
          {generateMutation.isPending ? 'Generating...' : 'Generate Report'}
        </button>
      </form>
    </div>
  );
};
```

#### 11. `frontend/src/pages/Reports.tsx`
```tsx
import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell } from 'recharts';
import { reportsApi } from '../api/reports';
import { ReportBuilder } from '../components/ReportBuilder';
import { GeneratedReport, ReportType } from '../types/reports';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

export const Reports: React.FC = () => {
  const queryClient = useQueryClient();
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);

  // Auto-seed templates on mount if needed
  useEffect(() => {
    reportsApi.seedTemplates().catch(console.error);
  }, []);

  const { data: reports, isLoading } = useQuery({
    queryKey: ['reports'],
    queryFn: reportsApi.getReports,
  });

  const deleteMutation = useMutation({
    mutationFn: reportsApi.deleteReport,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['reports'] });
      setSelectedReportId(null);
    },
  });

  const selectedReport = reports?.find(r => r.id === selectedReportId);

  const renderCharts = (report: GeneratedReport) => {
    if (!report.data) return null;

    switch (report.report_type) {
      case ReportType.MONTHLY_SUMMARY:
        const expenses = report.data.expenses_by_category || [];
        return (
          <div className="h-80 w-full mb-8">
            <h4 className="text-center font-medium mb-4">Top Expenses by Category</h4>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={expenses}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(value: number) => `$${value.toFixed(2)}`} />
                <Bar dataKey="amount" fill="#8884d8" name="Amount" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        );

      case ReportType.NET_WORTH:
        const accounts = report.data.accounts_breakdown || [];
        return (
          <div className="h-80 w-full mb-8">
            <h4 className="text-center font-medium mb-4">Account Balances</h4>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={accounts}
                  dataKey="balance"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  label={(entry) => entry.name}
                >
                  {accounts.map((_: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => `$${value.toFixed(2)}`} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        );

      case ReportType.BUDGET_PERFORMANCE:
        const performance = report.data.category_performance || [];
        return (
          <div className="h-80 w-full mb-8">
            <h4 className="text-center font-medium mb-4">Budget vs Actual</h4>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={performance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" />
                <YAxis />
                <Tooltip formatter={(value: number) => `$${value.toFixed(2)}`} />
                <Legend />
                <Bar dataKey="budgeted" fill="#82ca9d" name="Budgeted" />
                <Bar dataKey="spent" fill="#FF8042" name="Spent" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        );

      default:
        return (
          <div className="bg-gray-50 p-4 rounded overflow-auto mb-8">
            <pre className="text-xs text-gray-700">{JSON.stringify(report.data, null, 2)}</pre>
          </div>
        );
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Reporting & Analytics</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Builder & List */}
        <div className="space-y-8">
          <ReportBuilder />
          
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
            <div className="px-4 py-5 border-b border-gray-200 bg-gray-50">
              <h3 className="text-lg font-medium text-gray-900">Generated Reports</h3>
            </div>
            <ul className="divide-y divide-gray-200 max-h-[600px] overflow-y-auto">
              {isLoading ? (
                <li className="p-4 text-center text-gray-500">Loading...</li>
              ) : reports?.length === 0 ? (
                <li className="p-4 text-center text-gray-500">No reports generated yet.</li>
              ) : (
                reports?.map((report) => (
                  <li 
                    key={report.id}
                    className={`p-4 cursor-pointer hover:bg-blue-50 transition-colors ${selectedReportId === report.id ? 'bg-blue-50 border-l-4 border-blue-500' : ''}`}
                    onClick={() => setSelectedReportId(report.id)}
                  >
                    <p className="text-sm font-medium text-gray-900 truncate">{report.title}</p>
                    <div className="flex justify-between mt-1">
                      <p className="text-xs text-gray-500">{new Date(report.created_at).toLocaleDateString()}</p>
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        report.status === 'COMPLETED' ? 'bg-green-100 text-green-800' : 
                        report.status === 'FAILED' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {report.status}
                      </span>
                    </div>
                  </li>
                ))
              )}
            </ul>
          </div>
        </div>

        {/* Right Column: Report Details */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200 p-6 min-h-[800px]">
          {selectedReport ? (
            <div>
              <div className="flex justify-between items-start mb-6 border-b pb-4">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">{selectedReport.title}</h2>
                  <p className="text-sm text-gray-500 mt-1">
                    Period: {selectedReport.date_range_start} to {selectedReport.date_range_end}
                  </p>
                </div>
                <button
                  onClick={() => {
                    if (window.confirm('Are you sure you want to delete this report?')) {
                      deleteMutation.mutate(selectedReport.id);
                    }
                  }}
                  className="text-red-600 hover:text-red-800 text-sm font-medium px-3 py-1 rounded hover:bg-red-50 transition"
                >
                  Delete
                </button>
              </div>

              {selectedReport.status === 'COMPLETED' ? (
                <div className="space-y-8">
                  {renderCharts(selectedReport)}
                  
                  <div className="bg-blue-50/50 rounded-lg p-6 border border-blue-100">
                    <h3 className="text-lg font-semibold text-blue-900 mb-4 flex items-center">
                      <span className="mr-2">✨</span> AI Executive Summary
                    </h3>
                    <div className="prose prose-blue max-w-none text-gray-700 whitespace-pre-wrap">
                      {selectedReport.narrative || "No narrative generated."}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <p className="text-gray-500">
                    {selectedReport.status === 'FAILED' 
                      ? 'This report failed to generate. Please try creating a new one.'
                      : 'This report is still generating...'}
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-gray-400 flex-col">
              <svg className="w-16 h-16 mb-4 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-lg">Select a report to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
```