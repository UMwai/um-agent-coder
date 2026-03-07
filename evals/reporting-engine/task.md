# Task: Full-Stack Reporting Engine for um-cfo

## Prompt

Implement a complete Reporting & Analytics Engine for the um-cfo personal finance platform (src/core/reporting/ + src/api/routers/reports.py + frontend). The system should allow users to generate, customize, and view financial reports with AI-powered narrative insights.

### Required Files (11 total):

**Backend (7 files):**
1. `src/models/report.py` — SQLAlchemy models: ReportTemplate (name, description, report_type, config JSONB, is_system bool) and GeneratedReport (template_id FK, title, report_type, date_range_start, date_range_end, data JSONB, narrative text, status enum, generated_at)
2. `src/schemas/report.py` — Pydantic request/response schemas for templates and reports
3. `src/core/reporting/__init__.py` — Package init, export public API
4. `src/core/reporting/engine.py` — ReportEngine class: gathers data from existing models (accounts, transactions, budgets, debts, investments, bills), computes metrics (income/expense trends, budget variance, net worth over time, category analysis, debt progress), produces structured report data
5. `src/core/reporting/templates.py` — Built-in report template definitions (monthly summary, quarterly review, annual overview, net worth tracker, budget performance, debt progress, investment summary, custom)
6. `src/core/reporting/narratives.py` — AI narrative generator using Anthropic API: takes structured report data + report type, produces human-readable analysis with insights and recommendations
7. `src/api/routers/reports.py` — Full CRUD for templates + reports, generate endpoint, list/filter/delete

**Frontend (4 files):**
8. `frontend/src/api/reports.ts` — API client functions for all report endpoints
9. `frontend/src/pages/Reports.tsx` — Report list view with filtering, report detail viewer with charts and narrative display
10. `frontend/src/components/ReportBuilder.tsx` — Custom report configuration UI: select metrics, date ranges, groupings, visualization preferences
11. `frontend/src/types/reports.ts` — TypeScript interfaces matching backend schemas

### Requirements:
- All backend code must be async and use SQLAlchemy 2.0 async patterns
- Models must inherit from the project's BaseModel (provides id, created_at, updated_at)
- Use the existing database session pattern with `get_db` dependency
- ReportEngine must query real data from existing models — no mock/fake data
- AI narratives must actually call the Anthropic API (not stubs)
- Frontend must use React Query for data fetching, Recharts for visualizations
- All 11 files must be present and contain complete, runnable implementations
- No TODOs, no stubs, no placeholders
