# Task: Smart Budget Alert & Notification System for um-cfo

## Prompt

Build a smart budget alert and notification system for the um-cfo personal finance platform. The system should proactively monitor the user's financial state and send timely, actionable alerts through multiple channels.

### Core Capabilities:

1. **Budget Threshold Alerts** — Monitor spending against budgets and trigger alerts when approaching or exceeding thresholds. Use the existing `alert_threshold` field on the Budget model.

2. **Bill Due Date Reminders** — Use the existing RecurringBill model's `alert_days_before` field to send reminders before bills are due. Handle different bill frequencies correctly.

3. **Anomaly Detection** — Detect unusual spending patterns: large single transactions (>2x average for that category), sudden spending spikes (week-over-week), and new recurring charges.

4. **Cash Flow Warnings** — Integrate with the existing CashFlowEngine to warn when projected balances will go negative within the next 14 days.

5. **AI-Powered Insights** — Use the Anthropic API to generate a weekly digest with personalized financial insights and recommendations. Store these as Recommendation entries.

6. **Multi-Channel Delivery** — Use the existing Notification model and NotificationChannel enum to log all sent notifications. Implement actual delivery for at least Telegram (via bot API) and Dashboard (via in-app storage).

### Technical Requirements:
- All code must be async Python using SQLAlchemy 2.0 patterns
- Use the project's existing models, database patterns, and config — do NOT reinvent infrastructure
- The alert engine should be runnable as a background task (asyncio task or periodic scheduler)
- Include a FastAPI router for managing alert preferences and viewing alert history
- Frontend: a React page for viewing alerts and configuring notification preferences
- No TODOs, no stubs, no placeholders — fully working implementations
- All files must include proper error handling and logging

### What to Deliver:
Design the file structure yourself. Think about separation of concerns: monitoring logic, alert rules, notification dispatch, API layer, and frontend. The system should be production-ready, not a prototype.
