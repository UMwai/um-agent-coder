#!/usr/bin/env python3
"""Test harness for the Gemini Intelligence Layer on the deployed daemon.

Submits a complex financial advisor design task through each endpoint,
evaluates results, and reports.

Usage:
    python3 scripts/test_gemini_layer.py [BASE_URL]
"""

import asyncio
import json
import sys
import time

import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://um-agent-daemon-23o5bq3bfq-uc.a.run.app"
GEMINI = f"{BASE_URL}/api/gemini"

# The complex task: design an AI Financial Advisor engine for the um-cfo repo
COMPLEX_TASK = """You are designing the implementation for an AI Financial Advisor Engine
for a personal finance application (FastAPI + PostgreSQL + SQLAlchemy 2.0 async).

The existing codebase has:
- Account, Transaction, Budget, Bill, Debt, Investment models (SQLAlchemy ORM, UUID PKs)
- A CashFlowEngine that projects bills up to 365 days and forecasts daily balances
- A DebtOptimizer that runs avalanche/snowball simulations month-by-month
- A Recommendation model with types: BUDGET_ADJUSTMENT, DEBT_PAYMENT, INVESTMENT_REBALANCE,
  BILL_ALERT, SAVINGS_OPPORTUNITY, TAX_OPTIMIZATION, SPENDING_ALERT, CASHFLOW_WARNING, GENERAL_INSIGHT
- An approval workflow state machine: PENDING → APPROVED → EXECUTED or REJECTED
- Notification dispatcher (Discord + Slack webhooks)
- action_payload JSONB field on Recommendation for machine-readable execution instructions

Design the implementation for `src/core/ai/engine.py` — the main orchestrator that:
1. Pulls all accounts, last 90 days of transactions, budgets, bills, debts, investments
2. Summarizes financial data into structured metrics (avoid raw data in prompts)
3. Runs 6 analysis pipelines concurrently:
   - spending_analyzer: anomaly detection, category drift, merchant patterns
   - cashflow_advisor: wraps CashFlowEngine + adds narrative warnings
   - debt_strategist: wraps DebtOptimizer + recommends strategy changes
   - investment_monitor: portfolio drift, rebalancing alerts
   - savings_detector: subscription audit, optimization opportunities
   - tax_planner: tax-loss harvesting signals, estimated liability
4. Each pipeline generates typed Recommendation objects with:
   - confidence_score (0.0-1.0, threshold 0.7 to save)
   - impact_estimate (dollar amount)
   - urgency (IMMEDIATE/THIS_WEEK/THIS_MONTH/INFORMATIONAL)
   - action_payload (JSONB with executable parameters)
5. Deduplicates against existing pending recommendations
6. Saves to DB and dispatches urgent notifications

Provide:
- Complete Python class with async method signatures
- The gather_context() method that builds the summarized financial snapshot
- The prompt template each pipeline should use
- The deduplication and scoring logic
- How daily_analysis_job in the scheduler should call this engine

Be specific — include actual function signatures, data structures, and SQL queries."""


async def test_enhance(client: httpx.AsyncClient) -> dict:
    """Test 1: Enhanced query endpoint."""
    print("\n" + "=" * 70)
    print("TEST 1: POST /api/gemini/enhance")
    print("=" * 70)

    start = time.monotonic()
    resp = await client.post(
        f"{GEMINI}/enhance",
        json={
            "prompt": COMPLEX_TASK,
            "enable_enhancement": True,
            "enable_self_eval": True,
            "domain_hint": "code",
        },
        timeout=120.0,
    )
    elapsed = time.monotonic() - start

    if resp.status_code != 200:
        print(f"  FAILED: {resp.status_code} — {resp.text[:200]}")
        return {"test": "enhance", "passed": False, "error": resp.text[:200]}

    data = resp.json()
    print(f"  Status: {resp.status_code}")
    print(f"  Model: {data.get('model')}")
    print(f"  Duration: {data.get('duration_ms')}ms (wall: {elapsed:.1f}s)")
    print(f"  Response length: {len(data.get('response', ''))} chars")

    if data.get("enhancement"):
        enh = data["enhancement"]
        print(f"  Enhancement stages: {enh.get('stages_applied')}")
        print(f"  Complexity score: {enh.get('complexity_score')}")
        print(f"  Model selected: {enh.get('model_selected')}")

    if data.get("evaluation"):
        ev = data["evaluation"]
        print(f"  Eval score: {ev.get('score')}")
        print(f"  Accuracy: {ev.get('accuracy')}, Completeness: {ev.get('completeness')}")
        print(f"  Clarity: {ev.get('clarity')}, Actionability: {ev.get('actionability')}")
        print(f"  Issues: {ev.get('issues')}")
        print(f"  Retries: {ev.get('retry_count')}")

    # Show first 500 chars of response
    resp_text = data.get("response", "")
    print(f"\n  Response preview:\n  {'─' * 60}")
    for line in resp_text[:500].split("\n"):
        print(f"  {line}")
    if len(resp_text) > 500:
        print(f"  ... ({len(resp_text) - 500} more chars)")

    return {"test": "enhance", "passed": True, "data": data}


async def test_session(client: httpx.AsyncClient) -> dict:
    """Test 2: Multi-turn session — design then refine."""
    print("\n" + "=" * 70)
    print("TEST 2: Multi-turn session (create → message → follow-up)")
    print("=" * 70)

    # Create session
    resp = await client.post(
        f"{GEMINI}/sessions",
        json={
            "system_prompt": "You are a senior Python architect specializing in async FastAPI applications and financial systems.",
            "model": "pro",
            "temperature": 0.4,
        },
        timeout=30.0,
    )
    if resp.status_code != 200:
        print(f"  Create FAILED: {resp.status_code}")
        return {"test": "session", "passed": False}

    session = resp.json()
    sid = session["id"]
    print(f"  Session created: {sid}")
    print(f"  Model: {session['model']}, TTL expires: {session.get('expires_at', 'N/A')}")

    # Message 1: Initial design question
    resp = await client.post(
        f"{GEMINI}/sessions/{sid}/message",
        json={"content": COMPLEX_TASK, "enable_enhancement": True},
        timeout=120.0,
    )
    if resp.status_code != 200:
        print(f"  Message 1 FAILED: {resp.status_code} — {resp.text[:200]}")
        return {"test": "session", "passed": False}

    msg1 = resp.json()
    print(f"\n  Message 1 response:")
    print(f"    Tokens: {msg1.get('token_count')}, Duration: {msg1.get('duration_ms')}ms")
    print(f"    Enhanced: {msg1.get('enhancement_applied')}")
    print(f"    Preview: {msg1.get('content', '')[:200]}...")

    # Message 2: Follow-up that requires session context
    resp = await client.post(
        f"{GEMINI}/sessions/{sid}/message",
        json={
            "content": (
                "Now focus on the deduplication logic you described. "
                "The existing Recommendation model uses (type, title) as a soft unique key. "
                "Write the complete _deduplicate() method with proper SQL queries, "
                "considering that recommendations can be PENDING, APPROVED, or EXECUTED. "
                "Only deduplicate against PENDING ones."
            ),
        },
        timeout=120.0,
    )
    if resp.status_code != 200:
        print(f"  Message 2 FAILED: {resp.status_code}")
        return {"test": "session", "passed": False}

    msg2 = resp.json()
    print(f"\n  Message 2 (follow-up with context):")
    print(f"    Tokens: {msg2.get('token_count')}, Duration: {msg2.get('duration_ms')}ms")
    print(f"    Preview: {msg2.get('content', '')[:200]}...")

    # Verify context carried over (response should reference the engine design)
    has_context = any(
        kw in msg2.get("content", "").lower()
        for kw in ["engine", "pipeline", "gather_context", "recommendation", "deduplic"]
    )
    print(f"    Context preserved: {'YES' if has_context else 'NO'}")

    # List sessions to verify persistence
    resp = await client.get(f"{GEMINI}/sessions", timeout=10.0)
    sessions = resp.json()
    print(f"\n  Sessions listed: {sessions.get('total', 0)} total")

    # Get full session detail
    resp = await client.get(f"{GEMINI}/sessions/{sid}", timeout=10.0)
    if resp.status_code == 200:
        detail = resp.json()
        print(f"  Messages in session: {len(detail.get('messages', []))}")
        print(f"  Session total tokens: {detail['session'].get('total_tokens')}")
    else:
        print(f"  Session detail FAILED: {resp.status_code} — {resp.text[:200]}")

    # Cleanup
    await client.delete(f"{GEMINI}/sessions/{sid}", timeout=10.0)
    print(f"  Session deleted: {sid}")

    return {"test": "session", "passed": True, "context_preserved": has_context}


async def test_batch(client: httpx.AsyncClient) -> dict:
    """Test 3: Batch processing — submit 5 pipeline design queries."""
    print("\n" + "=" * 70)
    print("TEST 3: Batch processing (5 queries)")
    print("=" * 70)

    pipelines = [
        "Design the spending_analyzer.py pipeline: detect spending anomalies by comparing current month category totals vs 90-day averages. Flag merchants with >20% price increases. Output SPENDING_ALERT and BUDGET_ADJUSTMENT recommendations.",
        "Design the cashflow_advisor.py pipeline: wrap the existing CashFlowEngine (which returns daily_forecast list of {date, projected_balance, bills_due}). Generate CASHFLOW_WARNING when projected_balance drops below $500. Add natural language narrative.",
        "Design the debt_strategist.py pipeline: wrap the existing DebtOptimizer (avalanche vs snowball simulation). Compare strategies and recommend switching if savings > $100. Output DEBT_PAYMENT recommendations with action_payload containing the new strategy params.",
        "Design the savings_detector.py pipeline: identify recurring subscription charges by merchant pattern matching on transactions. Flag duplicates (e.g., multiple streaming services). Detect idle cash in low-yield accounts. Output SAVINGS_OPPORTUNITY recommendations.",
        "Design the tax_planner.py pipeline: analyze investment positions for tax-loss harvesting opportunities (unrealized losses > $1000). Estimate quarterly tax liability from brokerage gains. Output TAX_OPTIMIZATION recommendations.",
    ]

    resp = await client.post(
        f"{GEMINI}/batch",
        json={
            "queries": [{"prompt": p, "model": "auto"} for p in pipelines],
            "temperature": 0.4,
            "max_tokens": 4096,
            "enable_enhancement": True,
        },
        timeout=30.0,
    )
    if resp.status_code != 200:
        print(f"  Submit FAILED: {resp.status_code} — {resp.text[:200]}")
        return {"test": "batch", "passed": False}

    batch = resp.json()
    bid = batch["id"]
    print(f"  Batch submitted: {bid}")
    print(f"  Status: {batch['status']}, Queries: {batch['total_queries']}")

    # Poll until complete (max 5 minutes)
    for i in range(30):
        await asyncio.sleep(10)
        resp = await client.get(f"{GEMINI}/batch/{bid}", timeout=10.0)
        batch = resp.json()
        print(f"  Poll {i+1}: status={batch['status']} completed={batch['completed_queries']}/{batch['total_queries']} failed={batch['failed_queries']}")
        if batch["status"] in ("completed", "failed", "cancelled"):
            break

    if batch.get("results"):
        for r in batch["results"]:
            status = "OK" if r.get("response") else f"ERR: {r.get('error', 'unknown')}"
            resp_len = len(r.get("response", "")) if r.get("response") else 0
            print(f"    [{r['index']}] {r.get('model', '?')}: {status} ({resp_len} chars, {r.get('duration_ms', 0)}ms)")

    return {"test": "batch", "passed": batch["status"] == "completed", "data": batch}


async def test_agent(client: httpx.AsyncClient) -> dict:
    """Test 4: Agent task with tool use."""
    print("\n" + "=" * 70)
    print("TEST 4: Agent task (tool-use loop)")
    print("=" * 70)

    resp = await client.post(
        f"{GEMINI}/agent",
        json={
            "task": (
                "I need to understand the file structure of /home/umwai/um-cfo/src/core/ "
                "to design the AI engine. List the files in that directory, then read "
                "the __init__.py of the cashflow module to understand its interface. "
                "Finally, summarize what you found and recommend the method signatures "
                "for the new AI engine class."
            ),
            "tools": ["file_list", "file_read", "summarize"],
            "max_steps": 8,
            "model": "pro-3.1",
            "temperature": 0.3,
        },
        timeout=180.0,
    )
    if resp.status_code != 200:
        print(f"  FAILED: {resp.status_code} — {resp.text[:200]}")
        return {"test": "agent", "passed": False}

    data = resp.json()
    print(f"  Status: {data.get('status')}")
    print(f"  Steps: {data.get('total_steps')}")
    print(f"  Duration: {data.get('duration_ms')}ms")

    for step in data.get("steps", []):
        print(f"\n  Step {step['step']}:")
        if step.get("thought"):
            print(f"    Thought: {step['thought'][:100]}...")
        if step.get("action"):
            a = step["action"]
            print(f"    Action: {a['tool']}({json.dumps(a['args'])})")
            if a.get("result"):
                print(f"    Result: {a['result'][:100]}...")

    if data.get("answer"):
        print(f"\n  Final Answer preview:")
        for line in data["answer"][:300].split("\n"):
            print(f"    {line}")

    return {"test": "agent", "passed": data.get("status") in ("completed", "max_steps_reached")}


async def main():
    print(f"Gemini Intelligence Layer Test Harness")
    print(f"Target: {BASE_URL}")
    print(f"{'=' * 70}")

    # Health check
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/health", timeout=10.0)
        health = resp.json()
        print(f"Health: {health['status']} (v{health['version']})")

        results = []

        # Run tests sequentially (each is meaningful on its own)
        results.append(await test_enhance(client))
        results.append(await test_session(client))
        results.append(await test_batch(client))
        results.append(await test_agent(client))

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for r in results:
            status = "PASS" if r.get("passed") else "FAIL"
            print(f"  [{status}] {r['test']}")

        passed = sum(1 for r in results if r.get("passed"))
        print(f"\n  {passed}/{len(results)} tests passed")


if __name__ == "__main__":
    asyncio.run(main())
