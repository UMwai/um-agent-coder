#!/usr/bin/env python3
"""A/B comparison: Budget Alerts task — checklist eval vs legacy eval.

Submits the same task twice:
  A) Baseline: legacy eval (completeness/clarity/actionability use 1-10 scale)
  B) New: checklist eval (all 5 dimensions use pass/fail checklists)

Compares iteration count, scores, and output quality side by side.

Usage:
    python3 scripts/submit_budget_alerts_ab.py
    python3 scripts/submit_budget_alerts_ab.py --max-iterations 5 --threshold 0.85
    python3 scripts/submit_budget_alerts_ab.py --run-b-only   # skip baseline, just run new
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

BASE = "https://um-agent-daemon-23o5bq3bfq-uc.a.run.app"
TASK_DIR = Path(__file__).parent.parent / "evals" / "budget-alerts"
POLL_INTERVAL = 15  # seconds
LONG_POLL_INTERVAL = 30  # after 5 min


def load_task() -> tuple[str, str]:
    """Load task prompt and eval context from files."""
    task_md = (TASK_DIR / "task.md").read_text()
    eval_ctx = (TASK_DIR / "eval_context.md").read_text()

    # Extract just the prompt section
    prompt_start = task_md.index("## Prompt")
    prompt = task_md[prompt_start + len("## Prompt"):].strip()

    return prompt, eval_ctx


async def submit_and_poll(
    client: httpx.AsyncClient,
    prompt: str,
    eval_context: str,
    max_iterations: int,
    threshold: float,
    label: str,
) -> dict:
    """Submit iteration task and poll until completion."""
    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    print(f"  Prompt: {len(prompt):,} chars")
    print(f"  Eval context: {len(eval_context):,} chars")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Threshold: {threshold}")
    print()

    resp = await client.post(
        f"{BASE}/api/gemini/iterate",
        json={
            "prompt": prompt,
            "eval_context": eval_context,
            "model": "pro-3.1",
            "max_iterations": max_iterations,
            "score_threshold": threshold,
            "temperature": 0.7,
            "max_tokens": 65536,
            "enable_enhancement": True,
            "use_multi_turn": True,
            "domain_hint": "code",
        },
    )

    if resp.status_code != 200:
        print(f"  SUBMIT FAILED: {resp.status_code}")
        print(f"  {resp.text[:500]}")
        return {"error": f"Submit failed: {resp.status_code}"}

    data = resp.json()
    iteration_id = data["id"]
    print(f"  Submitted: {iteration_id}")
    print()

    # Poll
    start = time.monotonic()
    while True:
        await asyncio.sleep(POLL_INTERVAL if (time.monotonic() - start) < 300 else LONG_POLL_INTERVAL)
        elapsed = time.monotonic() - start

        resp = await client.get(f"{BASE}/api/gemini/iterate/{iteration_id}")
        if resp.status_code != 200:
            print(f"  [{elapsed:5.0f}s] Poll error: {resp.status_code}")
            continue

        data = resp.json()
        status = data["status"]
        steps = data.get("steps", [])
        best_score = data.get("best_score", 0)

        # Progress line
        step_scores = []
        for s in steps:
            ev = s.get("evaluation") or {}
            score = ev.get("score", 0)
            acc = ev.get("accuracy", 0)
            ful = ev.get("fulfillment", 0)
            step_scores.append(f"s{s['step']}={score:.2f}")

        progress = " | ".join(step_scores) if step_scores else "generating..."
        print(f"  [{elapsed:5.0f}s] {status} — best={best_score:.3f} — {progress}")

        if status not in ("running",):
            break

    return data


def print_step_detail(step: dict, label: str):
    """Print detailed step info."""
    ev = step.get("evaluation") or {}
    print(f"\n  --- {label} Step {step['step']} ---")
    print(f"    Tokens: {step.get('generation_tokens', 0):,}")
    print(f"    Duration: {step.get('generation_duration_ms', 0) / 1000:.1f}s")
    if ev:
        print(f"    Score:         {ev.get('score', 0):.3f}")
        print(f"    Accuracy:      {ev.get('accuracy', 0):.3f}")
        print(f"    Completeness:  {ev.get('completeness', 0):.3f}")
        print(f"    Clarity:       {ev.get('clarity', 0):.3f}")
        print(f"    Actionability: {ev.get('actionability', 0):.3f}")
        print(f"    Fulfillment:   {ev.get('fulfillment', 0):.3f}")

        # Count checks by type
        for check_type in ["accuracy_checks", "fulfillment_checks",
                           "completeness_checks", "clarity_checks",
                           "actionability_checks"]:
            checks = ev.get(check_type, [])
            if checks:
                passed = sum(1 for c in checks if c.get("status") == "pass")
                failed = sum(1 for c in checks if c.get("status") == "fail")
                print(f"    {check_type}: {passed}/{len(checks)} passed")

    strategies = step.get("strategies_applied", [])
    if strategies:
        print(f"    Strategies: {', '.join(strategies)}")

    resp_text = step.get("response", "")
    print(f"    Response: {len(resp_text):,} chars")


def compare_results(result_a: dict, result_b: dict):
    """Print side-by-side comparison."""
    print(f"\n{'=' * 80}")
    print("  A/B COMPARISON")
    print(f"{'=' * 80}")

    cols = [
        ("Status", "status"),
        ("Best Score", "best_score"),
        ("Best Iteration", "best_iteration"),
        ("Total Iterations", "total_iterations"),
        ("Total Tokens", "total_tokens"),
        ("Duration (s)", "duration_ms"),
    ]

    print(f"\n  {'Metric':<22} {'A (Baseline)':<20} {'B (Checklist)':<20} {'Delta':<15}")
    print(f"  {'-' * 75}")

    for label, key in cols:
        a_val = result_a.get(key, 0)
        b_val = result_b.get(key, 0)

        if key == "duration_ms":
            a_str = f"{a_val / 1000:.1f}s"
            b_str = f"{b_val / 1000:.1f}s"
            delta = f"{(b_val - a_val) / 1000:+.1f}s"
        elif key == "status":
            a_str = str(a_val)
            b_str = str(b_val)
            delta = ""
        elif isinstance(a_val, float):
            a_str = f"{a_val:.3f}"
            b_str = f"{b_val:.3f}"
            delta = f"{b_val - a_val:+.3f}"
        else:
            a_str = str(a_val)
            b_str = str(b_val)
            delta = f"{b_val - a_val:+d}" if isinstance(a_val, int) else ""

        print(f"  {label:<22} {a_str:<20} {b_str:<20} {delta:<15}")

    # Per-dimension comparison on final step
    steps_a = result_a.get("steps", [])
    steps_b = result_b.get("steps", [])

    if steps_a and steps_b:
        best_a = max(steps_a, key=lambda s: (s.get("evaluation") or {}).get("score", 0))
        best_b = max(steps_b, key=lambda s: (s.get("evaluation") or {}).get("score", 0))
        ev_a = best_a.get("evaluation") or {}
        ev_b = best_b.get("evaluation") or {}

        print(f"\n  Best Step Dimension Comparison:")
        print(f"  {'Dimension':<22} {'A (best)':<20} {'B (best)':<20} {'Delta':<15}")
        print(f"  {'-' * 75}")

        for dim in ["accuracy", "completeness", "clarity", "actionability", "fulfillment"]:
            a = ev_a.get(dim, 0)
            b = ev_b.get(dim, 0)
            print(f"  {dim:<22} {a:<20.3f} {b:<20.3f} {b - a:+.3f}")

        # Check counts
        print(f"\n  Checklist Details (best step):")
        for check_type in ["accuracy_checks", "fulfillment_checks",
                           "completeness_checks", "clarity_checks",
                           "actionability_checks"]:
            checks_a = ev_a.get(check_type, [])
            checks_b = ev_b.get(check_type, [])
            if checks_a or checks_b:
                p_a = sum(1 for c in checks_a if c.get("status") == "pass")
                p_b = sum(1 for c in checks_b if c.get("status") == "pass")
                label_a = f"{p_a}/{len(checks_a)}" if checks_a else "N/A"
                label_b = f"{p_b}/{len(checks_b)}" if checks_b else "N/A"
                print(f"  {check_type:<30} A: {label_a:<12} B: {label_b}")

    # Key insight
    a_score = result_a.get("best_score", 0)
    b_score = result_b.get("best_score", 0)
    a_iters = result_a.get("total_iterations", 0)
    b_iters = result_b.get("total_iterations", 0)

    print(f"\n  SUMMARY:")
    if b_score < a_score and b_iters >= a_iters:
        print(f"  → Checklist eval scores LOWER (more honest) — expected behavior")
        print(f"  → The baseline likely had inflated completeness/clarity/actionability")
    elif b_score >= a_score and b_iters <= a_iters:
        print(f"  → Checklist eval matched or exceeded baseline in fewer iterations!")
    elif b_iters > a_iters:
        print(f"  → Checklist eval took more iterations but gave more granular feedback")
    print()


async def main(args):
    prompt, eval_context = load_task()

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:

        result_a = None
        result_b = None

        if not args.run_b_only:
            # Run A: Baseline (before deploying new code)
            result_a = await submit_and_poll(
                client, prompt, eval_context,
                max_iterations=args.max_iterations,
                threshold=args.threshold,
                label="RUN A — Baseline (current deployed code)",
            )

            # Save A results
            if result_a and not result_a.get("error"):
                a_id = result_a.get("id", "unknown")
                save_data_a = _strip_large_fields(result_a)
                (TASK_DIR / f"results-A-{a_id[:8]}.json").write_text(
                    json.dumps(save_data_a, indent=2)
                )
                best = result_a.get("best_response", "")
                if best:
                    (TASK_DIR / f"response-A-{a_id[:8]}.md").write_text(best)

                for step in result_a.get("steps", []):
                    print_step_detail(step, "A")

        # Run B: New checklist eval (after deploying new code)
        result_b = await submit_and_poll(
            client, prompt, eval_context,
            max_iterations=args.max_iterations,
            threshold=args.threshold,
            label="RUN B — Checklist Eval (new code)",
        )

        # Save B results
        if result_b and not result_b.get("error"):
            b_id = result_b.get("id", "unknown")
            save_data_b = _strip_large_fields(result_b)
            (TASK_DIR / f"results-B-{b_id[:8]}.json").write_text(
                json.dumps(save_data_b, indent=2)
            )
            best = result_b.get("best_response", "")
            if best:
                (TASK_DIR / f"response-B-{b_id[:8]}.md").write_text(best)

            for step in result_b.get("steps", []):
                print_step_detail(step, "B")

        # Compare if both ran
        if result_a and result_b and not result_a.get("error") and not result_b.get("error"):
            compare_results(result_a, result_b)
        elif result_b and not result_b.get("error"):
            print(f"\n  Run B completed: score={result_b.get('best_score', 0):.3f} "
                  f"in {result_b.get('total_iterations', 0)} iterations")
            print(f"  To compare, run again after deploying the other version")

    print(f"\n  Results saved to: {TASK_DIR}/")


def _strip_large_fields(data: dict) -> dict:
    """Strip large text fields for JSON storage."""
    save = {**data}
    if "best_response" in save:
        save["best_response_chars"] = len(save.pop("best_response", ""))
    for s in save.get("steps", []):
        if "response" in s:
            s["response_chars"] = len(s.pop("response", ""))
        if "prompt_sent" in s:
            s["prompt_sent_chars"] = len(s.pop("prompt_sent", ""))
    return save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A/B test: Budget Alerts with checklist eval vs legacy eval"
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--run-b-only", action="store_true",
                        help="Skip baseline run, only run B (checklist eval)")
    args = parser.parse_args()

    asyncio.run(main(args))
