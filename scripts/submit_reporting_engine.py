#!/usr/bin/env python3
"""Submit the Reporting Engine task to the iteration runner.

Usage:
    python3 scripts/submit_reporting_engine.py [--max-iterations N] [--threshold F]
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

BASE = "https://um-agent-daemon-23o5bq3bfq-uc.a.run.app"

TASK_DIR = Path(__file__).parent.parent / "evals" / "reporting-engine"


def load_task() -> tuple[str, str]:
    """Load task prompt and eval context from files."""
    task_md = (TASK_DIR / "task.md").read_text()
    eval_ctx = (TASK_DIR / "eval_context.md").read_text()

    # Extract just the prompt section from task.md
    prompt_start = task_md.index("## Prompt")
    prompt = task_md[prompt_start + len("## Prompt") :].strip()

    return prompt, eval_ctx


async def submit(max_iterations: int, threshold: float):
    prompt, eval_context = load_task()

    print(f"Prompt: {len(prompt):,} chars")
    print(f"Eval context: {len(eval_context):,} chars")
    print(f"Max iterations: {max_iterations}")
    print(f"Score threshold: {threshold}")
    print()

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        # Submit
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
            print(f"SUBMIT FAILED: {resp.status_code}")
            print(resp.text[:500])
            return

        data = resp.json()
        iteration_id = data["id"]
        print(f"Submitted: {iteration_id}")
        print(f"Status: {data['status']}")
        print()

        # Poll for completion
        print("Polling for completion...")
        poll_interval = 15  # seconds
        start = time.monotonic()

        while True:
            await asyncio.sleep(poll_interval)
            elapsed = time.monotonic() - start

            resp = await client.get(f"{BASE}/api/gemini/iterate/{iteration_id}")
            if resp.status_code != 200:
                print(f"Poll error: {resp.status_code}")
                continue

            data = resp.json()
            status = data["status"]
            steps = data.get("steps", [])
            best_score = data.get("best_score", 0)

            # Show progress
            step_scores = []
            for s in steps:
                ev = s.get("evaluation", {})
                score = ev.get("score", 0) if ev else 0
                acc = ev.get("accuracy", 0) if ev else 0
                step_scores.append(f"s{s['step']}={score:.2f}(acc={acc:.2f})")

            progress = " | ".join(step_scores) if step_scores else "generating..."
            print(f"  [{elapsed:5.0f}s] {status} — best={best_score:.3f} — {progress}")

            if status not in ("running",):
                break

            # Increase interval for long-running tasks
            if elapsed > 300:
                poll_interval = 30

        # Final results
        print(f"\n{'=' * 80}")
        print(f"FINAL STATUS: {status}")
        print(f"BEST SCORE: {data.get('best_score', 0):.3f}")
        print(f"BEST ITERATION: {data.get('best_iteration', 0)}")
        print(f"TOTAL ITERATIONS: {data.get('total_iterations', 0)}")
        print(f"TOTAL TOKENS: {data.get('total_tokens', 0):,}")
        print(f"DURATION: {data.get('duration_ms', 0) / 1000:.1f}s")
        print(f"{'=' * 80}")

        # Show per-step detail
        for s in data.get("steps", []):
            ev = s.get("evaluation", {})
            print(f"\n--- Step {s['step']} ---")
            print(f"  Model: {s.get('generation_model', '?')}")
            print(f"  Tokens: {s.get('generation_tokens', 0):,}")
            print(f"  Duration: {s.get('generation_duration_ms', 0) / 1000:.1f}s")
            print(f"  Finish: {s.get('finish_reason', '?')}")
            if ev:
                print(f"  Score: {ev.get('score', 0):.3f}")
                print(f"  Accuracy: {ev.get('accuracy', 0):.3f}")
                print(f"  Completeness: {ev.get('completeness', 0):.3f}")
                print(f"  Clarity: {ev.get('clarity', 0):.3f}")
                print(f"  Actionability: {ev.get('actionability', 0):.3f}")
                print(f"  Fulfillment: {ev.get('fulfillment', 0):.3f}")

                # Show accuracy checks
                checks = ev.get("accuracy_checks", [])
                if checks:
                    passed = [c for c in checks if c["status"] == "pass"]
                    failed = [c for c in checks if c["status"] == "fail"]
                    print(f"  Accuracy checks: {len(passed)}/{len(checks)} passed")
                    for c in failed:
                        print(f"    FAIL [{c['severity']}] {c['check']}: {c['detail']}")

                # Show fulfillment checks
                f_checks = ev.get("fulfillment_checks", [])
                if f_checks:
                    f_passed = [c for c in f_checks if c["status"] == "pass"]
                    f_failed = [c for c in f_checks if c["status"] == "fail"]
                    print(f"  Fulfillment checks: {len(f_passed)}/{len(f_checks)} passed")
                    for c in f_failed:
                        print(f"    FAIL [{c['severity']}] {c['check']}: {c['detail']}")

                issues = ev.get("issues", [])
                if issues:
                    print(f"  Issues: {len(issues)}")
                    for issue in issues[:5]:
                        print(f"    - {issue}")

            strategies = s.get("strategies_applied", [])
            if strategies:
                print(f"  Strategies: {', '.join(strategies)}")

            # Response length
            resp_text = s.get("response", "")
            print(f"  Response: {len(resp_text):,} chars")

        # Save best response
        best = data.get("best_response", "")
        if best:
            out_path = TASK_DIR / f"response-{iteration_id[:8]}.md"
            out_path.write_text(best)
            print(f"\nBest response saved to: {out_path}")

        # Save full results
        results_path = TASK_DIR / f"results-{iteration_id[:8]}.json"
        # Remove large response text from JSON to keep it manageable
        save_data = {**data}
        if "best_response" in save_data:
            save_data["best_response_chars"] = len(save_data.pop("best_response", ""))
        for s in save_data.get("steps", []):
            if "response" in s:
                s["response_chars"] = len(s.pop("response", ""))
            if "prompt_sent" in s:
                s["prompt_sent_chars"] = len(s.pop("prompt_sent", ""))
        results_path.write_text(json.dumps(save_data, indent=2))
        print(f"Results saved to: {results_path}")

        if data.get("error"):
            print(f"\nERROR: {data['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit Reporting Engine task to iteration runner")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    asyncio.run(submit(args.max_iterations, args.threshold))
