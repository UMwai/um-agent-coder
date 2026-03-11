#!/usr/bin/env python3
"""Submit the Volatility Regime Classifier eval task.

Usage:
    python3 scripts/submit_vol_regime.py
    python3 scripts/submit_vol_regime.py --max-iterations 5 --threshold 0.85
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

BASE = "https://um-agent-daemon-23o5bq3bfq-uc.a.run.app"
TASK_DIR = Path(__file__).parent.parent / "evals" / "vol-regime-classifier"
POLL_INTERVAL = 15
LONG_POLL_INTERVAL = 30


def load_task() -> tuple[str, str]:
    task_md = (TASK_DIR / "task.md").read_text()
    eval_ctx = (TASK_DIR / "eval_context.md").read_text()
    prompt_start = task_md.index("## Prompt")
    prompt = task_md[prompt_start + len("## Prompt"):].strip()
    return prompt, eval_ctx


async def main(args):
    prompt, eval_context = load_task()

    print(f"Prompt: {len(prompt):,} chars")
    print(f"Eval context: {len(eval_context):,} chars")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Threshold: {args.threshold}")
    print()

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        resp = await client.post(
            f"{BASE}/api/gemini/iterate",
            json={
                "prompt": prompt,
                "eval_context": eval_context,
                "model": "pro-3.1",
                "max_iterations": args.max_iterations,
                "score_threshold": args.threshold,
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
        print()

        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            interval = POLL_INTERVAL if elapsed < 300 else LONG_POLL_INTERVAL
            await asyncio.sleep(interval)

            resp = await client.get(f"{BASE}/api/gemini/iterate/{iteration_id}")
            if resp.status_code != 200:
                print(f"[{elapsed:5.0f}s] Poll error: {resp.status_code}")
                continue

            data = resp.json()
            status = data["status"]
            steps = data.get("steps", [])
            best_score = data.get("best_score", 0)

            step_scores = []
            for s in steps:
                ev = s.get("evaluation") or {}
                score = ev.get("score", 0)
                step_scores.append(f"s{s['step']}={score:.2f}")

            progress = " | ".join(step_scores) if step_scores else "generating..."
            print(f"[{elapsed:5.0f}s] {status} — best={best_score:.3f} — {progress}")

            if status not in ("running",):
                break

        # Print results
        for s in data.get("steps", []):
            ev = s.get("evaluation") or {}
            print(f"\n--- Step {s['step']} ---")
            print(f"  Tokens: {s.get('generation_tokens', 0):,}")
            print(f"  Duration: {s.get('generation_duration_ms', 0) / 1000:.1f}s")
            if ev:
                print(f"  Score:         {ev.get('score', 0):.3f}")
                for dim in ["accuracy", "completeness", "clarity", "actionability", "fulfillment"]:
                    print(f"  {dim.capitalize():<15}{ev.get(dim, 0):.3f}")
                for ct in ["accuracy_checks", "fulfillment_checks", "completeness_checks",
                           "clarity_checks", "actionability_checks"]:
                    checks = ev.get(ct, [])
                    if checks:
                        passed = sum(1 for c in checks if c.get("status") == "pass")
                        print(f"  {ct}: {passed}/{len(checks)} passed")
                        for c in checks:
                            if c.get("status") == "fail":
                                print(f"    FAIL [{c.get('severity', '?')}] {c['check']}: {c.get('detail', '')}")

            strategies = s.get("strategies_applied", [])
            if strategies:
                print(f"  Strategies: {', '.join(strategies)}")

        # Save results
        if not data.get("error"):
            rid = data.get("id", "unknown")[:8]
            save = {**data}
            if "best_response" in save:
                save["best_response_chars"] = len(save.pop("best_response", ""))
            for s in save.get("steps", []):
                if "response" in s:
                    s["response_chars"] = len(s.pop("response", ""))
                if "prompt_sent" in s:
                    s["prompt_sent_chars"] = len(s.pop("prompt_sent", ""))
            (TASK_DIR / f"results-{rid}.json").write_text(json.dumps(save, indent=2))

            best_resp = data.get("best_response", "")
            if best_resp:
                (TASK_DIR / f"response-{rid}.md").write_text(best_resp)

        print(f"\nResults saved to: {TASK_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit Vol Regime Classifier eval task")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    asyncio.run(main(args))
