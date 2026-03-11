# Gemini Intelligence Layer — Evaluation Iterations

Tracking prompt/output iterations for the AI Financial Advisor Engine task submitted through the Gemini Intelligence Layer deployed on GCP Cloud Run.

**Live endpoint**: `https://um-agent-daemon-23o5bq3bfq-uc.a.run.app/api/gemini/`

## Iteration Summary

| # | Method | Key Change | Files | Score (Pro) | Score (Claude) |
|---|--------|-----------|-------|------------|----------------|
| [001](iteration-001.md) | `/enhance` | Baseline — no eval_context, Flash eval | 6/10 (truncated) | N/A (0.70 default) | ~0.35 est |
| [002](iteration-002.md) | `/enhance` | Added eval_context with API signatures | 4/10 (truncated) | N/A (eval parser broken) | N/A |
| [003](iteration-003-multiturn.md) | `/sessions` | System prompt + "don't stop" + "must call LLM" | **11/10** | **0.85** | **0.875** |

## Key Lessons

### Why the model stops at ~6K tokens
- `finish_reason: STOP` — the model *chooses* to stop, not hitting any limit
- `maxOutputTokens: 65536` is set but the model produces ~6K tokens and considers itself done
- Fix: explicit system prompt ("never stop mid-file") + task instruction ("do NOT stop until every file is complete")

### Why the evaluator returned 0.0 scores
1. **Iteration 001**: Flash evaluator returned partial/malformed JSON → parser fell back to default 0.70
2. **Iteration 002**: Eval prompt too large → model echoed input instead of scoring → parser failed
3. **Fix**: Clear `=== DELIMITERS ===` in eval prompt, `max_tokens=8192` for eval, truncated JSON repair

### What makes evaluations accurate
- **Flash** is too lenient (gave 10/10 to code with real issues)
- **Pro** is calibrated well (0.85, caught async/sync issues + missing features)
- **Claude** is most thorough (0.875, caught architectural issues like wrong LLM provider)
- **eval_context** is critical — without API signatures, the evaluator can't verify compatibility

## Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `POST /api/gemini/enhance` | Single-shot generation with enhancement pipeline + self-eval |
| `POST /api/gemini/evaluate` | Standalone evaluation of any prompt+response pair |
| `POST /api/gemini/sessions` | Create multi-turn conversation session |
| `POST /api/gemini/sessions/{id}/message` | Send message in session context |

## Evaluation Dimensions

| Dimension | What it measures |
|-----------|-----------------|
| Accuracy | Are API calls, field names, patterns correct vs reference? |
| Completeness | Are all requested files/features present? |
| Clarity | Is the code well-organized and readable? |
| Actionability | Can someone directly use this output in production? |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/submit_umcfo_task.py` | Iteration 001 submission |
| `scripts/submit_umcfo_task_v2.py` | Iteration 002 submission (with eval_context) |
| `scripts/submit_umcfo_multiturn.py` | Iteration 003 multi-turn submission |
| `scripts/eval_iteration_003_v2.py` | Standalone eval via `/evaluate` endpoint |

## Next Steps

- [ ] Improve eval prompt to catch architectural issues (wrong LLM provider, missing __init__.py)
- [ ] Add eval dimension for "integration correctness" (do API calls match signatures exactly?)
- [ ] Test with Pro 3.1 as evaluator (slower but potentially more thorough)
- [ ] Iterate on generation prompt to fix remaining issues (async/sync, OpenAI → project LLM)
- [ ] Add automated regression: re-run all iterations when eval changes, compare scores
