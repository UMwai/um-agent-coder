"""Iteration runner — generate → evaluate → strategize → retry loop.

POST /iterate   — Start an iteration run (background task)
GET  /iterate/{id}  — Get iteration status + results + step history
GET  /iterations    — List iteration runs
DELETE /iterate/{id} — Cancel a running iteration
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

from um_agent_coder.daemon.auth import verify_api_key

from ._evaluator import (
    AccuracyCheck,
    EvalResult,
    PreGenChecklist,
    evaluate_accuracy,
    evaluate_actionability,
    evaluate_clarity,
    evaluate_completeness,
    evaluate_fulfillment,
    evaluate_response,
    generate_pre_gen_checklist,
    score_pre_gen_checklist,
)
from ._file_extractor import extract_files
from ._firestore import (
    list_iterations_from_firestore,
    persist_iteration_to_firestore,
)
from ._mistake_library import (
    build_mistake_preamble,
    get_relevant_mistakes,
    record_failures,
)
from ._notifier import send_iteration_webhook, should_notify
from ._pipeline import enhance_prompt
from ._router import select_model
from ._strategies import build_strategic_retry_prompt, select_strategies
from ._syntax_validator import validate_code_blocks
from .context_extractor import generate_eval_context
from .models import (
    AccuracyCheckInfo,
    ActionabilityCheckInfo,
    BatchIterateItem,
    BatchIterateItemStatus,
    BatchIterateRequest,
    BatchIterateResponse,
    ClarityCheckInfo,
    CompletenessCheckInfo,
    EvalInfo,
    FulfillmentCheckInfo,
    GeminiModelTier,
    IterateRequest,
    IterateResponse,
    IterationStatus,
    IterationStepInfo,
    IterationSummaryResponse,
    PreGenCheckInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Track running iteration tasks for cancellation
_iteration_tasks: dict[str, asyncio.Task] = {}


def _get_db():
    from um_agent_coder.daemon.app import get_db

    return get_db()


def _get_client():
    from um_agent_coder.daemon.app import get_gemini_client

    return get_gemini_client()


def _get_settings():
    from um_agent_coder.daemon.app import get_settings

    return get_settings()


def _resolve_model(model: GeminiModelTier, prompt: str, settings) -> str:
    from .models import get_model_map

    if model == GeminiModelTier.auto:
        return select_model(prompt, threshold=settings.gemini_complexity_threshold)
    return get_model_map()[model.value]


def _eval_result_to_info(
    er: EvalResult,
    pre_gen_checklist: Optional[PreGenChecklist] = None,
) -> EvalInfo:
    pre_gen_checks = []
    if pre_gen_checklist:
        for c in pre_gen_checklist.checks:
            pre_gen_checks.append(
                PreGenCheckInfo(
                    dimension=c.dimension,
                    check=c.check,
                    status="",  # filled per-step, but this shows the original checklist
                    severity=c.severity,
                    detail=c.detail,
                    source=c.source,
                )
            )
    return EvalInfo(
        score=er.score,
        accuracy=er.accuracy,
        completeness=er.completeness,
        clarity=er.clarity,
        actionability=er.actionability,
        fulfillment=er.fulfillment,
        issues=er.issues,
        retry_count=er.retry_count,
        accuracy_checks=[
            AccuracyCheckInfo(
                check=c.check,
                status=c.status,
                severity=c.severity,
                detail=c.detail,
            )
            for c in er.accuracy_checks
        ],
        fulfillment_checks=[
            FulfillmentCheckInfo(
                check=c.check,
                status=c.status,
                severity=c.severity,
                detail=c.detail,
            )
            for c in er.fulfillment_checks
        ],
        completeness_checks=[
            CompletenessCheckInfo(
                check=c.check,
                status=c.status,
                severity=c.severity,
                detail=c.detail,
            )
            for c in er.completeness_checks
        ],
        clarity_checks=[
            ClarityCheckInfo(
                check=c.check,
                status=c.status,
                severity=c.severity,
                detail=c.detail,
            )
            for c in er.clarity_checks
        ],
        actionability_checks=[
            ActionabilityCheckInfo(
                check=c.check,
                status=c.status,
                severity=c.severity,
                detail=c.detail,
            )
            for c in er.actionability_checks
        ],
        pre_gen_checks=pre_gen_checks,
    )


async def _multi_model_evaluate(
    client,
    prompt: str,
    response: str,
    eval_models: List[str],
    eval_context: Optional[str] = None,
) -> EvalResult:
    """Run evaluation across multiple models and take conservative (min) scores.

    For each dimension, takes the minimum score across all models.
    Deduplicates issues across models.
    """
    if len(eval_models) == 1:
        return await evaluate_response(
            client,
            prompt,
            response,
            model=eval_models[0],
            eval_context=eval_context,
        )

    # Run all evals concurrently
    tasks = [
        evaluate_response(
            client,
            prompt,
            response,
            model=model,
            eval_context=eval_context,
        )
        for model in eval_models
    ]
    results: List[EvalResult] = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    valid_results = [r for r in results if isinstance(r, EvalResult)]
    if not valid_results:
        logger.warning("All eval models failed, returning default")
        return EvalResult(score=0.7)

    # Take minimum per dimension (conservative — Flash leniency doesn't override Pro strictness)
    accuracy = min(r.accuracy for r in valid_results)
    completeness = min(r.completeness for r in valid_results)
    clarity = min(r.clarity for r in valid_results)
    actionability = min(r.actionability for r in valid_results)
    overall = (accuracy + completeness + clarity + actionability) / 4

    # Deduplicate issues
    all_issues: list[str] = []
    seen: set[str] = set()
    for r in valid_results:
        for issue in r.issues:
            normalized = issue.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                all_issues.append(issue)

    return EvalResult(
        score=overall,
        accuracy=accuracy,
        completeness=completeness,
        clarity=clarity,
        actionability=actionability,
        issues=all_issues,
    )


async def _persist_to_firestore_if_enabled(iteration_id: str):
    """Read from SQLite and write to Firestore if enabled."""
    settings = _get_settings()
    if not settings.gemini_firestore_enabled:
        return
    db = _get_db()
    try:
        iteration = await db.get_gemini_iteration(iteration_id)
        if not iteration:
            return
        steps = await db.get_gemini_iteration_steps(iteration_id)
        await persist_iteration_to_firestore(
            iteration_id,
            iteration,
            steps,
            collection=settings.gemini_firestore_collection,
        )
    except Exception as e:
        logger.warning("Firestore persistence failed for %s: %s", iteration_id, e)


async def _run_iteration(iteration_id: str, req: IterateRequest):
    """Background task: generate → evaluate → strategize → retry loop."""
    db = _get_db()
    client = _get_client()
    settings = _get_settings()
    overall_start = time.monotonic()

    # Resolve generation model
    gen_model = _resolve_model(req.model, req.prompt, settings)

    # Resolve eval models
    if req.eval_models:
        eval_models = req.eval_models
    else:
        eval_models = [
            m.strip() for m in settings.gemini_iterate_eval_models.split(",") if m.strip()
        ]

    best_response: Optional[str] = None
    best_score = 0.0
    best_iteration = 0
    total_tokens = 0
    current_prompt = req.prompt
    current_system = req.system_prompt or ""
    current_temp = req.temperature

    try:
        # --- Auto eval_context generation from source_files ---
        if settings.gemini_auto_eval_context_enabled and req.source_files and not req.eval_context:
            try:
                limited_files = dict(
                    list(req.source_files.items())[: settings.gemini_auto_eval_context_max_files]
                )
                req.eval_context = generate_eval_context(limited_files)
                logger.info(
                    "Iteration %s: auto-generated eval_context from %d source files",
                    iteration_id,
                    len(limited_files),
                )
            except Exception as e:
                logger.warning("Auto eval_context generation failed: %s", e)

        # --- Mistake library: prepend known mistakes to first prompt ---
        if settings.gemini_mistake_library_enabled:
            try:
                relevant_mistakes = await get_relevant_mistakes(
                    db,
                    req.prompt,
                    top_k=settings.gemini_mistake_library_top_k,
                    min_occurrences=settings.gemini_mistake_library_min_occurrences,
                )
                if relevant_mistakes:
                    preamble = build_mistake_preamble(relevant_mistakes)
                    current_prompt = preamble + "\n" + current_prompt
                    logger.info(
                        "Iteration %s: prepended %d known mistakes to prompt",
                        iteration_id,
                        len(relevant_mistakes),
                    )
            except Exception as e:
                logger.warning("Mistake library retrieval failed: %s", e)

        # --- Pre-generation checklist: fixed checks generated once before loop ---
        pre_gen_checklist: Optional[PreGenChecklist] = None
        if settings.gemini_pregen_checklist_enabled:
            try:
                pre_gen_checklist = await generate_pre_gen_checklist(
                    client,
                    req.prompt,
                    req.eval_context,
                    max_checks=settings.gemini_pregen_checklist_max_checks,
                )
                if pre_gen_checklist.checks:
                    total_tokens += pre_gen_checklist.generation_tokens
                    logger.info(
                        "Iteration %s: generated %d pre-gen checks (%d tokens)",
                        iteration_id,
                        len(pre_gen_checklist.checks),
                        pre_gen_checklist.generation_tokens,
                    )
                else:
                    pre_gen_checklist = None
                    logger.info("Iteration %s: pre-gen checklist empty, skipping", iteration_id)
            except Exception as e:
                logger.warning("Pre-gen checklist generation failed: %s", e)
                pre_gen_checklist = None

        recent_scores: List[float] = []
        next_step_model_override: Optional[str] = (
            None  # one-shot model switch from oscillation jolt
        )

        for step_num in range(1, req.max_iterations + 1):
            step_start = time.monotonic()
            # Apply one-shot model override from previous step's oscillation jolt
            if next_step_model_override:
                gen_model_this_step = next_step_model_override
                next_step_model_override = None
            else:
                gen_model_this_step = gen_model

            # --- Generate ---
            # Optionally inject pre-gen checklist into prompt so model knows grading criteria
            checklist_suffix = ""
            if (
                pre_gen_checklist
                and settings.gemini_pregen_checklist_in_prompt
                and pre_gen_checklist.checks
            ):
                checklist_suffix = "\n\n" + pre_gen_checklist.format_for_prompt()

            if step_num == 1 and req.use_multi_turn:
                # Step 1: multi-turn for completeness
                prompt_to_send = current_prompt
                if req.enable_enhancement and settings.gemini_enhance_enabled:
                    enhanced = enhance_prompt(prompt_to_send, domain_hint=req.domain_hint)
                    prompt_to_send = enhanced.enhanced
                prompt_to_send += checklist_suffix

                # Build multi-turn contents with system prompt injected
                contents = []
                if current_system:
                    contents.append({"role": "user", "parts": [{"text": current_system}]})
                    contents.append(
                        {
                            "role": "model",
                            "parts": [{"text": "Understood. I'll follow these instructions."}],
                        }
                    )
                contents.append({"role": "user", "parts": [{"text": prompt_to_send}]})

                gen_result = await client.generate_multi_turn(
                    contents=contents,
                    model=gen_model_this_step,
                    temperature=max(0.0, min(2.0, current_temp)),
                    max_tokens=req.max_tokens,
                    timeout=300.0,
                )
            else:
                # Steps 2+: single-shot with strategic prompt
                prompt_to_send = current_prompt + checklist_suffix

                gen_result = await client.generate(
                    prompt=prompt_to_send,
                    model=gen_model_this_step,
                    system_prompt=current_system or None,
                    temperature=max(0.0, min(2.0, current_temp)),
                    max_tokens=req.max_tokens,
                    timeout=300.0,
                )

            gen_duration_ms = int((time.monotonic() - step_start) * 1000)
            response_text = gen_result.get("text", "")
            usage = gen_result.get("usage", {})
            gen_tokens = usage.get("total_tokens", 0)
            finish_reason = usage.get("finish_reason", "")
            total_tokens += gen_tokens

            # --- File extraction + syntax validation (before LLM eval) ---
            extraction_result = None
            syntax_checks: List[AccuracyCheck] = []

            if settings.gemini_file_extraction_enabled:
                extraction_result = extract_files(response_text)
                logger.info(
                    "Iteration %s step %d: extracted %d files (%s)",
                    iteration_id,
                    step_num,
                    extraction_result.total_files,
                    ", ".join(extraction_result.languages),
                )

                if settings.gemini_syntax_validation_enabled and extraction_result.files:
                    syntax_result = validate_code_blocks(extraction_result.files)
                    if syntax_result.issues:
                        logger.info(
                            "Iteration %s step %d: %d syntax issues found",
                            iteration_id,
                            step_num,
                            len(syntax_result.issues),
                        )
                        for issue in syntax_result.issues:
                            syntax_checks.append(
                                AccuracyCheck(
                                    check=f"Syntax error in {issue.file_path or issue.language}",
                                    status="fail",
                                    severity="breaking",
                                    detail=(
                                        f"Line {issue.line}, col {issue.column}: {issue.message}"
                                    ),
                                )
                            )

            # --- Evaluate (accuracy-first cascade + fulfillment) ---
            eval_start = time.monotonic()
            accuracy_passed = True

            use_checklist = settings.gemini_checklist_eval_enabled

            # Determine which dimensions have pre-gen checks available
            _has_pregen = {
                "accuracy": bool(pre_gen_checklist and pre_gen_checklist.accuracy_checks),
                "fulfillment": bool(pre_gen_checklist and pre_gen_checklist.fulfillment_checks),
                "completeness": bool(pre_gen_checklist and pre_gen_checklist.completeness_checks),
            }

            if req.eval_context:
                # Phase 1: Accuracy check — pre-gen or legacy
                if _has_pregen["accuracy"]:
                    accuracy_result = await score_pre_gen_checklist(
                        client,
                        req.prompt,
                        response_text,
                        pre_gen_checks=pre_gen_checklist.accuracy_checks,
                        dimension="accuracy",
                        eval_context=req.eval_context,
                    )
                else:
                    accuracy_result = await evaluate_accuracy(
                        client,
                        req.prompt,
                        response_text,
                        eval_context=req.eval_context,
                    )

                # Inject syntax errors as breaking accuracy checks
                if syntax_checks:
                    from ._evaluator import _score_accuracy_checks

                    accuracy_result.accuracy_checks.extend(syntax_checks)
                    accuracy_result.issues.extend(
                        f"[SYNTAX:BREAKING] {c.check}: {c.detail}" for c in syntax_checks
                    )
                    accuracy_result.accuracy = _score_accuracy_checks(
                        accuracy_result.accuracy_checks
                    )
                    accuracy_result.score = accuracy_result.accuracy

                if accuracy_result.accuracy < 0.7:
                    # Accuracy failed — skip full eval, use accuracy result only
                    accuracy_passed = False
                    eval_result = accuracy_result
                    logger.info(
                        "Iteration %s step %d: accuracy=%.3f < 0.7, skipping full eval "
                        "(%d checks, %d failed)",
                        iteration_id,
                        step_num,
                        accuracy_result.accuracy,
                        len(accuracy_result.accuracy_checks),
                        sum(1 for c in accuracy_result.accuracy_checks if c.status == "fail"),
                    )
                elif use_checklist:
                    # Accuracy passed — run 4 parallel checklist evals
                    # Use pre-gen scoring for dimensions with pre-gen checks, else post-hoc
                    if _has_pregen["completeness"]:
                        comp_task = score_pre_gen_checklist(
                            client,
                            req.prompt,
                            response_text,
                            pre_gen_checks=pre_gen_checklist.completeness_checks,
                            dimension="completeness",
                        )
                    else:
                        comp_task = evaluate_completeness(client, req.prompt, response_text)

                    # Clarity: always post-hoc
                    clar_task = evaluate_clarity(client, req.prompt, response_text)
                    # Actionability: always post-hoc
                    act_task = evaluate_actionability(client, req.prompt, response_text)

                    if _has_pregen["fulfillment"]:
                        ful_task = score_pre_gen_checklist(
                            client,
                            req.prompt,
                            response_text,
                            pre_gen_checks=pre_gen_checklist.fulfillment_checks,
                            dimension="fulfillment",
                        )
                    else:
                        ful_task = evaluate_fulfillment(client, req.prompt, response_text)

                    comp_r, clar_r, act_r, ful_r = await asyncio.gather(
                        comp_task,
                        clar_task,
                        act_task,
                        ful_task,
                    )

                    # Only include successfully parsed dimensions in the average
                    # Parse-failed evals default to 0.5 which would unfairly drag score down
                    dims = [accuracy_result.accuracy]
                    parse_failures = []
                    for label, result_obj, dim_score in [
                        ("completeness", comp_r, comp_r.completeness),
                        ("clarity", clar_r, clar_r.clarity),
                        ("actionability", act_r, act_r.actionability),
                        ("fulfillment", ful_r, ful_r.fulfillment),
                    ]:
                        if result_obj.parse_failed:
                            parse_failures.append(label)
                        else:
                            dims.append(dim_score)

                    if parse_failures:
                        logger.warning(
                            "Iteration %s step %d: %d eval parse failures: %s",
                            iteration_id,
                            step_num,
                            len(parse_failures),
                            ", ".join(parse_failures),
                        )

                    eval_result = EvalResult(
                        score=sum(dims) / len(dims) if dims else 0.5,
                        accuracy=accuracy_result.accuracy,
                        completeness=comp_r.completeness,
                        clarity=clar_r.clarity,
                        actionability=act_r.actionability,
                        fulfillment=ful_r.fulfillment,
                        issues=(
                            accuracy_result.issues
                            + comp_r.issues
                            + clar_r.issues
                            + act_r.issues
                            + ful_r.issues
                        ),
                        accuracy_checks=accuracy_result.accuracy_checks,
                        fulfillment_checks=ful_r.fulfillment_checks,
                        completeness_checks=comp_r.completeness_checks,
                        clarity_checks=clar_r.clarity_checks,
                        actionability_checks=act_r.actionability_checks,
                        parse_failed=bool(parse_failures),
                        parse_failed_dimensions=parse_failures,
                    )
                else:
                    # Legacy: _multi_model_evaluate + fulfillment
                    full_task = _multi_model_evaluate(
                        client,
                        req.prompt,
                        response_text,
                        eval_models=eval_models,
                        eval_context=req.eval_context,
                    )
                    fulfill_task = evaluate_fulfillment(
                        client,
                        req.prompt,
                        response_text,
                    )
                    full_result, fulfill_result = await asyncio.gather(
                        full_task,
                        fulfill_task,
                    )

                    dims = [
                        accuracy_result.accuracy,
                        full_result.completeness,
                        full_result.clarity,
                        full_result.actionability,
                        fulfill_result.fulfillment,
                    ]
                    eval_result = EvalResult(
                        score=sum(dims) / len(dims),
                        accuracy=accuracy_result.accuracy,
                        completeness=full_result.completeness,
                        clarity=full_result.clarity,
                        actionability=full_result.actionability,
                        fulfillment=fulfill_result.fulfillment,
                        issues=(
                            accuracy_result.issues + full_result.issues + fulfill_result.issues
                        ),
                        accuracy_checks=accuracy_result.accuracy_checks,
                        fulfillment_checks=fulfill_result.fulfillment_checks,
                    )
            else:
                # No eval_context — run checklist evals or legacy
                if use_checklist:
                    if _has_pregen["completeness"]:
                        comp_task = score_pre_gen_checklist(
                            client,
                            req.prompt,
                            response_text,
                            pre_gen_checks=pre_gen_checklist.completeness_checks,
                            dimension="completeness",
                        )
                    else:
                        comp_task = evaluate_completeness(client, req.prompt, response_text)
                    clar_task = evaluate_clarity(client, req.prompt, response_text)
                    act_task = evaluate_actionability(client, req.prompt, response_text)
                    if _has_pregen["fulfillment"]:
                        ful_task = score_pre_gen_checklist(
                            client,
                            req.prompt,
                            response_text,
                            pre_gen_checks=pre_gen_checklist.fulfillment_checks,
                            dimension="fulfillment",
                        )
                    else:
                        ful_task = evaluate_fulfillment(client, req.prompt, response_text)

                    comp_r, clar_r, act_r, ful_r = await asyncio.gather(
                        comp_task,
                        clar_task,
                        act_task,
                        ful_task,
                    )

                    # No accuracy checklist without eval_context — use completeness as proxy
                    dims = [
                        comp_r.completeness,
                        clar_r.clarity,
                        act_r.actionability,
                        ful_r.fulfillment,
                    ]
                    eval_result = EvalResult(
                        score=sum(dims) / len(dims),
                        completeness=comp_r.completeness,
                        clarity=clar_r.clarity,
                        actionability=act_r.actionability,
                        fulfillment=ful_r.fulfillment,
                        issues=(comp_r.issues + clar_r.issues + act_r.issues + ful_r.issues),
                        fulfillment_checks=ful_r.fulfillment_checks,
                        completeness_checks=comp_r.completeness_checks,
                        clarity_checks=clar_r.clarity_checks,
                        actionability_checks=act_r.actionability_checks,
                    )
                else:
                    full_task = _multi_model_evaluate(
                        client,
                        req.prompt,
                        response_text,
                        eval_models=eval_models,
                        eval_context=None,
                    )
                    fulfill_task = evaluate_fulfillment(
                        client,
                        req.prompt,
                        response_text,
                    )
                    full_result, fulfill_result = await asyncio.gather(
                        full_task,
                        fulfill_task,
                    )

                    dims = [
                        full_result.accuracy,
                        full_result.completeness,
                        full_result.clarity,
                        full_result.actionability,
                        fulfill_result.fulfillment,
                    ]
                    eval_result = EvalResult(
                        score=sum(dims) / len(dims),
                        accuracy=full_result.accuracy,
                        completeness=full_result.completeness,
                        clarity=full_result.clarity,
                        actionability=full_result.actionability,
                        fulfillment=fulfill_result.fulfillment,
                        issues=full_result.issues + fulfill_result.issues,
                        fulfillment_checks=fulfill_result.fulfillment_checks,
                    )

            eval_duration_ms = int((time.monotonic() - eval_start) * 1000)

            # Track scores for oscillation detection
            recent_scores.append(eval_result.score)

            # Track best
            if eval_result.score > best_score:
                best_score = eval_result.score
                best_response = response_text
                best_iteration = step_num

            # --- Record mistakes from failing checks ---
            if settings.gemini_mistake_library_enabled:
                try:
                    all_failing_checks = (
                        eval_result.accuracy_checks
                        + eval_result.fulfillment_checks
                        + eval_result.completeness_checks
                        + eval_result.clarity_checks
                        + eval_result.actionability_checks
                    )
                    if all_failing_checks:
                        recorded = await record_failures(
                            db,
                            all_failing_checks,
                            "mixed",
                        )
                        if recorded:
                            logger.info(
                                "Iteration %s step %d: recorded %d mistake patterns",
                                iteration_id,
                                step_num,
                                recorded,
                            )
                except Exception as e:
                    logger.warning("Mistake recording failed: %s", e)

            # --- Select strategies ---
            # Use a dynamic threshold: at least 0.7, but raise to (score_threshold - 0.1)
            # so dimensions between 0.7-0.85 get fix strategies when targeting 0.95
            strategy_threshold = max(0.7, req.score_threshold - 0.1)
            strategies = select_strategies(
                eval_result,
                eval_context=req.eval_context,
                threshold=strategy_threshold,
                max_strategies=settings.gemini_iterate_max_strategies,
            )
            strategy_names = [s.name for s in strategies]

            # --- Oscillation detection ---
            oscillation_detected = False
            osc_window = settings.gemini_iterate_oscillation_window
            if len(recent_scores) >= osc_window:
                last_n = recent_scores[-osc_window:]
                spread = max(last_n) - min(last_n)
                if spread <= settings.gemini_iterate_oscillation_spread:
                    oscillation_detected = True
                    logger.info(
                        "Iteration %s step %d: oscillation detected "
                        "(last %d scores: %s, spread=%.3f)",
                        iteration_id,
                        step_num,
                        osc_window,
                        [f"{s:.3f}" for s in last_n],
                        spread,
                    )

            if oscillation_detected:
                # Temperature jolt — force a different sampling regime
                if current_temp > 0.5:
                    current_temp = 0.2
                else:
                    current_temp = 0.9

                # Model switch for next step (one-shot)
                alt_model = settings.gemini_model_pro
                if gen_model != alt_model:
                    next_step_model_override = alt_model
                    logger.info(
                        "Oscillation jolt: next step will use %s, temp=%.1f",
                        alt_model,
                        current_temp,
                    )

            # --- Store step ---
            eval_scores = {
                "score": eval_result.score,
                "accuracy": eval_result.accuracy,
                "completeness": eval_result.completeness,
                "clarity": eval_result.clarity,
                "actionability": eval_result.actionability,
                "fulfillment": eval_result.fulfillment,
                "issues": eval_result.issues,
                "accuracy_checks": [
                    {
                        "check": c.check,
                        "status": c.status,
                        "severity": c.severity,
                        "detail": c.detail,
                    }
                    for c in eval_result.accuracy_checks
                ],
                "fulfillment_checks": [
                    {
                        "check": c.check,
                        "status": c.status,
                        "severity": c.severity,
                        "detail": c.detail,
                    }
                    for c in eval_result.fulfillment_checks
                ],
                "completeness_checks": [
                    {
                        "check": c.check,
                        "status": c.status,
                        "severity": c.severity,
                        "detail": c.detail,
                    }
                    for c in eval_result.completeness_checks
                ],
                "clarity_checks": [
                    {
                        "check": c.check,
                        "status": c.status,
                        "severity": c.severity,
                        "detail": c.detail,
                    }
                    for c in eval_result.clarity_checks
                ],
                "actionability_checks": [
                    {
                        "check": c.check,
                        "status": c.status,
                        "severity": c.severity,
                        "detail": c.detail,
                    }
                    for c in eval_result.actionability_checks
                ],
                "pre_gen_checks": [
                    {
                        "dimension": c.dimension,
                        "check": c.check,
                        "severity": c.severity,
                        "source": c.source,
                    }
                    for c in (pre_gen_checklist.checks if pre_gen_checklist else [])
                ],
                "accuracy_passed": accuracy_passed,
                "parse_failed": eval_result.parse_failed,
                "extraction": (
                    {
                        "total_files": extraction_result.total_files if extraction_result else 0,
                        "languages": extraction_result.languages if extraction_result else [],
                        "truncated_files": (
                            extraction_result.truncated_files if extraction_result else 0
                        ),
                        "syntax_issues": len(syntax_checks),
                    }
                    if extraction_result
                    else None
                ),
            }
            await db.add_gemini_iteration_step(
                iteration_id=iteration_id,
                step_number=step_num,
                prompt_sent=prompt_to_send[:50000],  # truncate for DB
                response=response_text,
                generation_model=gen_model_this_step,
                generation_duration_ms=gen_duration_ms,
                generation_tokens=gen_tokens,
                eval_scores=eval_scores,
                eval_models=eval_models,
                eval_duration_ms=eval_duration_ms,
                strategies_applied=strategy_names,
                finish_reason=finish_reason,
            )

            # Update iteration progress
            int((time.monotonic() - overall_start) * 1000)
            await db.update_gemini_iteration(
                iteration_id,
                best_response=best_response,
                best_score=best_score,
                best_iteration=best_iteration,
                total_iterations=step_num,
                total_tokens=total_tokens,
            )

            logger.info(
                "Iteration %s step %d: score=%.3f (best=%.3f@%d) strategies=%s",
                iteration_id,
                step_num,
                eval_result.score,
                best_score,
                best_iteration,
                strategy_names,
            )

            # --- Per-step webhook ---
            if req.webhook_url and should_notify("step_complete", req.webhook_events):
                asyncio.create_task(
                    send_iteration_webhook(
                        iteration_id,
                        "step_complete",
                        req.webhook_url,
                        webhook_headers=req.webhook_headers,
                        payload={
                            "step": step_num,
                            "score": eval_result.score,
                            "best_score": best_score,
                            "strategies": strategy_names,
                            "model": gen_model_this_step,
                            "gen_duration_ms": gen_duration_ms,
                            "eval_duration_ms": eval_duration_ms,
                            "oscillation_detected": oscillation_detected,
                        },
                        timeout_seconds=settings.gemini_webhook_timeout_seconds,
                        max_retries=settings.gemini_webhook_max_retries,
                    )
                )

            # --- Check threshold ---
            if eval_result.score >= req.score_threshold:
                now = datetime.now(timezone.utc).isoformat()
                int((time.monotonic() - overall_start) * 1000)
                await db.update_gemini_iteration(
                    iteration_id,
                    status="threshold_met",
                    best_response=best_response,
                    best_score=best_score,
                    best_iteration=best_iteration,
                    total_iterations=step_num,
                    total_tokens=total_tokens,
                    completed_at=now,
                )
                logger.info(
                    "Iteration %s threshold met at step %d (%.3f >= %.3f)",
                    iteration_id,
                    step_num,
                    eval_result.score,
                    req.score_threshold,
                )
                await _persist_to_firestore_if_enabled(iteration_id)
                if req.webhook_url and should_notify("threshold_met", req.webhook_events):
                    asyncio.create_task(
                        send_iteration_webhook(
                            iteration_id,
                            "threshold_met",
                            req.webhook_url,
                            webhook_headers=req.webhook_headers,
                            payload={"best_score": best_score, "total_iterations": step_num},
                            timeout_seconds=settings.gemini_webhook_timeout_seconds,
                            max_retries=settings.gemini_webhook_max_retries,
                        )
                    )
                return

            # --- Build strategic retry prompt for next step ---
            # When accuracy cascade fails, only apply accuracy strategy
            # (other dimensions were never evaluated, their 0.0 is not a real score)
            if not accuracy_passed:
                strategies = [s for s in strategies if s.dimension == "accuracy"]

            # Use best response for retry context if current step regressed
            anchor_response = response_text
            if best_response and eval_result.score < best_score:
                anchor_response = best_response
                logger.info(
                    "Iteration %s step %d: score %.3f regressed from best %.3f, "
                    "anchoring retry to step %d response",
                    iteration_id,
                    step_num,
                    eval_result.score,
                    best_score,
                    best_iteration,
                )

            if strategies:
                retry_prompt, system_addendum, temp_delta = build_strategic_retry_prompt(
                    original_prompt=req.prompt,
                    previous_response=anchor_response,
                    eval_result=eval_result,
                    strategies=strategies,
                    eval_context=req.eval_context,
                )
                current_prompt = retry_prompt
                # Reset system prompt each step — don't accumulate conflicting instructions
                base_system = req.system_prompt or ""
                current_system = (
                    (base_system + " " + system_addendum).strip()
                    if system_addendum
                    else base_system
                )
                if not oscillation_detected:
                    current_temp = max(0.0, min(2.0, req.temperature + temp_delta))
                # else: keep oscillation-jolted temperature
            else:
                # No specific strategy — use generic retry
                current_prompt = (
                    f"{req.prompt}\n\n"
                    f"[Previous attempt scored {eval_result.score:.2f}. "
                    f"Please improve the response.]"
                )

        # Max iterations reached — finalize with best
        now = datetime.now(timezone.utc).isoformat()
        int((time.monotonic() - overall_start) * 1000)
        await db.update_gemini_iteration(
            iteration_id,
            status="max_iterations_reached",
            best_response=best_response,
            best_score=best_score,
            best_iteration=best_iteration,
            total_iterations=req.max_iterations,
            total_tokens=total_tokens,
            completed_at=now,
        )
        logger.info(
            "Iteration %s max iterations reached. Best score=%.3f at step %d",
            iteration_id,
            best_score,
            best_iteration,
        )
        await _persist_to_firestore_if_enabled(iteration_id)
        if req.webhook_url and should_notify("max_iterations_reached", req.webhook_events):
            asyncio.create_task(
                send_iteration_webhook(
                    iteration_id,
                    "max_iterations_reached",
                    req.webhook_url,
                    webhook_headers=req.webhook_headers,
                    payload={"best_score": best_score, "total_iterations": req.max_iterations},
                    timeout_seconds=settings.gemini_webhook_timeout_seconds,
                    max_retries=settings.gemini_webhook_max_retries,
                )
            )

    except asyncio.CancelledError:
        now = datetime.now(timezone.utc).isoformat()
        await db.update_gemini_iteration(
            iteration_id,
            status="cancelled",
            best_response=best_response,
            best_score=best_score,
            best_iteration=best_iteration,
            total_tokens=total_tokens,
            completed_at=now,
        )
        logger.info("Iteration %s cancelled", iteration_id)
        await _persist_to_firestore_if_enabled(iteration_id)
        if req.webhook_url and should_notify("cancelled", req.webhook_events):
            asyncio.create_task(
                send_iteration_webhook(
                    iteration_id,
                    "cancelled",
                    req.webhook_url,
                    webhook_headers=req.webhook_headers,
                    timeout_seconds=settings.gemini_webhook_timeout_seconds,
                    max_retries=settings.gemini_webhook_max_retries,
                )
            )
    except Exception as e:
        logger.exception("Iteration %s failed: %s", iteration_id, e)
        now = datetime.now(timezone.utc).isoformat()
        await db.update_gemini_iteration(
            iteration_id,
            status="failed",
            error=str(e),
            best_response=best_response,
            best_score=best_score,
            best_iteration=best_iteration,
            total_tokens=total_tokens,
            completed_at=now,
        )
        await _persist_to_firestore_if_enabled(iteration_id)
        if req.webhook_url and should_notify("failed", req.webhook_events):
            asyncio.create_task(
                send_iteration_webhook(
                    iteration_id,
                    "failed",
                    req.webhook_url,
                    webhook_headers=req.webhook_headers,
                    payload={"error": str(e)},
                    timeout_seconds=settings.gemini_webhook_timeout_seconds,
                    max_retries=settings.gemini_webhook_max_retries,
                )
            )


# --- Helper: build response from DB ---


async def _build_iterate_response(iteration_id: str) -> IterateResponse:
    db = _get_db()
    iteration = await db.get_gemini_iteration(iteration_id)
    if not iteration:
        raise HTTPException(status_code=404, detail="Iteration not found")

    steps_rows = await db.get_gemini_iteration_steps(iteration_id)
    steps = []
    for s in steps_rows:
        eval_scores = s.get("eval_scores") or {}
        evaluation = (
            EvalInfo(
                score=eval_scores.get("score", 0.0),
                accuracy=eval_scores.get("accuracy", 0.0),
                completeness=eval_scores.get("completeness", 0.0),
                clarity=eval_scores.get("clarity", 0.0),
                actionability=eval_scores.get("actionability", 0.0),
                fulfillment=eval_scores.get("fulfillment", 0.0),
                issues=eval_scores.get("issues", []),
                accuracy_passed=eval_scores.get("accuracy_passed"),
                parse_failed=eval_scores.get("parse_failed"),
                accuracy_checks=[
                    AccuracyCheckInfo(**c) for c in eval_scores.get("accuracy_checks", [])
                ],
                fulfillment_checks=[
                    FulfillmentCheckInfo(**c) for c in eval_scores.get("fulfillment_checks", [])
                ],
                completeness_checks=[
                    CompletenessCheckInfo(**c) for c in eval_scores.get("completeness_checks", [])
                ],
                clarity_checks=[
                    ClarityCheckInfo(**c) for c in eval_scores.get("clarity_checks", [])
                ],
                actionability_checks=[
                    ActionabilityCheckInfo(**c) for c in eval_scores.get("actionability_checks", [])
                ],
                pre_gen_checks=[
                    PreGenCheckInfo(**c) for c in eval_scores.get("pre_gen_checks", [])
                ],
            )
            if eval_scores
            else None
        )

        steps.append(
            IterationStepInfo(
                step=s["step_number"],
                prompt_sent=s.get("prompt_sent") or "",
                response=s.get("response") or "",
                generation_model=s.get("generation_model") or "",
                generation_duration_ms=s.get("generation_duration_ms", 0),
                generation_tokens=s.get("generation_tokens", 0),
                evaluation=evaluation,
                eval_models_used=s.get("eval_models") or [],
                strategies_applied=s.get("strategies_applied") or [],
                finish_reason=s.get("finish_reason") or "",
            )
        )

    # Calculate duration
    duration_ms = 0
    if iteration.get("started_at") and iteration.get("completed_at"):
        try:
            started = datetime.fromisoformat(iteration["started_at"])
            completed = datetime.fromisoformat(iteration["completed_at"])
            duration_ms = int((completed - started).total_seconds() * 1000)
        except (ValueError, TypeError):
            pass

    return IterateResponse(
        id=iteration["id"],
        status=iteration["status"],
        original_prompt=iteration["original_prompt"],
        best_response=iteration.get("best_response"),
        best_score=iteration.get("best_score", 0.0),
        best_iteration=iteration.get("best_iteration", 0),
        total_iterations=iteration.get("total_iterations", 0),
        total_tokens=iteration.get("total_tokens", 0),
        duration_ms=duration_ms,
        config=iteration.get("config"),
        steps=steps,
        error=iteration.get("error"),
        created_at=iteration.get("created_at"),
    )


# --- Endpoints ---


@router.post("/iterate", response_model=IterateResponse)
async def start_iteration(
    req: IterateRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Start an iteration loop: generate → evaluate → strategize → retry."""
    db = _get_db()
    iteration_id = f"gi-{uuid.uuid4().hex[:12]}"

    config = {
        "model": req.model.value,
        "eval_models": req.eval_models,
        "max_iterations": req.max_iterations,
        "score_threshold": req.score_threshold,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "enable_enhancement": req.enable_enhancement,
        "use_multi_turn": req.use_multi_turn,
        "domain_hint": req.domain_hint,
    }

    await db.create_gemini_iteration(
        iteration_id=iteration_id,
        original_prompt=req.prompt,
        system_prompt=req.system_prompt,
        eval_context=req.eval_context,
        config=config,
    )

    # Launch background task
    task = asyncio.create_task(_run_iteration(iteration_id, req))
    _iteration_tasks[iteration_id] = task

    def _cleanup(t):
        _iteration_tasks.pop(iteration_id, None)

    task.add_done_callback(_cleanup)

    return await _build_iterate_response(iteration_id)


# --- Batch iteration endpoints (MUST be before /iterate/{iteration_id} to avoid capture) ---

_batch_tasks: dict[str, asyncio.Task] = {}


async def _run_batch_iterations(batch_id: str, req: BatchIterateRequest):
    """Background task: run multiple iteration items concurrently."""
    db = _get_db()
    settings = _get_settings()
    max_concurrent = req.max_concurrent or settings.gemini_batch_max_concurrent
    semaphore = asyncio.Semaphore(max_concurrent)

    now = datetime.now(timezone.utc).isoformat()
    await db.update_gemini_batch(batch_id, status="running", started_at=now)

    iteration_ids: dict[int, str] = {}  # index → iteration_id
    completed = 0
    failed = 0

    async def _run_item(index: int, item: BatchIterateItem):
        nonlocal completed, failed
        async with semaphore:
            iteration_id = f"gi-{uuid.uuid4().hex[:12]}"
            iteration_ids[index] = iteration_id

            # Build per-item IterateRequest using shared defaults + per-item overrides
            item_req = IterateRequest(
                prompt=item.prompt,
                system_prompt=item.system_prompt,
                eval_context=item.eval_context,
                model=item.model or req.model,
                eval_models=req.eval_models,
                max_iterations=req.max_iterations,
                score_threshold=req.score_threshold,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                enable_enhancement=req.enable_enhancement,
                use_multi_turn=req.use_multi_turn,
                domain_hint=req.domain_hint,
                webhook_url=req.webhook_url,
                webhook_headers=req.webhook_headers,
                webhook_events=req.webhook_events,
            )

            config = {
                "model": item_req.model.value,
                "eval_models": item_req.eval_models,
                "max_iterations": item_req.max_iterations,
                "score_threshold": item_req.score_threshold,
                "temperature": item_req.temperature,
                "max_tokens": item_req.max_tokens,
                "batch_id": batch_id,
                "batch_index": index,
                "label": item.label,
            }

            await db.create_gemini_iteration(
                iteration_id=iteration_id,
                original_prompt=item.prompt,
                system_prompt=item.system_prompt,
                eval_context=item.eval_context,
                config=config,
            )

            try:
                await _run_iteration(iteration_id, item_req)
                completed += 1
            except Exception as e:
                logger.exception("Batch %s item %d failed: %s", batch_id, index, e)
                failed += 1

            # Update batch progress
            await db.update_gemini_batch(
                batch_id,
                completed_queries=completed,
                failed_queries=failed,
                results={"iteration_ids": {str(k): v for k, v in iteration_ids.items()}},
            )

    try:
        tasks = [_run_item(i, item) for i, item in enumerate(req.items)]
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        logger.info("Batch %s cancelled", batch_id)
    finally:
        final_status = "completed" if failed == 0 else "failed"
        now = datetime.now(timezone.utc).isoformat()
        await db.update_gemini_batch(
            batch_id,
            status=final_status,
            completed_queries=completed,
            failed_queries=failed,
            completed_at=now,
            results={"iteration_ids": {str(k): v for k, v in iteration_ids.items()}},
        )


async def _build_batch_response(batch_id: str) -> BatchIterateResponse:
    """Build batch response from DB."""
    db = _get_db()
    batch = await db.get_gemini_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    results = batch.get("results") or {}
    iteration_ids = results.get("iteration_ids") or {}

    items: List[BatchIterateItemStatus] = []
    for idx_str, iter_id in sorted(iteration_ids.items(), key=lambda x: int(x[0])):
        iteration = await db.get_gemini_iteration(iter_id)
        if iteration:
            items.append(
                BatchIterateItemStatus(
                    index=int(idx_str),
                    label=(iteration.get("config") or {}).get("label"),
                    iteration_id=iter_id,
                    status=iteration["status"],
                    best_score=iteration.get("best_score", 0.0),
                    total_iterations=iteration.get("total_iterations", 0),
                    error=iteration.get("error"),
                )
            )
        else:
            items.append(
                BatchIterateItemStatus(
                    index=int(idx_str),
                    iteration_id=iter_id,
                    status=IterationStatus.failed,
                    error="Iteration not found",
                )
            )

    return BatchIterateResponse(
        batch_id=batch["id"],
        status=batch["status"],
        total_items=batch.get("total_queries", 0),
        completed_items=batch.get("completed_queries", 0),
        failed_items=batch.get("failed_queries", 0),
        items=items,
        created_at=batch.get("created_at"),
        completed_at=batch.get("completed_at"),
    )


@router.post("/iterate/batch", response_model=BatchIterateResponse)
async def start_batch_iteration(
    req: BatchIterateRequest,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Start a batch of iteration runs concurrently."""
    db = _get_db()
    batch_id = f"gb-{uuid.uuid4().hex[:12]}"

    config = {
        "model": req.model.value,
        "max_iterations": req.max_iterations,
        "score_threshold": req.score_threshold,
        "total_items": len(req.items),
    }

    await db.create_gemini_batch(
        batch_id=batch_id,
        total_queries=len(req.items),
        model=req.model.value,
        config=config,
    )

    task = asyncio.create_task(_run_batch_iterations(batch_id, req))
    _batch_tasks[batch_id] = task

    def _cleanup(t):
        _batch_tasks.pop(batch_id, None)

    task.add_done_callback(_cleanup)

    return await _build_batch_response(batch_id)


@router.get("/iterate/batch/{batch_id}", response_model=BatchIterateResponse)
async def get_batch_iteration(
    batch_id: str,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Get batch iteration status and per-item details."""
    return await _build_batch_response(batch_id)


@router.delete("/iterate/batch/{batch_id}")
async def cancel_batch_iteration(
    batch_id: str,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Cancel all iterations in a batch."""
    db = _get_db()
    batch = await db.get_gemini_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    if batch["status"] not in ("pending", "running"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel batch in '{batch['status']}' state",
        )

    # Cancel the background task (which will cancel all sub-iterations)
    task = _batch_tasks.get(batch_id)
    if task and not task.done():
        task.cancel()

    # Also cancel any individual iteration tasks
    results = batch.get("results") or {}
    for iter_id in (results.get("iteration_ids") or {}).values():
        iter_task = _iteration_tasks.get(iter_id)
        if iter_task and not iter_task.done():
            iter_task.cancel()

    now = datetime.now(timezone.utc).isoformat()
    await db.update_gemini_batch(batch_id, status="cancelled", completed_at=now)
    return {"status": "cancelled", "batch_id": batch_id}


# --- Firestore history endpoint ---


@router.get("/iterations/history")
async def list_iteration_history(
    limit: int = 50,
    status: Optional[str] = None,
    _key: Optional[str] = Depends(verify_api_key),
):
    """List iteration history from Firestore (persists across restarts)."""
    settings = _get_settings()
    if not settings.gemini_firestore_enabled:
        raise HTTPException(status_code=400, detail="Firestore persistence is not enabled")

    results = await list_iterations_from_firestore(
        collection=settings.gemini_firestore_collection,
        limit=limit,
        status=status,
    )
    return results


@router.get("/iterate/{iteration_id}", response_model=IterateResponse)
async def get_iteration(
    iteration_id: str,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Get iteration status, results, and step history."""
    return await _build_iterate_response(iteration_id)


@router.get("/iterations", response_model=List[IterationSummaryResponse])
async def list_iterations(
    limit: int = 50,
    status: Optional[str] = None,
    _key: Optional[str] = Depends(verify_api_key),
):
    """List iteration runs."""
    db = _get_db()
    iterations = await db.list_gemini_iterations(limit=limit, status=status)

    result = []
    for it in iterations:
        duration_ms = 0
        if it.get("started_at") and it.get("completed_at"):
            try:
                started = datetime.fromisoformat(it["started_at"])
                completed = datetime.fromisoformat(it["completed_at"])
                duration_ms = int((completed - started).total_seconds() * 1000)
            except (ValueError, TypeError):
                pass

        result.append(
            IterationSummaryResponse(
                id=it["id"],
                status=it["status"],
                best_score=it.get("best_score", 0.0),
                total_iterations=it.get("total_iterations", 0),
                total_tokens=it.get("total_tokens", 0),
                duration_ms=duration_ms,
                created_at=it.get("created_at"),
            )
        )
    return result


@router.delete("/iterate/{iteration_id}")
async def cancel_iteration(
    iteration_id: str,
    _key: Optional[str] = Depends(verify_api_key),
):
    """Cancel a running iteration."""
    db = _get_db()
    iteration = await db.get_gemini_iteration(iteration_id)
    if not iteration:
        raise HTTPException(status_code=404, detail="Iteration not found")

    if iteration["status"] != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel iteration in '{iteration['status']}' state",
        )

    # Cancel the background task
    task = _iteration_tasks.get(iteration_id)
    if task and not task.done():
        task.cancel()

    now = datetime.now(timezone.utc).isoformat()
    await db.update_gemini_iteration(iteration_id, status="cancelled", completed_at=now)
    return {"status": "cancelled", "iteration_id": iteration_id}
