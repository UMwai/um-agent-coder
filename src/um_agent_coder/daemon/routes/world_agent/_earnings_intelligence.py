"""Earnings Intelligence — pre/post earnings analysis via Gemini.

Pre-earnings: analyze historical surprise patterns, IV crush risk, and
generate hold/exit/reduce recommendations for held positions approaching
earnings.

Post-earnings: score the actual beat/miss and recommend position adjustments.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PRE_EARNINGS_PROMPT = """\
You are a quantitative earnings analyst for an AI hedge fund.

Analyze the following historical earnings data for {symbol} and produce a
pre-earnings risk assessment.

## Historical Earnings Data
{earnings_data}

## Current Position Context
The fund holds {symbol}. Earnings are on {earnings_date}.

## Task
1. Calculate the average earnings surprise % (last 4-8 quarters).
2. Identify the typical post-earnings price reaction pattern.
3. Assess IV crush risk — does the stock consistently gap or fade after earnings?
4. Evaluate revenue trend (growing, flat, declining).
5. Check if guidance patterns suggest beat-and-raise or sandbagging.

Return a JSON object:
{{
  "symbol": "{symbol}",
  "earnings_date": "{earnings_date}",
  "historical_surprise_avg": <float, avg surprise %>,
  "surprise_consistency": <float, 0-1 how consistent the surprises are>,
  "typical_reaction_pct": <float, avg post-earnings move %>,
  "revenue_trend": "growing | flat | declining",
  "guidance_pattern": "beat_and_raise | sandbagging | mixed | no_pattern",
  "recommendation": "HOLD | EXIT | REDUCE",
  "iv_crush_warning": <bool>,
  "reasoning": "<detailed reasoning with data points>"
}}
"""

POST_EARNINGS_PROMPT = """\
You are a quantitative earnings analyst for an AI hedge fund.

Analyze this earnings result for {symbol}:
- Actual EPS: ${actual_eps}
- Estimated EPS: ${est_eps}
- Surprise: {surprise_pct:.1f}%

## Historical Context
{earnings_data}

## Task
1. Score the beat/miss magnitude relative to historical surprises.
2. Predict the likely price reaction based on historical patterns.
3. Recommend position adjustment.

Return a JSON object:
{{
  "symbol": "{symbol}",
  "actual_eps": {actual_eps},
  "estimated_eps": {est_eps},
  "surprise_pct": {surprise_pct:.1f},
  "beat_magnitude": "large_beat | small_beat | inline | small_miss | large_miss",
  "reaction_score": <float, -1.0 (very bearish) to 1.0 (very bullish)>,
  "expected_move_pct": <float, expected next-day move %>,
  "recommendation": "ADD | HOLD | REDUCE | EXIT",
  "reasoning": "<detailed reasoning>"
}}
"""


def _fetch_earnings_history(symbol: str) -> Dict[str, Any]:
    """Fetch historical earnings data via yfinance (synchronous)."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        result: Dict[str, Any] = {"symbol": symbol}

        # Earnings dates and history
        try:
            earnings_dates = ticker.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                rows = []
                for idx, row in earnings_dates.head(8).iterrows():
                    rows.append({
                        "date": str(idx),
                        "eps_estimate": float(row.get("EPS Estimate", 0) or 0),
                        "reported_eps": float(row.get("Reported EPS", 0) or 0),
                        "surprise_pct": float(row.get("Surprise(%)", 0) or 0),
                    })
                result["earnings_history"] = rows

                # Calculate stats
                surprises = [r["surprise_pct"] for r in rows if r["surprise_pct"] != 0]
                if surprises:
                    result["avg_surprise_pct"] = sum(surprises) / len(surprises)
                    result["surprise_std"] = (
                        sum((s - result["avg_surprise_pct"]) ** 2 for s in surprises)
                        / len(surprises)
                    ) ** 0.5
        except Exception:
            result["earnings_history"] = []

        # Revenue trend from financials
        try:
            financials = ticker.quarterly_financials
            if financials is not None and not financials.empty:
                if "Total Revenue" in financials.index:
                    revenues = financials.loc["Total Revenue"].dropna().head(4).tolist()
                    if len(revenues) >= 2:
                        result["recent_revenues"] = [float(r) for r in revenues]
                        result["revenue_trend"] = (
                            "growing" if revenues[0] > revenues[-1]
                            else "declining" if revenues[0] < revenues[-1]
                            else "flat"
                        )
        except Exception:
            pass

        # Next earnings date
        try:
            cal = ticker.calendar
            if cal and isinstance(cal, dict):
                ed = cal.get("Earnings Date", [])
                if ed:
                    result["next_earnings"] = str(ed[0]) if isinstance(ed, list) else str(ed)
        except Exception:
            pass

        return result

    except Exception as exc:
        logger.warning("Failed to fetch earnings history for %s: %s", symbol, exc)
        return {"symbol": symbol, "error": str(exc)}


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Parse LLM JSON response, handling markdown fences."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse earnings analysis JSON")
    return {}


async def analyze_earnings(symbol: str, mode: str = "pre", **kwargs) -> Dict[str, Any]:
    """Analyze earnings for a symbol.

    Args:
        symbol: Ticker symbol.
        mode: "pre" for pre-earnings analysis, "post" for post-earnings.
        **kwargs: For post mode: actual_eps, est_eps.

    Returns:
        Structured analysis dict with recommendation.
    """
    from um_agent_coder.daemon.app import get_gemini_client, get_settings

    settings = get_settings()
    client = get_gemini_client()
    if not client:
        logger.error("Gemini client not available for earnings analysis")
        return {"symbol": symbol, "error": "Gemini client unavailable"}

    # Fetch historical data (runs sync yfinance in background)
    import asyncio

    loop = asyncio.get_running_loop()
    earnings_data = await loop.run_in_executor(None, _fetch_earnings_history, symbol)

    if earnings_data.get("error"):
        return {"symbol": symbol, "error": earnings_data["error"]}

    earnings_date = earnings_data.get("next_earnings", "unknown")
    earnings_str = json.dumps(earnings_data, indent=2, default=str)

    if mode == "pre":
        prompt = PRE_EARNINGS_PROMPT.format(
            symbol=symbol,
            earnings_date=earnings_date,
            earnings_data=earnings_str,
        )
    elif mode == "post":
        actual_eps = kwargs.get("actual_eps", 0)
        est_eps = kwargs.get("est_eps", 0)
        surprise_pct = ((actual_eps - est_eps) / abs(est_eps) * 100) if est_eps else 0
        prompt = POST_EARNINGS_PROMPT.format(
            symbol=symbol,
            actual_eps=actual_eps,
            est_eps=est_eps,
            surprise_pct=surprise_pct,
            earnings_data=earnings_str,
        )
    else:
        return {"symbol": symbol, "error": f"Invalid mode: {mode}"}

    try:
        response = await client.generate(
            prompt=prompt,
            model=settings.gemini_model_pro,
            temperature=0.2,
            max_tokens=4096,
        )
        text = response["text"] if isinstance(response, dict) else str(response)
        result = _parse_json_response(text)

        if result:
            # Ensure symbol is present
            result.setdefault("symbol", symbol)
            result.setdefault("mode", mode)
            result.setdefault("analyzed_at", datetime.now(timezone.utc).isoformat())
            logger.info(
                "Earnings analysis: %s mode=%s rec=%s",
                symbol, mode, result.get("recommendation", "?"),
            )
        return result

    except Exception as exc:
        logger.error("Earnings analysis failed for %s: %s", symbol, exc)
        return {"symbol": symbol, "error": str(exc)}
