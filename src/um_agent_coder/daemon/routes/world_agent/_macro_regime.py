"""Macro regime scorer — feeds market macro data to Gemini Pro for regime classification.

Fetches VIX, yields, credit proxies, commodities, and futures via Yahoo Finance,
computes derived signals (yield curve spread, credit spread proxy, VIX term structure),
then asks Gemini Pro to classify the current macro regime with position sizing guidance.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Macro instruments to fetch
MACRO_SYMBOLS = {
    # Volatility
    "^VIX": "VIX (spot)",
    "^VIX3M": "VIX 3-month (term structure)",
    # Yields
    "^TNX": "10Y Treasury Yield",
    "^IRX": "3M Treasury Yield",
    # Dollar
    "DX-Y.NYB": "US Dollar Index",
    # Commodities
    "GC=F": "Gold Futures",
    "CL=F": "Crude Oil Futures",
    # Equity futures
    "ES=F": "S&P 500 Futures",
    "NQ=F": "Nasdaq 100 Futures",
    # Fixed income / credit
    "HYG": "High Yield Corporate Bond ETF",
    "TLT": "20+ Year Treasury Bond ETF",
    "IEF": "7-10 Year Treasury Bond ETF",
}

MACRO_REGIME_SYSTEM_PROMPT = """\
You are a macro regime analyst for a quantitative hedge fund.

Given the current macro market data (prices, yields, spreads, term structures),
classify the current regime and provide actionable guidance.

Return ONLY valid JSON with this exact structure:
{
  "current_regime": "risk-on|neutral|risk-off|crisis",
  "regime_confidence": 0.85,
  "regime_shift_probability": 0.35,
  "predicted_regime_1w": "risk-off",
  "key_drivers": ["yield curve flattening", "credit spreads widening"],
  "position_sizing_multiplier": 0.8,
  "directional_bias": "defensive|neutral|aggressive",
  "sector_recommendations": {
    "overweight": ["XLU", "GLD"],
    "underweight": ["XLK", "XLY"]
  }
}

Guidelines:
- position_sizing_multiplier: 1.0 = full size, 0.5 = half, 0.0 = sit out
- crisis = VIX > 30, credit blowing out, yields spiking chaotically
- risk-off = VIX 20-30, credit widening, defensive rotation
- neutral = VIX 15-20, mixed signals, no strong trend
- risk-on = VIX < 15, tight credit, risk appetite strong
- Consider yield curve inversion as a medium-term warning, not an immediate crisis
- Factor in dollar strength: strong dollar = headwind for EM/commodities
- Gold rallying + bonds rallying = flight to safety signal
"""


async def _fetch_yahoo_quote(
    client: httpx.AsyncClient, symbol: str, range_: str = "5d"
) -> Optional[Dict[str, Any]]:
    """Fetch a single symbol's quote data from Yahoo Finance v8 API."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={range_}"
    try:
        resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None
        result = resp.json().get("chart", {}).get("result", [])
        if not result:
            return None
        meta = result[0].get("meta", {})
        indicators = result[0].get("indicators", {})
        closes = []
        if indicators.get("quote") and indicators["quote"][0].get("close"):
            closes = [c for c in indicators["quote"][0]["close"] if c is not None]

        return {
            "symbol": symbol,
            "price": meta.get("regularMarketPrice", 0),
            "previous_close": meta.get("chartPreviousClose", 0),
            "change_pct": (
                ((meta.get("regularMarketPrice", 0) - meta.get("chartPreviousClose", 0))
                 / meta.get("chartPreviousClose", 1) * 100)
                if meta.get("chartPreviousClose", 0) > 0 else 0
            ),
            "closes_5d": closes[-5:] if len(closes) >= 5 else closes,
        }
    except Exception as e:
        logger.debug("Failed to fetch %s: %s", symbol, e)
        return None


def _compute_return_5d(closes: List[float]) -> float:
    """Compute 5-day return from a list of closes."""
    if len(closes) >= 2 and closes[0] > 0:
        return (closes[-1] - closes[0]) / closes[0] * 100
    return 0.0


async def _fetch_macro_data() -> Dict[str, Any]:
    """Fetch all macro indicators and compute derived signals."""
    quotes: Dict[str, Dict[str, Any]] = {}

    async with httpx.AsyncClient(timeout=15.0) as client:
        for symbol in MACRO_SYMBOLS:
            quote = await _fetch_yahoo_quote(client, symbol)
            if quote:
                quotes[symbol] = quote

    # Build summary
    data: Dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat()}

    # Raw prices
    for symbol, label in MACRO_SYMBOLS.items():
        q = quotes.get(symbol)
        if q:
            data[label] = {
                "price": round(q["price"], 4),
                "change_1d_pct": round(q["change_pct"], 2),
                "return_5d_pct": round(_compute_return_5d(q.get("closes_5d", [])), 2),
            }

    # Derived: Yield curve spread (10Y - 3M)
    tnx = quotes.get("^TNX")
    irx = quotes.get("^IRX")
    if tnx and irx:
        # TNX and IRX are in percentage points (e.g. 4.25 = 4.25%)
        spread = tnx["price"] - irx["price"]
        data["yield_curve_spread_10y_3m"] = {
            "spread_pct": round(spread, 3),
            "inverted": spread < 0,
            "ten_year": round(tnx["price"], 3),
            "three_month": round(irx["price"], 3),
        }

    # Derived: Credit spread proxy (IEF 5d return - HYG 5d return)
    ief = quotes.get("IEF")
    hyg = quotes.get("HYG")
    if ief and hyg:
        ief_ret = _compute_return_5d(ief.get("closes_5d", []))
        hyg_ret = _compute_return_5d(hyg.get("closes_5d", []))
        credit_spread_proxy = ief_ret - hyg_ret
        data["credit_spread_proxy"] = {
            "spread_5d_pct": round(credit_spread_proxy, 3),
            "widening": credit_spread_proxy > 0,
            "ief_return_5d": round(ief_ret, 3),
            "hyg_return_5d": round(hyg_ret, 3),
        }

    # Derived: VIX term structure (spot vs VIX3M)
    vix = quotes.get("^VIX")
    vix3m = quotes.get("^VIX3M")
    if vix and vix3m and vix3m["price"] > 0:
        ratio = vix["price"] / vix3m["price"]
        data["vix_term_structure"] = {
            "spot_vix": round(vix["price"], 2),
            "vix3m": round(vix3m["price"], 2),
            "ratio": round(ratio, 3),
            "backwardation": ratio > 1.0,
            "contango": ratio < 1.0,
        }
    elif vix:
        data["vix_term_structure"] = {
            "spot_vix": round(vix["price"], 2),
            "vix3m": None,
            "ratio": None,
            "backwardation": None,
            "contango": None,
        }

    return data


async def score_macro_regime() -> Dict[str, Any]:
    """Score the current macro regime using market data + Gemini Pro analysis.

    Returns a dict with regime classification, sizing multiplier, and sector guidance.
    Falls back to a rules-based regime if LLM is unavailable.
    """
    import json

    # 1. Fetch macro data
    try:
        macro_data = await _fetch_macro_data()
    except Exception as e:
        logger.error("Failed to fetch macro data: %s", e)
        return _fallback_regime(error=str(e))

    if not macro_data or len(macro_data) <= 1:
        return _fallback_regime(error="no macro data available")

    # 2. Feed to Gemini Pro
    try:
        from um_agent_coder.daemon.app import get_llm_router, get_settings

        settings = get_settings()
        model = settings.gemini_model_pro
        router = get_llm_router()

        user_prompt = (
            "Analyze the following macro market data and classify the current regime.\n\n"
            f"```json\n{json.dumps(macro_data, indent=2, default=str)}\n```\n\n"
            "Return ONLY valid JSON matching the schema in the system prompt."
        )

        llm_result = await router.generate(
            prompt=user_prompt,
            system_prompt=MACRO_REGIME_SYSTEM_PROMPT,
            model=model,
            temperature=0.2,
            max_tokens=1024,
            provider=settings.world_agent_llm_provider or None,
        )

        raw_text = llm_result.get("text", "")
        regime_data = _parse_regime_response(raw_text)

        # Attach raw macro data for transparency
        regime_data["macro_data"] = macro_data
        regime_data["model_used"] = model
        regime_data["scored_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Macro regime scored: %s (confidence=%.2f, sizing=%.2f)",
            regime_data.get("current_regime", "unknown"),
            regime_data.get("regime_confidence", 0),
            regime_data.get("position_sizing_multiplier", 1.0),
        )

        return regime_data

    except Exception as e:
        logger.warning("LLM regime scoring failed, using rules-based fallback: %s", e)
        return _rules_based_regime(macro_data)


def _parse_regime_response(raw_text: str) -> Dict[str, Any]:
    """Parse the LLM response into a regime dict."""
    import json
    import re

    # Try to extract JSON from the response
    # Handle markdown code blocks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
    if json_match:
        raw_text = json_match.group(1).strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        brace_match = re.search(r"\{[\s\S]*\}", raw_text)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                return _fallback_regime(error="failed to parse LLM response")
        else:
            return _fallback_regime(error="no JSON in LLM response")

    # Validate required fields
    valid_regimes = {"risk-on", "neutral", "risk-off", "crisis"}
    regime = data.get("current_regime", "neutral")
    if regime not in valid_regimes:
        regime = "neutral"
    data["current_regime"] = regime

    # Clamp numeric values
    data["regime_confidence"] = max(0.0, min(1.0, float(data.get("regime_confidence", 0.5))))
    data["regime_shift_probability"] = max(0.0, min(1.0, float(data.get("regime_shift_probability", 0.5))))
    data["position_sizing_multiplier"] = max(0.0, min(1.5, float(data.get("position_sizing_multiplier", 1.0))))

    valid_biases = {"defensive", "neutral", "aggressive"}
    if data.get("directional_bias") not in valid_biases:
        data["directional_bias"] = "neutral"

    return data


def _rules_based_regime(macro_data: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback: simple rules-based regime when LLM is unavailable."""
    vix_data = macro_data.get("vix_term_structure", {})
    vix = vix_data.get("spot_vix", 18)
    yield_data = macro_data.get("yield_curve_spread_10y_3m", {})
    credit_data = macro_data.get("credit_spread_proxy", {})

    if vix >= 30:
        regime = "crisis"
        multiplier = 0.1
        bias = "defensive"
    elif vix >= 20:
        regime = "risk-off"
        multiplier = 0.5
        bias = "defensive"
    elif vix >= 15:
        regime = "neutral"
        multiplier = 0.8
        bias = "neutral"
    else:
        regime = "risk-on"
        multiplier = 1.0
        bias = "aggressive"

    # Adjust for credit stress
    if credit_data.get("widening"):
        multiplier *= 0.9

    # Adjust for yield curve inversion
    if yield_data.get("inverted"):
        multiplier *= 0.95

    return {
        "current_regime": regime,
        "regime_confidence": 0.5,
        "regime_shift_probability": 0.5,
        "predicted_regime_1w": regime,
        "key_drivers": [f"VIX at {vix:.1f}", "rules-based fallback"],
        "position_sizing_multiplier": round(max(0.0, min(1.5, multiplier)), 2),
        "directional_bias": bias,
        "sector_recommendations": {"overweight": [], "underweight": []},
        "macro_data": macro_data,
        "model_used": "rules-based-fallback",
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def _fallback_regime(error: str = "") -> Dict[str, Any]:
    """Default safe regime when nothing works."""
    return {
        "current_regime": "neutral",
        "regime_confidence": 0.3,
        "regime_shift_probability": 0.5,
        "predicted_regime_1w": "neutral",
        "key_drivers": [f"fallback: {error}" if error else "no data available"],
        "position_sizing_multiplier": 0.7,
        "directional_bias": "neutral",
        "sector_recommendations": {"overweight": [], "underweight": []},
        "macro_data": {},
        "model_used": "fallback",
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
