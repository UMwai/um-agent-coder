"""SEC Filing Analyzer — deep analysis of recent SEC filings via Gemini.

For held positions:
- Checks EDGAR RSS for recent 8-K filings (last 7 days)
- Identifies material events (items 1.01, 2.01, 5.02, 8.01)
- Uses Gemini Pro for deep analysis on material events
- Uses Flash for quick screening of routine filings
- Checks insider transactions

Returns structured analysis with impact scores and recommendations.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

# Material 8-K item numbers that require deep analysis
MATERIAL_ITEMS = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.04": "Triggering Events That Accelerate or Increase Obligations",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Transfer",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure/Election of Directors or Officers",
    "5.03": "Amendments to Articles or Bylaws",
    "8.01": "Other Events (material events not covered above)",
}

MATERIAL_ANALYSIS_PROMPT = """\
You are a securities analyst for an AI hedge fund. Analyze this SEC filing
for {symbol} and determine its impact on the investment thesis.

## Filing Details
{filing_details}

## Task
1. Summarize the filing in 2-3 sentences.
2. Assess the impact on the stock price (positive, negative, neutral).
3. Score the impact from 0.0 (no impact) to 1.0 (major impact).
4. Recommend action for a holder of this stock.

Return a JSON object:
{{
  "symbol": "{symbol}",
  "filing_type": "{filing_type}",
  "summary": "<2-3 sentence summary>",
  "impact": "positive | negative | neutral",
  "impact_score": <float 0.0 to 1.0>,
  "recommendation": "HOLD | EXIT | WATCH",
  "key_items": [<list of material item numbers found>],
  "reasoning": "<detailed reasoning>"
}}
"""

ROUTINE_SCREENING_PROMPT = """\
Quick screen this SEC filing for {symbol}:
Filing type: {filing_type}
Title: {title}

Is this material? Rate impact 0.0 (routine) to 1.0 (critical).
Reply as JSON: {{"impact_score": <float>, "summary": "<one line>", "recommendation": "HOLD|EXIT|WATCH"}}
"""

SEC_EDGAR_RSS_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_EDGAR_FULLTEXT = "https://efts.sec.gov/LATEST/search-index"
SEC_EDGAR_COMPANY = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_USER_AGENT = "um-agent-coder/1.0 admin@example.com"


async def _fetch_company_cik(symbol: str, client: httpx.AsyncClient) -> str | None:
    """Look up CIK number for a ticker symbol from SEC EDGAR."""
    try:
        resp = await client.get(
            f"https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&dateRange=custom"
            f"&forms=10-K,10-Q,8-K&from=0&size=1",
            headers={"User-Agent": EDGAR_USER_AGENT},
        )
        if resp.status_code == 200:
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                return hits[0].get("_source", {}).get("entity_id")
    except Exception:
        pass
    return None


async def _fetch_recent_filings(
    symbol: str,
    client: httpx.AsyncClient,
    days: int = 7,
) -> List[Dict[str, Any]]:
    """Fetch recent SEC filings for a symbol from EDGAR full-text search."""
    filings: List[Dict[str, Any]] = []
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    try:
        # Search EDGAR by company ticker
        resp = await client.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q": f'"{symbol}"',
                "dateRange": "custom",
                "startdt": start_date.strftime("%Y-%m-%d"),
                "enddt": end_date.strftime("%Y-%m-%d"),
                "forms": "8-K,8-K/A",
            },
            headers={"User-Agent": EDGAR_USER_AGENT},
        )

        if resp.status_code != 200:
            # Fallback: RSS feed
            return await _fetch_rss_filings(symbol, client, days)

        data = resp.json()
        for hit in data.get("hits", {}).get("hits", [])[:10]:
            source = hit.get("_source", {})
            company = source.get("display_names", [""])[0] if source.get("display_names") else ""

            # Check if this filing is actually for our symbol
            tickers = source.get("tickers", [])
            if tickers and symbol.upper() not in [t.upper() for t in tickers]:
                # Try company name match as fallback
                if symbol.upper() not in company.upper():
                    continue

            filing = {
                "symbol": symbol,
                "company": company,
                "filing_type": source.get("form_type", "8-K"),
                "filed_date": source.get("file_date", ""),
                "title": source.get("display_description", ""),
                "items": source.get("items", []),
                "file_url": source.get("file_url", ""),
            }
            filings.append(filing)

    except Exception as exc:
        logger.warning("EDGAR search failed for %s: %s", symbol, exc)
        return await _fetch_rss_filings(symbol, client, days)

    return filings


async def _fetch_rss_filings(
    symbol: str,
    client: httpx.AsyncClient,
    days: int = 7,
) -> List[Dict[str, Any]]:
    """Fallback: fetch filings from EDGAR RSS feed."""
    filings: List[Dict[str, Any]] = []
    try:
        resp = await client.get(
            SEC_EDGAR_COMPANY,
            params={
                "action": "getcompany",
                "company": symbol,
                "type": "8-K",
                "dateb": "",
                "owner": "include",
                "count": "10",
                "search_text": "",
                "output": "atom",
            },
            headers={"User-Agent": EDGAR_USER_AGENT},
        )
        if resp.status_code != 200:
            return filings

        entry_pat = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
        title_pat = re.compile(r"<title[^>]*>(.*?)</title>")
        updated_pat = re.compile(r"<updated>(.*?)</updated>")

        for match in entry_pat.finditer(resp.text):
            entry = match.group(1)
            title_m = title_pat.search(entry)
            updated_m = updated_pat.search(entry)

            if title_m:
                filings.append({
                    "symbol": symbol,
                    "filing_type": "8-K",
                    "title": title_m.group(1)[:200],
                    "filed_date": updated_m.group(1) if updated_m else "",
                    "items": [],
                })

    except Exception as exc:
        logger.debug("SEC RSS fallback failed for %s: %s", symbol, exc)

    return filings


async def _fetch_insider_transactions(
    symbol: str,
    client: httpx.AsyncClient,
) -> List[Dict[str, Any]]:
    """Check insider transactions via EDGAR."""
    transactions: List[Dict[str, Any]] = []
    try:
        # Use SEC EDGAR full-text search for Form 4 filings
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)

        resp = await client.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q": f'"{symbol}"',
                "dateRange": "custom",
                "startdt": start_date.strftime("%Y-%m-%d"),
                "enddt": end_date.strftime("%Y-%m-%d"),
                "forms": "4",
            },
            headers={"User-Agent": EDGAR_USER_AGENT},
        )

        if resp.status_code == 200:
            data = resp.json()
            for hit in data.get("hits", {}).get("hits", [])[:5]:
                source = hit.get("_source", {})
                transactions.append({
                    "symbol": symbol,
                    "filer": source.get("display_names", [""])[0] if source.get("display_names") else "",
                    "filed_date": source.get("file_date", ""),
                    "form_type": "4",
                    "description": source.get("display_description", ""),
                })

    except Exception as exc:
        logger.debug("Insider transaction fetch failed for %s: %s", symbol, exc)

    return transactions


def _is_material_filing(filing: Dict[str, Any]) -> bool:
    """Check if a filing contains material event items."""
    items = filing.get("items", [])
    if not items:
        # If no items parsed, check title for material keywords
        title = filing.get("title", "").lower()
        material_keywords = [
            "acquisition", "merger", "officer", "director",
            "impairment", "restatement", "delisting", "default",
            "material", "amendment", "termination",
        ]
        return any(kw in title for kw in material_keywords)

    for item in items:
        item_str = str(item).strip()
        for material_item in MATERIAL_ITEMS:
            if item_str.startswith(material_item):
                return True
    return False


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Parse LLM JSON response."""
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
    return {}


async def analyze_sec_filings(symbols: list[str]) -> Dict[str, Any]:
    """Analyze recent SEC filings for a list of symbols.

    For each symbol:
    1. Fetch recent 8-K filings from EDGAR (last 7 days)
    2. Classify as material or routine
    3. Deep-analyze material filings via Gemini Pro
    4. Quick-screen routine filings via Gemini Flash
    5. Check insider transactions

    Returns:
        {"filings": [{"symbol", "filing_type", "impact_score", "summary",
                       "recommendation", ...}],
         "insider_transactions": [...],
         "analyzed_at": "..."}
    """
    from um_agent_coder.daemon.app import get_gemini_client, get_settings

    settings = get_settings()
    client = get_gemini_client()
    if not client:
        logger.error("Gemini client not available for SEC analysis")
        return {"filings": [], "error": "Gemini client unavailable"}

    all_filings: List[Dict[str, Any]] = []
    all_insider: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=20.0) as http:
        for symbol in symbols[:20]:  # Cap to avoid rate limits
            try:
                # Fetch filings
                filings = await _fetch_recent_filings(symbol, http)

                for filing in filings:
                    is_material = _is_material_filing(filing)

                    if is_material:
                        # Deep analysis with Pro model
                        try:
                            filing_details = json.dumps(filing, indent=2, default=str)
                            prompt = MATERIAL_ANALYSIS_PROMPT.format(
                                symbol=symbol,
                                filing_type=filing.get("filing_type", "8-K"),
                                filing_details=filing_details,
                            )
                            response = await client.generate(
                                prompt=prompt,
                                model=settings.gemini_model_pro,
                                temperature=0.2,
                                max_tokens=2048,
                            )
                            text = response["text"] if isinstance(response, dict) else str(response)
                            analysis = _parse_json_response(text)
                            if analysis:
                                analysis["is_material"] = True
                                analysis.setdefault("symbol", symbol)
                                all_filings.append(analysis)
                            else:
                                all_filings.append({
                                    "symbol": symbol,
                                    "filing_type": filing.get("filing_type", "8-K"),
                                    "impact_score": 0.5,
                                    "summary": filing.get("title", "Material filing — analysis failed"),
                                    "recommendation": "WATCH",
                                    "is_material": True,
                                })
                        except Exception as exc:
                            logger.warning("Deep analysis failed for %s: %s", symbol, exc)
                            all_filings.append({
                                "symbol": symbol,
                                "filing_type": filing.get("filing_type", "8-K"),
                                "impact_score": 0.3,
                                "summary": filing.get("title", "Filing detected"),
                                "recommendation": "WATCH",
                                "is_material": True,
                                "error": str(exc),
                            })
                    else:
                        # Quick screening with Flash model
                        try:
                            prompt = ROUTINE_SCREENING_PROMPT.format(
                                symbol=symbol,
                                filing_type=filing.get("filing_type", "8-K"),
                                title=filing.get("title", "Unknown"),
                            )
                            response = await client.generate(
                                prompt=prompt,
                                model=settings.gemini_model_flash,
                                temperature=0.1,
                                max_tokens=512,
                            )
                            text = response["text"] if isinstance(response, dict) else str(response)
                            analysis = _parse_json_response(text)
                            if analysis:
                                analysis["symbol"] = symbol
                                analysis["filing_type"] = filing.get("filing_type", "8-K")
                                analysis["is_material"] = False
                                all_filings.append(analysis)
                        except Exception:
                            # Routine filings — skip on error
                            pass

                # Insider transactions
                insiders = await _fetch_insider_transactions(symbol, http)
                all_insider.extend(insiders)

            except Exception as exc:
                logger.warning("SEC analysis failed for %s: %s", symbol, exc)

    # Sort by impact score descending
    all_filings.sort(key=lambda x: x.get("impact_score", 0), reverse=True)

    result = {
        "filings": all_filings,
        "insider_transactions": all_insider,
        "symbols_analyzed": len(symbols),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "SEC analysis: %d filings across %d symbols (%d material)",
        len(all_filings),
        len(symbols),
        sum(1 for f in all_filings if f.get("is_material")),
    )

    return result
