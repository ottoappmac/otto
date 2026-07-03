#!/usr/bin/env python3
"""Built-in MCP server: SEC EDGAR Filings.

Read-only tools over the SEC's free EDGAR APIs:

* full-text filing search          (efts.sec.gov)
* company submissions / filings    (data.sec.gov/submissions)
* company XBRL facts and concepts  (data.sec.gov/api/xbrl)
* cross-company XBRL frames        (data.sec.gov/api/xbrl/frames)
* CIK lookup by ticker             (sec.gov/files/company_tickers.json)
* filing-level document index      (sec.gov/Archives)

The SEC requires a polite ``User-Agent`` header identifying the caller
(see https://www.sec.gov/os/accessing-edgar-data).  We pull it from the
``EDGAR_USER_AGENT`` env var, which the backend hydrates from the OS
keychain at spawn time — the LLM context never sees the value.

This file is the canonical source for the ``edgar-sec`` builtin MCP.
The backend copies it into the per-MCP folder under
``mcp_server/edgar-sec/`` on every startup, then runs it inside a uv-
provisioned venv (``mcp[cli]`` + ``httpx``).

Trust boundaries:
* Only the SEC's documented public endpoints are reached.
* No credential is logged, returned, or echoed; ``EDGAR_USER_AGENT`` is
  itself effectively public (it's a contact string), but we still treat
  it as a secret for symmetry with the rest of the MCP credential flow.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.edgar_sec")


SEARCH_INDEX_URL = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANY_CONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{concept}.json"
)
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
FRAMES_URL = (
    "https://data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json"
)
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVES_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/index.json"
)

DEFAULT_TIMEOUT = 30.0
LONG_TIMEOUT = 60.0
MAX_LIMIT = 100

# EDGAR_USER_AGENT remains in ``required_secrets`` so the credential
# slot still appears in the Tools page → SEC EDGAR Filings → Credentials
# dialog (and so ``request_credential("edgar-sec", "EDGAR_USER_AGENT", ...)``
# pops the same secure dialog mid-chat).  Customising the value is
# strongly recommended — SEC asks every API client to identify itself
# so they can email the operator if traffic looks abusive.  The default
# below lets the MCP run out of the box for casual exploration; users
# who plan to do anything serious should replace it via the UI.
_REQUIRED_SECRETS = ("EDGAR_USER_AGENT",)

_DEFAULT_USER_AGENT = (
    "Otto Research Agent (built-in edgar-sec MCP) contact@otto.local"
)

_TICKERS_CACHE: dict[str, Any] = {"data": None}
_FALLBACK_WARNED = False


mcp = FastMCP("SEC EDGAR Filings")


def _resolve_user_agent() -> str:
    """Return the User-Agent header value, preferring the vault entry.

    Falls back to ``_DEFAULT_USER_AGENT`` when the user hasn't customised
    it yet so the MCP is usable on first run.  Logs a one-time WARNING so
    operators notice they should personalise the contact string via:

      * Tools page → SEC EDGAR Filings → Credentials, OR
      * ``request_credential("edgar-sec", "EDGAR_USER_AGENT", …)`` in chat.
    """
    global _FALLBACK_WARNED
    val = (os.environ.get("EDGAR_USER_AGENT") or "").strip()
    if val:
        return val
    if not _FALLBACK_WARNED:
        logger.warning(
            "EDGAR_USER_AGENT is not set; using built-in default %r. "
            "SEC asks every API client to identify itself — set a contact "
            "string ('Your Name your-email@example.com') via the Tools page "
            "(SEC EDGAR Filings → Credentials) or have the agent call "
            "request_credential('edgar-sec', 'EDGAR_USER_AGENT', ...).",
            _DEFAULT_USER_AGENT,
        )
        _FALLBACK_WARNED = True
    return _DEFAULT_USER_AGENT


def _check_secrets() -> None:
    """Kept as a no-op for callers that still invoke it.

    Previously raised when ``EDGAR_USER_AGENT`` was missing; now we
    transparently fall back to ``_DEFAULT_USER_AGENT`` in
    :func:`_resolve_user_agent` so the MCP works out of the box.  Left
    in place (as a no-op) to keep call sites untouched.
    """
    return None


def _headers() -> dict[str, str]:
    """Build the standard SEC request headers."""
    return {
        "User-Agent": _resolve_user_agent(),
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }


def _client(*, timeout: float = DEFAULT_TIMEOUT) -> httpx.Client:
    """Return a configured ``httpx.Client``.

    A new client per call keeps the FastMCP tool functions trivially
    re-entrant — SEC's rate limit (10 req/s) is enforced by the caller,
    not by us, but spinning up a client is cheap relative to the HTTP
    round-trip.
    """
    return httpx.Client(timeout=timeout, follow_redirects=True)


def _normalize_cik(value: str) -> str:
    """Return a 10-digit zero-padded CIK string from a CIK or ticker."""
    if value is None:
        raise ValueError("CIK or ticker is required")
    value = value.strip()
    if not value:
        raise ValueError("CIK or ticker is required")

    if value.isdigit():
        return value.zfill(10)

    cik = _ticker_to_cik(value)
    if cik is None:
        raise ValueError(
            f"Could not resolve {value!r} to a CIK. Pass a 10-digit CIK or "
            f"a known ticker symbol; use search_company_by_ticker for fuzzy "
            f"name lookups."
        )
    return cik


def _load_ticker_map() -> dict[str, dict[str, Any]]:
    """Fetch and cache the SEC's ticker→CIK mapping.

    SEC publishes a flat ``company_tickers.json`` snapshot daily.  We
    cache it for the lifetime of the subprocess; the mapping changes
    rarely (new IPOs / delistings) and the file is small (~2MB).
    """
    cached = _TICKERS_CACHE.get("data")
    if cached is not None:
        return cached  # type: ignore[return-value]
    _check_secrets()
    with _client() as client:
        resp = client.get(TICKERS_URL, headers=_headers())
        resp.raise_for_status()
        raw = resp.json()
    ticker_map: dict[str, dict[str, Any]] = {}
    for row in raw.values():
        ticker = str(row.get("ticker", "")).upper()
        cik = str(row.get("cik_str", "")).zfill(10)
        if not ticker or not cik:
            continue
        ticker_map[ticker] = {
            "cik": cik,
            "ticker": ticker,
            "title": row.get("title", ""),
        }
    _TICKERS_CACHE["data"] = ticker_map
    return ticker_map


def _ticker_to_cik(ticker: str) -> Optional[str]:
    return _load_ticker_map().get(ticker.upper(), {}).get("cik")


@mcp.tool()
def search_filings(
    query: str,
    form_type: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 10,
) -> dict[str, Any]:
    """Full-text search SEC EDGAR filings.

    Args:
        query: Keywords, phrase, company name, or ticker.
        form_type: Optional comma-separated list (e.g. ``"10-K,10-Q"``).
        date_from: Inclusive ISO date (``YYYY-MM-DD``); empty for no lower bound.
        date_to: Inclusive ISO date (``YYYY-MM-DD``); empty for no upper bound.
        limit: Max hits to return (1-100).

    Returns the total match count and a list of hits with entity name,
    form type, filing date, accession number, and a link to the filing.
    """
    _check_secrets()
    if not query or not query.strip():
        raise ValueError("query is required")
    limit = max(1, min(int(limit), MAX_LIMIT))

    params: dict[str, str] = {"q": query.strip(), "from": "0", "size": str(limit)}
    if form_type:
        params["forms"] = form_type
    if date_from or date_to:
        params["dateRange"] = "custom"
    if date_from:
        params["startdt"] = date_from
    if date_to:
        params["enddt"] = date_to

    with _client() as client:
        resp = client.get(SEARCH_INDEX_URL, params=params, headers=_headers())
        resp.raise_for_status()
        data = resp.json()

    hits = data.get("hits", {}).get("hits", []) or []
    total = data.get("hits", {}).get("total", {}).get("value", 0)

    results: list[dict[str, Any]] = []
    for hit in hits:
        src = hit.get("_source", {}) or {}
        adsh = src.get("adsh", "") or _id_to_accession(hit.get("_id", ""))
        cik_int = _first_int(src.get("ciks", []))
        results.append(
            {
                "accession_number": adsh,
                "entity_name": src.get("display_names", [""])[0],
                "ciks": src.get("ciks", []),
                "form_type": src.get("form", ""),
                "file_date": src.get("file_date", ""),
                "period_of_report": src.get("period_of_report", ""),
                "filing_url": _filing_url(cik_int, adsh) if cik_int and adsh else "",
            }
        )
    return {"query": query, "total": total, "results": results}


@mcp.tool()
def get_company_submissions(cik: str) -> dict[str, Any]:
    """Get a company's profile and recent filings.

    Args:
        cik: 10-digit CIK or ticker symbol (e.g. ``"AAPL"`` or ``"0000320193"``).

    Returns company metadata (name, SIC, tickers, exchanges) and the
    most-recent 50 filings with form type, date, and accession number.
    """
    cik_padded = _normalize_cik(cik)
    _check_secrets()
    with _client() as client:
        resp = client.get(
            SUBMISSIONS_URL.format(cik=cik_padded), headers=_headers(),
        )
        if resp.status_code == 404:
            return {"error": f"No company found for CIK {cik_padded}"}
        resp.raise_for_status()
        data = resp.json()

    recent = data.get("filings", {}).get("recent", {}) or {}
    accn = recent.get("accessionNumber", []) or []
    cik_int = int(cik_padded)
    filings: list[dict[str, Any]] = []
    for i in range(min(len(accn), 50)):
        acc = accn[i]
        filings.append(
            {
                "accession_number": acc,
                "filing_date": _safe_index(recent.get("filingDate"), i),
                "report_date": _safe_index(recent.get("reportDate"), i),
                "form": _safe_index(recent.get("form"), i),
                "primary_document": _safe_index(recent.get("primaryDocument"), i),
                "primary_doc_description": _safe_index(
                    recent.get("primaryDocDescription"), i,
                ),
                "size": _safe_index(recent.get("size"), i),
                "filing_url": _filing_url(cik_int, acc),
            }
        )
    return {
        "cik": data.get("cik"),
        "name": data.get("name"),
        "sic": data.get("sic"),
        "sic_description": data.get("sicDescription"),
        "tickers": data.get("tickers", []),
        "exchanges": data.get("exchanges", []),
        "state_of_incorporation": data.get("stateOfIncorporation"),
        "fiscal_year_end": data.get("fiscalYearEnd"),
        "ein": data.get("ein"),
        "category": data.get("category"),
        "recent_filings": filings,
        "total_recent_filings": len(accn),
    }


@mcp.tool()
def search_company_by_ticker(query: str) -> dict[str, Any]:
    """Look up CIKs by ticker or company name (substring, case-insensitive).

    Args:
        query: Ticker (exact match) or company name fragment.

    Returns up to 25 matches with CIK, ticker, and registered name.
    Use the returned CIK with ``get_company_submissions`` or
    ``get_company_facts``.
    """
    if not query or not query.strip():
        raise ValueError("query is required")
    needle = query.strip().upper()
    ticker_map = _load_ticker_map()

    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    if needle in ticker_map:
        row = ticker_map[needle]
        results.append(row)
        seen.add(row["cik"])

    for row in ticker_map.values():
        if len(results) >= 25:
            break
        if row["cik"] in seen:
            continue
        if needle in row["ticker"].upper() or needle in row["title"].upper():
            results.append(row)
            seen.add(row["cik"])

    return {"query": query, "results": results}


@mcp.tool()
def get_company_facts(
    cik: str, taxonomy: str = "us-gaap", concept: str = "",
) -> dict[str, Any]:
    """Get XBRL facts (financial line items over time) for a company.

    When ``concept`` is provided, returns a recent time series for that
    concept (e.g. ``"Revenues"``, ``"Assets"``).  When omitted, returns
    a summary of every available concept's latest value.

    Args:
        cik: CIK or ticker.
        taxonomy: Usually ``"us-gaap"`` (default), sometimes ``"ifrs-full"`` or ``"dei"``.
        concept: Optional XBRL tag; leave blank for the full summary.
    """
    cik_padded = _normalize_cik(cik)
    _check_secrets()

    if concept:
        url = COMPANY_CONCEPT_URL.format(
            cik=cik_padded, taxonomy=taxonomy, concept=concept,
        )
        with _client() as client:
            resp = client.get(url, headers=_headers())
            if resp.status_code == 404:
                return {
                    "error": f"Concept {concept!r} not found for CIK "
                             f"{cik_padded} under taxonomy {taxonomy!r}.",
                }
            resp.raise_for_status()
            data = resp.json()
        units = data.get("units", {}) or {}
        recent: list[dict[str, Any]] = []
        for unit_type, values in units.items():
            for v in values:
                recent.append({**v, "unit": unit_type})
        recent.sort(key=lambda r: r.get("end", ""), reverse=True)
        return {
            "cik": cik_padded,
            "entity_name": data.get("entityName"),
            "taxonomy": taxonomy,
            "concept": concept,
            "label": data.get("label"),
            "description": (data.get("description") or "")[:500],
            "recent_values": recent[:25],
        }

    url = COMPANY_FACTS_URL.format(cik=cik_padded)
    with _client(timeout=LONG_TIMEOUT) as client:
        resp = client.get(url, headers=_headers())
        if resp.status_code == 404:
            return {"error": f"No XBRL facts found for CIK {cik_padded}"}
        resp.raise_for_status()
        data = resp.json()

    tax_facts = (data.get("facts", {}) or {}).get(taxonomy, {}) or {}
    summary: dict[str, Any] = {}
    for concept_name, concept_data in list(tax_facts.items())[:200]:
        latest = _latest_concept_value(concept_data.get("units", {}))
        summary[concept_name] = {
            "label": concept_data.get("label", ""),
            "latest": latest,
        }
    return {
        "cik": cik_padded,
        "entity_name": data.get("entityName"),
        "taxonomy": taxonomy,
        "concepts_count": len(tax_facts),
        "concepts_summary": summary,
    }


@mcp.tool()
def get_filing_document(cik: str, accession_number: str) -> dict[str, Any]:
    """Get the document index for one specific filing.

    Args:
        cik: CIK or ticker.
        accession_number: e.g. ``"0000320193-24-000123"`` (with hyphens).

    Returns links to every document in the filing (10-K HTML, exhibits,
    XBRL files, etc.).
    """
    cik_padded = _normalize_cik(cik)
    _check_secrets()
    cik_int = int(cik_padded)
    acc_clean = accession_number.replace("-", "")
    url = ARCHIVES_INDEX_URL.format(cik_int=cik_int, acc_clean=acc_clean)

    with _client() as client:
        resp = client.get(url, headers=_headers())
        if resp.status_code == 404:
            return {
                "error": f"No filing index for CIK {cik_padded} accession "
                         f"{accession_number}.",
            }
        resp.raise_for_status()
        data = resp.json()

    base = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}"
    docs = []
    for item in (data.get("directory", {}) or {}).get("item", []) or []:
        name = item.get("name", "")
        docs.append(
            {
                "name": name,
                "type": item.get("type", ""),
                "size": item.get("size", ""),
                "last_modified": item.get("last-modified", ""),
                "url": f"{base}/{name}" if name else "",
            }
        )
    return {
        "cik": cik_padded,
        "accession_number": accession_number,
        "filing_index_url": f"{base}/",
        "documents": docs,
    }


@mcp.tool()
def get_xbrl_frames(
    taxonomy: str,
    concept: str,
    year: int,
    quarter: int = 0,
    unit: str = "USD",
    instantaneous: bool = False,
) -> dict[str, Any]:
    """Cross-company snapshot of one XBRL concept in one period.

    Args:
        taxonomy: Usually ``"us-gaap"``.
        concept: XBRL tag (e.g. ``"Assets"``, ``"Revenues"``).
        year: Calendar year (4-digit).
        quarter: 1-4 for a specific quarter, 0 for the full calendar year.
        unit: e.g. ``"USD"``, ``"shares"``, ``"USD-per-shares"``.
        instantaneous: True for point-in-time concepts (balances), False
            for duration concepts (revenues, expenses).

    Returns the top 25 entities ranked by reported value.
    """
    _check_secrets()
    period = f"CY{year}"
    if quarter and 1 <= quarter <= 4:
        period += f"Q{quarter}"
    if instantaneous:
        period += "I"

    url = FRAMES_URL.format(
        taxonomy=taxonomy, concept=concept, unit=unit, period=period,
    )
    with _client() as client:
        resp = client.get(url, headers=_headers())
        if resp.status_code == 404:
            return {
                "error": f"No frame data for {taxonomy}/{concept}/{unit} in "
                         f"{period}. Try toggling instantaneous or quarter.",
            }
        resp.raise_for_status()
        data = resp.json()

    entries = data.get("data", []) or []
    entries_sorted = sorted(
        entries,
        key=lambda e: e.get("val", 0) if isinstance(e.get("val"), (int, float)) else 0,
        reverse=True,
    )
    return {
        "taxonomy": taxonomy,
        "concept": concept,
        "unit": unit,
        "period": period,
        "label": data.get("label", ""),
        "description": (data.get("description") or "")[:300],
        "total_entities": data.get("pts", len(entries)),
        "top_entities": entries_sorted[:25],
    }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _safe_index(seq: Optional[list[Any]], i: int) -> Any:
    if not seq or i >= len(seq):
        return None
    return seq[i]


def _first_int(values: Any) -> Optional[int]:
    if not values:
        return None
    head = values[0] if isinstance(values, list) else values
    try:
        return int(head)
    except (TypeError, ValueError):
        return None


def _id_to_accession(eid: str) -> str:
    """EDGAR full-text-search ``_id`` is ``"<accession>:<filename>"``."""
    if not eid:
        return ""
    return eid.split(":", 1)[0]


_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")


def _filing_url(cik_int: Optional[int], accession_number: str) -> str:
    """Build the human-readable index page URL for a filing.

    Returns the empty string when ``cik_int`` or ``accession_number`` is
    missing or malformed — callers treat empty as "no link available".
    """
    if not cik_int or not accession_number:
        return ""
    if not _ACCESSION_RE.match(accession_number):
        return ""
    acc_clean = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/"
        f"{accession_number}-index.htm"
    )


def _latest_concept_value(units: dict[str, Any]) -> Optional[dict[str, Any]]:
    latest: Optional[dict[str, Any]] = None
    for unit_type, values in (units or {}).items():
        if not values:
            continue
        candidate = max(values, key=lambda v: v.get("end", ""))
        candidate = {
            "value": candidate.get("val"),
            "end": candidate.get("end"),
            "form": candidate.get("form"),
            "fy": candidate.get("fy"),
            "fp": candidate.get("fp"),
            "unit": unit_type,
        }
        if latest is None or (candidate["end"] or "") > (latest["end"] or ""):
            latest = candidate
    return latest


if __name__ == "__main__":
    mcp.run()
