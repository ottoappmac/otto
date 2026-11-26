"""Default Gemini model names, and the retired names that map onto them.

Google retires Gemini model names. A retired name returns 404 NOT_FOUND, so a
saved setting such as ``gemini-2.5-flash`` would stop every Gemini chat and
video call. ``current_gemini_model`` swaps those old defaults for the current
one. Any other name is passed through unchanged.
"""

from __future__ import annotations

# The "latest" alias follows Google's current Flash model, so it does not retire.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_GEMINI_LIVE_MODEL = "gemini-3.8-live"

# Chat requests: give up after 3 minutes and retry twice, not langchain's
# default of no timeout and 6 retries.
GEMINI_CHAT_TIMEOUT_S = 180
GEMINI_CHAT_MAX_RETRIES = 2

# Old Otto defaults (and their dated variants) that Google has retired.
RETIRED_GEMINI_MODELS = frozenset({
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.5-flash",
})
RETIRED_GEMINI_LIVE_MODELS = frozenset({
    "gemini-2.0-flash-live-001",
    "gemini-live-2.5-flash-preview",
})


def current_gemini_model(name: str | None) -> str:
    """*name*, or the default when it is empty or a retired default."""
    name = (name or "").strip()
    if not name or name.removeprefix("models/") in RETIRED_GEMINI_MODELS:
        return DEFAULT_GEMINI_MODEL
    return name


def current_gemini_live_model(name: str | None) -> str:
    """*name*, or the Live default when it is empty or retired."""
    name = (name or "").strip()
    if not name or name.removeprefix("models/") in RETIRED_GEMINI_LIVE_MODELS:
        return DEFAULT_GEMINI_LIVE_MODEL
    return name


def gemini_error_message(exc: BaseException) -> tuple[str, str] | None:
    """A (user_message, error_code) for a Gemini busy/timeout/missing-model error, else None.

    google-genai puts the status in the text, e.g. ``503 UNAVAILABLE. {...
    'This model is currently experiencing high demand' ...}``. langchain wraps
    that error, so this checks the whole exception chain.
    """
    texts, seen, cur = [], set(), exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        texts.append(f"{type(cur).__module__}.{type(cur).__name__} {cur}")
        cur = cur.__cause__ or cur.__context__
    text = " ".join(texts)
    low = text.lower()
    # Only Gemini errors: google.genai / langchain_google_genai types, or a Gemini URL or model name.
    if not any(k in low for k in ("google.genai", "langchain_google_genai", "generativelanguage", "gemini")):
        return None
    if "unavailable" in low or "high demand" in low or "503" in text:
        return ("Gemini is busy right now (503: high demand). Try again in a minute, "
                "or choose another Gemini model in Settings → LLM.", "llm_overloaded")
    if "deadline_exceeded" in low or "504" in text or "timeout" in low or "timed out" in low:
        return ("Gemini took too long to answer (timeout). Try again, or choose a faster "
                "Gemini model in Settings → LLM.", "timeout")
    if "not_found" in low or ("404" in text and "model" in low):
        return ("Google does not offer this Gemini model any more, or the name is wrong. "
                f"Choose another model in Settings → LLM (for example {DEFAULT_GEMINI_MODEL}).",
                "llm_model_not_found")
    return None
