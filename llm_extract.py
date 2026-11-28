#!/usr/bin/env python3
"""
llm_extract.py
==============
Generic "unstructured document -> structured JSON" extraction, using the
repo's existing Groq (OpenAI-compatible) LLM pattern (see repo-inventory/
canonicalize.py) with a Gemini fallback -- both already configured via
~/.env.local for the "AI Advisor" feature.

Derived from patents MY269019 "System and method for generating queryable
structured document from an unstructured document using machine learning"
(IIIT-Hyderabad, LTRC) and MY269128 "System for extracting and analyzing
nutritional information from food labels using a large language model"
(IIIT-Hyderabad, MLL) -- same core technique (unstructured text -> a fixed
schema via an LLM), applied here to two places in this repo that currently
rely on brittle regex/layout parsing:
  - SEBI DRHP filing text (SEBI_web_scraper.ipynb)
  - Portfolio broker-statement text (portfolio.py's PDF parsing)

Usage:
    from llm_extract import extract_structured

    schema = {
        "company_name": "string",
        "issue_size_cr": "number, issue size in INR crore",
        "price_band_low": "number",
        "price_band_high": "number",
        "listing_exchange": "string, e.g. NSE/BSE/both",
    }
    result = extract_structured(drhp_text, schema)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Provider registry: name -> (api-key env var, model env var, default model, base URL).
# Adding a provider is one new row here; no caller changes.
PROVIDERS: dict = {
    "groq": {
        "key_var": "GROQ_API_KEY",
        "model_var": "GROQ_MODEL",
        "default_model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "gemini": {
        "key_var": "GEMINI_API_KEY",
        "model_var": "GEMINI_MODEL",
        "default_model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}
DEFAULT_PROVIDER = "groq"
DEFAULT_FALLBACK_PROVIDER = "gemini"
TEMPERATURE = 0


def _load_env_local(env_path: Optional[Path] = None) -> dict:
    """Minimal .env.local loader (avoids a hard python-dotenv dependency).

    Reads KEY=VALUE lines from *env_path* (default ``~/.env.local``); blank and
    ``#`` lines are skipped. Returns {} when the file does not exist.
    """
    env_path = env_path or Path.home() / ".env.local"
    values: dict = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        values[key] = val
    return values


def _client_and_model(provider: str):
    """Return (openai.OpenAI-compatible client, model_name) for a PROVIDERS entry.

    Credentials come from ``~/.env.local`` overlaid by the process environment.
    """
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(f"unknown provider {provider!r}")
    env = {**_load_env_local(), **os.environ}
    from openai import OpenAI

    key = env.get(spec["key_var"])
    if not key:
        raise RuntimeError(f"{spec['key_var']} not found in ~/.env.local or environment")
    client = OpenAI(api_key=key, base_url=spec["base_url"])
    return client, env.get(spec["model_var"], spec["default_model"])


def _extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of a model response.

    Models sometimes wrap the object in prose or ```json fences; both are tolerated.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start < 0:
        raise ValueError(f"no JSON object found in model response: {text[:200]!r}")
    # raw_decode stops at the end of the first complete object, so trailing prose
    # (which may itself contain braces) is ignored.
    obj, _ = json.JSONDecoder().raw_decode(candidate, start)
    return obj


def _normalize_whitespace(s: str) -> str:
    """Collapse runs of whitespace to one space and strip the ends."""
    return re.sub(r"\s+", " ", s).strip()


def _verify_citation(span: object, source_text: str) -> bool:
    """True if *span* appears verbatim (allowing whitespace differences) in
    source_text. Array-typed fields (e.g. a list of holdings) legitimately
    come back as a list of per-item citations rather than one string --
    found via the promptfoo pilot, where a correctly-extracted `holdings`
    array was silently flagged grounded=False because this only accepted
    str. Every element must verify for the field to count as grounded."""
    if isinstance(span, list):
        return bool(span) and all(_verify_citation(item, source_text) for item in span)
    if not isinstance(span, str) or not span.strip():
        return False
    return _contains_whole(_normalize_whitespace(source_text), _normalize_whitespace(span))


def _splits_token(text: str, i: int) -> bool:
    """True if cutting *text* before index *i* would split a word or a number.

    A cut between two letters/digits splits a word; a cut beside a ``,`` or ``.``
    that sits between digits splits a number such as ``15,230.75``.
    """
    if i <= 0 or i >= len(text):
        return False
    left, right = text[i - 1], text[i]
    if left.isalnum() and right.isalnum():
        return True
    if left.isdigit() and right in ",.":
        return text[i + 1 : i + 2].isdigit()
    if left in ",." and right.isdigit():
        return i >= 2 and text[i - 2].isdigit()
    return False


def _contains_whole(source: str, span: str) -> bool:
    """True if some occurrence of *span* in *source* starts and ends on token boundaries."""
    start = source.find(span)
    while start >= 0:
        if not _splits_token(source, start) and not _splits_token(source, start + len(span)):
            return True
        start = source.find(span, start + 1)
    return False


def extract_structured(
    text: str,
    schema: dict,
    instructions: str = "",
    provider: str = DEFAULT_PROVIDER,
    fallback_provider: Optional[str] = DEFAULT_FALLBACK_PROVIDER,
    require_citations: bool = True,
    derived: Optional[dict] = None,
) -> dict:
    """
    Extract a JSON object matching *schema* out of unstructured *text* using
    an LLM. *schema* maps field name -> a short description of what to put
    there (used verbatim in the prompt, not validated as a JSON-schema spec --
    kept simple since every caller here just wants a flat field->value dict).

    require_citations (default True): also asks the model for a verbatim
    source-text span backing each non-null field, then verifies that span
    actually appears in *text* before trusting it -- citation-grounding, one
    of the few concretely effective mitigations in the literature on LLM
    extraction from financial documents. That literature is sobering: on
    FinanceBench (real 10-Ks/10-Qs/8-Ks), GPT-4-Turbo with retrieval got 81%
    of questions wrong; FinTagging found the best model only 17% accurate at
    linking extracted facts to correct XBRL taxonomy concepts. Single-shot
    "trust the JSON" extraction is not a safe default for this domain.

    Adds a "_grounding" key to the result: {field: {"grounded": bool, "span":
    str|None}}. An ungrounded field is NOT dropped from the result -- it's
    still returned as extracted -- but is flagged grounded=False so callers
    can decide whether to trust it (e.g. skip it, flag it for manual review,
    or log it). This catches the class of hallucination where the model
    invents a value with no textual basis in the document at all, which
    schema-only prompting (require_citations=False, the original behavior)
    can't catch.

    KNOWN LIMITATION, confirmed by testing: grounding only verifies the
    cited SPAN is real text copied from the document -- it does not verify
    that a COMPUTED/DERIVED value is actually correct given that span. E.g.
    asking for a "price_band_midpoint" (an arithmetic mean the model must
    calculate, not a literal quote) got grounded=True because the model
    correctly cited the real "Rs. 210 to Rs. 222" span -- but nothing here
    checks that 216 is actually the midpoint of 210 and 222. For
    computed/derived fields, add a separate correctness check (e.g.
    recompute simple arithmetic yourself from the grounded raw values rather
    than trusting the model's derived one).

    derived (optional) closes that gap for fields you can recompute:
    ``{field: (input_field_names, function)}``, for example
    ``{"mid": (["low", "high"], lambda lo, hi: (lo + hi) / 2)}``. See
    check_derived(). It needs require_citations=True.

    Falls back to *fallback_provider* if the primary provider's call or JSON
    parse fails, then raises if that fails too -- callers should catch and
    degrade gracefully (e.g. skip the field, don't crash a whole report).
    """
    if derived and not require_citations:
        raise ValueError("derived checks need require_citations=True")
    prompt = build_prompt(text, schema, instructions, require_citations)
    providers = [provider] + ([fallback_provider] if fallback_provider else [])
    last_error: Optional[Exception] = None
    for p in providers:
        try:
            client, model = _client_and_model(p)
            parsed = _extract_json(_complete(client, model, prompt))
            if not require_citations:
                return parsed
            return check_derived(ground_fields(parsed, schema, text), derived or {})
        except Exception as exc:  # any provider/parse/grounding failure -> next provider
            logger.warning("extract_structured: provider %s failed: %s", p, exc)
            last_error = exc
    raise RuntimeError(f"extract_structured failed on all providers {providers}: {last_error}")


def build_prompt(text: str, schema: dict, instructions: str, require_citations: bool) -> str:
    """Render the extraction prompt (pure; no I/O).

    With *require_citations* the model is asked for ``{"fields", "citations"}``;
    otherwise for the flat field dict alone.
    """
    schema_lines = "\n".join(f'  "{k}": {v}' for k, v in schema.items())
    if require_citations:
        return (
            "Extract the following fields from the document text below. "
            "Respond with ONLY a single JSON object with exactly two top-level keys, "
            '"fields" and "citations" -- no prose, no markdown fences.\n\n'
            f'"fields" must have these keys:\n{schema_lines}\n'
            "Use null for any field not present in the text.\n\n"
            '"citations" must have the SAME keys as "fields". For each non-null field, '
            "the value must be the exact verbatim substring of the document text below "
            "that supports the extracted value (copy it exactly, do not paraphrase or "
            "summarize). For null fields, use null.\n\n"
            f"{instructions}\n\n"
            f"Document text:\n{text}"
        )
    return (
        "Extract the following fields from the document text below. "
        "Respond with ONLY a single JSON object -- no prose, no markdown fences. "
        "Use null for any field not present in the text.\n\n"
        f"Fields:\n{schema_lines}\n\n"
        f"{instructions}\n\n"
        f"Document text:\n{text}"
    )


def ground_fields(parsed: dict, schema: dict, text: str) -> dict:
    """Attach ``_grounding`` to the model's ``{"fields", "citations"}`` reply (pure).

    For every schema key: a null value counts as grounded; otherwise the cited
    span (str, or list of str for array fields) must appear verbatim in *text*.
    Ungrounded values are kept, only flagged.
    """
    fields = parsed.get("fields", {})
    citations = parsed.get("citations", {})
    grounding: dict = {}
    for key in schema:
        value = fields.get(key)
        span = citations.get(key)
        grounding[key] = {
            "grounded": (value is None) or _verify_citation(span, text),
            "span": span,
        }
    result = dict(fields)
    result["_grounding"] = grounding
    return result


def _as_number(value: object) -> object:
    """"15,230.75" -> 15230.75; anything that is not a plain number is returned unchanged."""
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return value
    return value


def check_derived(result: dict, derived: dict) -> dict:
    """Recompute calculated fields and record ``derived_ok`` in ``_grounding`` (pure).

    *derived* maps a field to ``(input_field_names, function)``. ``derived_ok`` is
    True if the recomputed value matches the model's, False if it differs, and
    None if it cannot be checked (an input is missing or ungrounded, or the
    function fails). Values themselves are never changed. This proves the
    arithmetic, not that the inputs were read from the right place.
    """
    grounding = result.get("_grounding", {})
    for field, (inputs, func) in derived.items():
        if field not in grounding:
            raise ValueError(f"derived field {field!r} is not in the schema")
        verdict = None
        usable = all(
            result.get(name) is not None and grounding.get(name, {}).get("grounded")
            for name in inputs
        )
        if usable:
            try:
                expected = func(*(_as_number(result[name]) for name in inputs))
                actual = _as_number(result.get(field))
                if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                    verdict = math.isclose(expected, actual, rel_tol=1e-9, abs_tol=0.005)
                else:
                    verdict = expected == actual
            except (ArithmeticError, TypeError, ValueError):
                verdict = None
        grounding[field]["derived_ok"] = verdict
    return result


def _complete(client: Any, model: str, prompt: str) -> str:
    """Single chat completion at temperature 0; returns the raw message text."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    return resp.choices[0].message.content


DEMO_DRHP_TEXT = (
    "Vantage Cold Chain Logistics Limited proposes to raise up to "
    "Rs. 850 crore through a fresh issue of equity shares, with a "
    "price band of Rs. 210 to Rs. 222 per equity share of face value "
    "Rs. 10 each. The equity shares are proposed to be listed on both "
    "BSE and NSE. The issue opens on 14 August and closes on 18 August."
)
DEMO_DRHP_SCHEMA = {
    "company_name": "string",
    "issue_size_cr": "number, total issue size in INR crore",
    "price_band_low": "number",
    "price_band_high": "number",
    "listing_exchange": "string, e.g. NSE/BSE/both",
}
DEMO_PORTFOLIO_TEXT = (
    "Holdings as of statement date: 120 shares of RELIANCE INDUSTRIES LTD "
    "purchased at an average price of Rs 2,410.50; 40 shares of TCS at "
    "average cost Rs 3,550.00; cash balance Rs 15,230.75."
)
DEMO_PORTFOLIO_SCHEMA = {
    "holdings": "array of {symbol, quantity, avg_price}",
    "cash_balance": "number",
}
DEMOS = {
    "drhp": (DEMO_DRHP_TEXT, DEMO_DRHP_SCHEMA),
    "portfolio": (DEMO_PORTFOLIO_TEXT, DEMO_PORTFOLIO_SCHEMA),
}


def main(argv: Optional[list] = None) -> None:
    """CLI: run one of the built-in demos against the configured providers."""
    ap = argparse.ArgumentParser(description="LLM structured extraction demo")
    ap.add_argument("--demo", choices=sorted(DEMOS), default="drhp")
    args = ap.parse_args(argv)
    sample_text, schema = DEMOS[args.demo]
    result = extract_structured(sample_text, schema)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
