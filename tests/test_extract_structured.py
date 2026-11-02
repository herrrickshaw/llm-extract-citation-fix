"""Characterisation tests for extract_structured() with the LLM call stubbed out."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import llm_extract

REPO = Path(__file__).resolve().parent.parent
TEXT = (
    "Holdings as of statement date: 120 shares of RELIANCE INDUSTRIES LTD "
    "purchased at an average price of Rs 2,410.50; 40 shares of TCS at "
    "average cost Rs 3,550.00; cash balance Rs 15,230.75."
)
SCHEMA = {"holdings": "array of {symbol, quantity, avg_price}", "cash_balance": "number"}


class FakeClient:
    """Minimal stand-in for openai.OpenAI exposing chat.completions.create."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, temperature):
        self.prompts.append(messages[0]["content"])
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=nxt))])


def _patch_providers(monkeypatch, clients):
    """Map provider name -> FakeClient; records which providers were asked for."""
    asked = []

    def fake(provider):
        asked.append(provider)
        if provider not in clients:
            raise RuntimeError(f"{provider} not configured")
        return clients[provider], f"{provider}-model"

    monkeypatch.setattr(llm_extract, "_client_and_model", fake)
    return asked


def test_grounded_array_field_and_scalar(monkeypatch):
    reply = json.dumps(
        {
            "fields": {
                "holdings": [{"symbol": "RELIANCE", "quantity": 120}, {"symbol": "TCS"}],
                "cash_balance": 15230.75,
            },
            "citations": {
                "holdings": ["120 shares of RELIANCE INDUSTRIES LTD", "40 shares of TCS"],
                "cash_balance": "cash balance Rs 15,230.75",
            },
        }
    )
    client = FakeClient([reply])
    _patch_providers(monkeypatch, {"groq": client})

    out = llm_extract.extract_structured(TEXT, SCHEMA)

    assert out["cash_balance"] == 15230.75
    assert out["_grounding"]["holdings"]["grounded"] is True
    assert out["_grounding"]["cash_balance"]["grounded"] is True
    assert "citations" in client.prompts[0] and TEXT in client.prompts[0]


def test_ungrounded_value_is_kept_but_flagged(monkeypatch):
    reply = json.dumps(
        {"fields": {"holdings": None, "cash_balance": 1}, "citations": {"cash_balance": "Rs 999"}}
    )
    _patch_providers(monkeypatch, {"groq": FakeClient([reply])})

    out = llm_extract.extract_structured(TEXT, SCHEMA)

    assert out["cash_balance"] == 1
    assert out["_grounding"]["cash_balance"] == {"grounded": False, "span": "Rs 999"}
    # null fields count as grounded (nothing to verify)
    assert out["_grounding"]["holdings"] == {"grounded": True, "span": None}


def test_without_citations_returns_parsed_object_verbatim(monkeypatch):
    client = FakeClient(['{"cash_balance": 2}'])
    _patch_providers(monkeypatch, {"groq": client})

    out = llm_extract.extract_structured(TEXT, SCHEMA, require_citations=False)

    assert out == {"cash_balance": 2}
    assert "citations" not in client.prompts[0]


def test_falls_back_to_second_provider_on_bad_json(monkeypatch):
    groq = FakeClient(["not json at all"])
    gemini = FakeClient(['{"cash_balance": 3}'])
    asked = _patch_providers(monkeypatch, {"groq": groq, "gemini": gemini})

    out = llm_extract.extract_structured(TEXT, SCHEMA, require_citations=False)

    assert out == {"cash_balance": 3}
    assert asked == ["groq", "gemini"]


def test_raises_when_all_providers_fail(monkeypatch):
    _patch_providers(monkeypatch, {"groq": FakeClient([RuntimeError("boom")])})

    with pytest.raises(RuntimeError, match="failed on all providers.*gemini not configured"):
        llm_extract.extract_structured(TEXT, SCHEMA)


def test_no_fallback_when_fallback_provider_none(monkeypatch):
    asked = _patch_providers(monkeypatch, {"groq": FakeClient([RuntimeError("boom")])})

    with pytest.raises(RuntimeError):
        llm_extract.extract_structured(TEXT, SCHEMA, fallback_provider=None)
    assert asked == ["groq"]


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        llm_extract._client_and_model("nope")


def test_cli_help_smoke():
    proc = subprocess.run(
        [sys.executable, str(REPO / "llm_extract.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--demo" in proc.stdout
