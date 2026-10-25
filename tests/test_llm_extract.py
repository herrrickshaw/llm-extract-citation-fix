"""Tests for llm_extract.py. The LLM is mocked; no network, no API keys."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm_extract as le  # noqa: E402

DOC = (
    "Vantage Cold Chain Logistics Limited proposes to raise up to Rs. 850 crore "
    "through a fresh issue, with a price band of Rs. 210 to Rs. 222 per share."
)
SCHEMA = {"company_name": "string", "issue_size_cr": "number", "listing_exchange": "string"}


class FakeClient:
    """Mimics openai.OpenAI().chat.completions.create, returning canned text."""

    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


@pytest.fixture
def providers(monkeypatch):
    clients = {}

    def fake(provider):
        if provider not in clients:
            raise RuntimeError(f"no key for {provider}")
        return clients[provider], f"{provider}-model"

    monkeypatch.setattr(le, "_client_and_model", fake)
    return clients


# ---- _extract_json ---------------------------------------------------------

def test_extract_json_plain_object():
    assert le._extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_nested_object():
    txt = 'Sure:\n```json\n{"fields": {"a": 1}, "citations": {"a": "x"}}\n```\nDone.'
    assert le._extract_json(txt) == {"fields": {"a": 1}, "citations": {"a": "x"}}


def test_extract_json_object_surrounded_by_prose():
    assert le._extract_json('Here you go: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}


def test_extract_json_first_object_when_prose_after_has_braces():
    """Docstring: 'Pull the first {...} JSON object'. A trailing remark that itself
    contains braces must not break parsing of the first object."""
    txt = '{"a": 1}\nNote: fields I could not find are set to null {as requested}.'
    assert le._extract_json(txt) == {"a": 1}


def test_extract_json_no_object_raises():
    with pytest.raises(ValueError):
        le._extract_json("I cannot help with that.")


# ---- _verify_citation ------------------------------------------------------

@pytest.mark.parametrize("span,ok", [
    ("Rs. 850 crore", True),
    ("Rs.   850\ncrore", True),            # whitespace differences allowed
    ("Rs. 900 crore", False),              # not in text
    ("", False), ("   ", False), (None, False), (850, False),
    (["Rs. 210", "Rs. 222"], True),        # array fields: every element must verify
    (["Rs. 210", "Rs. 999"], False),
    ([], False),
])
def test_verify_citation(span, ok):
    assert le._verify_citation(span, DOC) is ok


# ---- extract_structured ----------------------------------------------------

def test_grounding_flags_hallucinated_span(providers):
    reply = {
        "fields": {"company_name": "Vantage Cold Chain Logistics Limited",
                   "issue_size_cr": 850, "listing_exchange": "NSE"},
        "citations": {"company_name": "Vantage Cold Chain Logistics Limited",
                      "issue_size_cr": "Rs. 850 crore", "listing_exchange": "listed on NSE"},
    }
    providers["groq"] = FakeClient(json.dumps(reply))
    out = le.extract_structured(DOC, SCHEMA)
    g = out["_grounding"]
    assert g["company_name"]["grounded"] and g["issue_size_cr"]["grounded"]
    assert g["listing_exchange"]["grounded"] is False
    assert out["listing_exchange"] == "NSE"          # flagged, not dropped
    assert providers["groq"].calls[0]["temperature"] == 0


def test_null_field_counts_as_grounded(providers):
    reply = {"fields": {"company_name": None}, "citations": {"company_name": None}}
    providers["groq"] = FakeClient(json.dumps(reply))
    out = le.extract_structured(DOC, {"company_name": "string"})
    assert out["_grounding"]["company_name"] == {"grounded": True, "span": None}


def test_without_citations_returns_raw_json(providers):
    providers["groq"] = FakeClient('{"company_name": "X"}')
    out = le.extract_structured(DOC, SCHEMA, require_citations=False)
    assert out == {"company_name": "X"}
    assert "citations" not in providers["groq"].calls[0]["messages"][0]["content"]


def test_falls_back_to_second_provider_on_error(providers):
    providers["groq"] = FakeClient(exc=TimeoutError("boom"))
    providers["gemini"] = FakeClient('{"fields": {}, "citations": {}}')
    out = le.extract_structured(DOC, {"a": "x"})
    assert out["_grounding"]["a"]["grounded"] is True
    assert len(providers["gemini"].calls) == 1


def test_raises_when_all_providers_fail(providers):
    providers["groq"] = FakeClient("not json at all")
    with pytest.raises(RuntimeError, match="all providers"):
        le.extract_structured(DOC, SCHEMA, fallback_provider=None)


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **k: None))
    with pytest.raises(ValueError):
        le._client_and_model("nope")


# ---- _load_env_local -------------------------------------------------------

def test_env_loader_skips_comments_and_blank(monkeypatch, tmp_path):
    (tmp_path / ".env.local").write_text("# c\n\nA=1\nB = two \nnoequals\n")
    monkeypatch.setattr(le.Path, "home", staticmethod(lambda: tmp_path))
    assert le._load_env_local() == {"A": "1", "B": "two"}


def test_env_loader_strips_quotes_and_export(monkeypatch, tmp_path):
    """.env files commonly quote values and prefix `export`; the quotes must not
    end up inside the API key that is sent to the provider."""
    (tmp_path / ".env.local").write_text(
        'GROQ_API_KEY="gsk_test"\nexport GEMINI_API_KEY=\'gem_test\'\n'
    )
    monkeypatch.setattr(le.Path, "home", staticmethod(lambda: tmp_path))
    assert le._load_env_local() == {"GROQ_API_KEY": "gsk_test", "GEMINI_API_KEY": "gem_test"}


def test_env_loader_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(le.Path, "home", staticmethod(lambda: tmp_path))
    assert le._load_env_local() == {}
