"""Unit tests for the pure (no I/O, no LLM) helpers in llm_extract."""

from __future__ import annotations

import pytest

import llm_extract
from llm_extract import _extract_json, _normalize_whitespace, _verify_citation


def test_normalize_whitespace_collapses_runs_and_strips():
    assert _normalize_whitespace("  a \n\t b   c ") == "a b c"


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        'Here you go:\n{"a": 1}\nThanks',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
    ],
)
def test_extract_json_accepts_bare_prose_wrapped_and_fenced(raw):
    assert _extract_json(raw) == {"a": 1}


def test_extract_json_raises_when_no_object():
    with pytest.raises(ValueError, match="no JSON object"):
        _extract_json("nothing here")


SOURCE = "Price band of Rs. 210 to Rs. 222 per share.\n120 shares of RELIANCE; 40 shares of TCS."


def test_verify_citation_exact_and_whitespace_tolerant():
    assert _verify_citation("Rs. 210 to Rs. 222", SOURCE)
    assert _verify_citation("Rs.   210 to\nRs. 222", SOURCE)


def test_verify_citation_rejects_missing_empty_and_non_string():
    assert not _verify_citation("Rs. 999", SOURCE)
    assert not _verify_citation("   ", SOURCE)
    assert not _verify_citation(None, SOURCE)
    assert not _verify_citation(42, SOURCE)


def test_verify_citation_list_requires_every_item_to_match():
    # The bug this repo exists to fix: array-typed fields cite per item.
    assert _verify_citation(["120 shares of RELIANCE", "40 shares of TCS"], SOURCE)
    assert not _verify_citation(["120 shares of RELIANCE", "7 shares of INFY"], SOURCE)
    assert not _verify_citation([], SOURCE)


def test_load_env_local_parses_simple_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text("# comment\nGROQ_API_KEY = abc \n\nBROKEN LINE\nX=y=z\n")
    monkeypatch.setattr(llm_extract.Path, "home", staticmethod(lambda: tmp_path))
    assert llm_extract._load_env_local() == {"GROQ_API_KEY": "abc", "X": "y=z"}


def test_load_env_local_missing_file_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_extract.Path, "home", staticmethod(lambda: tmp_path))
    assert llm_extract._load_env_local() == {}


def test_load_env_local_accepts_explicit_path(tmp_path):
    env_file = tmp_path / "custom.env"
    env_file.write_text("K=v\n")
    assert llm_extract._load_env_local(env_file) == {"K": "v"}


def test_build_prompt_with_and_without_citations():
    schema = {"a": "string", "b": "number"}
    with_cites = llm_extract.build_prompt("DOC", schema, "extra", True)
    without = llm_extract.build_prompt("DOC", schema, "extra", False)
    for prompt in (with_cites, without):
        assert '  "a": string\n  "b": number' in prompt
        assert prompt.endswith("extra\n\nDocument text:\nDOC")
    assert '"citations" must have the SAME keys' in with_cites
    assert "citations" not in without


def test_ground_fields_flags_each_key_and_keeps_values():
    parsed = {
        "fields": {"a": "x", "b": None, "c": 7},
        "citations": {"a": "Rs. 210", "b": None, "c": "nope"},
    }
    out = llm_extract.ground_fields(parsed, {"a": "", "b": "", "c": ""}, SOURCE)
    assert {k: out[k] for k in "abc"} == {"a": "x", "b": None, "c": 7}
    assert out["_grounding"] == {
        "a": {"grounded": True, "span": "Rs. 210"},
        "b": {"grounded": True, "span": None},
        "c": {"grounded": False, "span": "nope"},
    }


def test_ground_fields_tolerates_missing_keys():
    out = llm_extract.ground_fields({}, {"a": ""}, SOURCE)
    assert out == {"_grounding": {"a": {"grounded": True, "span": None}}}
