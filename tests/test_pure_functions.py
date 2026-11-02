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
