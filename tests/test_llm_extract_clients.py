"""Client construction, env loading, _complete and grounding corner cases (no network)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import llm_extract

TEXT = "Alpha Corp raised Rs. 850 crore.   Listing on NSE and BSE."


@pytest.fixture
def fake_openai(monkeypatch):
    """Replace the `openai` package with a recorder; returns the list of constructor calls."""
    made = []

    class OpenAI:
        def __init__(self, api_key, base_url):
            made.append({"api_key": api_key, "base_url": base_url})

    mod = types.ModuleType("openai")
    mod.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    return made


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No ambient credentials; ~/.env.local resolves into tmp_path."""
    for var in ("GROQ_API_KEY", "GROQ_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


# ---- _client_and_model -------------------------------------------------------


def test_groq_defaults_from_environment(fake_openai, clean_env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k-groq")
    client, model = llm_extract._client_and_model("groq")
    assert model == "llama-3.3-70b-versatile"
    assert fake_openai == [{"api_key": "k-groq", "base_url": "https://api.groq.com/openai/v1"}]
    assert client.__class__.__name__ == "OpenAI"


def test_gemini_uses_its_own_key_url_and_model_override(fake_openai, clean_env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k-gem")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom")
    _, model = llm_extract._client_and_model("gemini")
    assert model == "gemini-custom"
    assert fake_openai == [
        {"api_key": "k-gem", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"}
    ]


def test_env_local_supplies_credentials_and_process_env_overrides_it(
    fake_openai, clean_env, monkeypatch
):
    (clean_env / ".env.local").write_text("GROQ_API_KEY=from-file\nGROQ_MODEL=file-model\n")
    _, model = llm_extract._client_and_model("groq")
    assert fake_openai[-1]["api_key"] == "from-file" and model == "file-model"
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    _, model = llm_extract._client_and_model("groq")
    assert (
        fake_openai[-1]["api_key"] == "from-env" and model == "file-model"
    )  # only the key was overridden


def test_missing_key_names_the_variable_and_never_builds_a_client(fake_openai, clean_env):
    with pytest.raises(RuntimeError, match="GROQ_API_KEY not found in ~/.env.local or environment"):
        llm_extract._client_and_model("groq")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY not found"):
        llm_extract._client_and_model("gemini")
    assert fake_openai == []


def test_empty_key_counts_as_missing(fake_openai, clean_env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY not found"):
        llm_extract._client_and_model("groq")


def test_unknown_provider_is_rejected_before_reading_any_credentials(monkeypatch):
    monkeypatch.setattr(
        llm_extract, "_load_env_local", lambda *a, **k: pytest.fail("must not read env")
    )
    with pytest.raises(ValueError, match="unknown provider 'mistral'"):
        llm_extract._client_and_model("mistral")


def test_provider_registry_shape():
    assert set(llm_extract.PROVIDERS) == {"groq", "gemini"}
    for spec in llm_extract.PROVIDERS.values():
        assert set(spec) == {"key_var", "model_var", "default_model", "base_url"}
    assert (
        llm_extract.DEFAULT_PROVIDER,
        llm_extract.DEFAULT_FALLBACK_PROVIDER,
        llm_extract.TEMPERATURE,
    ) == ("groq", "gemini", 0)


# ---- _load_env_local edge cases ----------------------------------------------


def parse(tmp_path, body):
    p = tmp_path / "env"
    p.write_text(body)
    return llm_extract._load_env_local(p)


def test_env_value_may_contain_equals_signs(tmp_path):
    assert parse(tmp_path, "URL=https://x.test/?a=1&b=2\n") == {"URL": "https://x.test/?a=1&b=2"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('K="abc"', "abc"),
        ("K='abc'", "abc"),
        ("K=\"abc'", "\"abc'"),
        ('K="', '"'),
        ("K=''", ""),
        ("K=  spaced  ", "spaced"),
        ("K=", ""),
    ],
)
def test_env_quote_stripping_rules(tmp_path, raw, expected):
    assert parse(tmp_path, raw + "\n")["K"] == expected


def test_env_export_prefix_whitespace_and_junk_lines(tmp_path):
    body = "  export   A = 1 \n# comment\n\nnot a pair\n  B=2\n"
    assert parse(tmp_path, body) == {"A": "1", "B": "2"}


def test_env_default_path_is_home_env_local(clean_env):
    (clean_env / ".env.local").write_text("X=1\n")
    assert llm_extract._load_env_local() == {"X": "1"}


# ---- _complete ---------------------------------------------------------------


def test_complete_sends_one_user_message_at_temperature_zero():
    seen = {}

    def create(**kw):
        seen.update(kw)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="out"))]
        )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    assert llm_extract._complete(client, "m1", "hello") == "out"
    assert seen == {
        "model": "m1",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }


# ---- _extract_json corners ---------------------------------------------------


def test_extract_json_fence_with_language_tag_and_without():
    assert llm_extract._extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm_extract._extract_json('```\n{"a": 2}\n```') == {"a": 2}


def test_extract_json_prefers_fenced_block_over_earlier_braces():
    assert llm_extract._extract_json('see {x} then ```json\n{"a": 3}\n``` done') == {"a": 3}


def test_extract_json_error_message_truncates_the_response_to_200_chars():
    with pytest.raises(ValueError) as exc:
        llm_extract._extract_json("no braces here " * 40)
    assert "no JSON object found in model response" in str(exc.value)
    assert len(str(exc.value)) < 300


def test_extract_json_truncated_object_raises_decode_error():
    with pytest.raises(json.JSONDecodeError):
        llm_extract._extract_json('{"a": 1, "b": ')


# ---- grounding corners -------------------------------------------------------

SCHEMA = {"a": "x", "b": "y", "c": "z"}


def test_ground_fields_null_value_is_grounded_even_with_a_bogus_span():
    out = llm_extract.ground_fields(
        {"fields": {"a": None}, "citations": {"a": "not in text"}}, {"a": "x"}, TEXT
    )
    assert out["_grounding"]["a"] == {"grounded": True, "span": "not in text"}


def test_ground_fields_non_null_value_without_span_is_ungrounded():
    out = llm_extract.ground_fields({"fields": {"a": 850}, "citations": {}}, {"a": "x"}, TEXT)
    assert out["a"] == 850 and out["_grounding"]["a"] == {"grounded": False, "span": None}


def test_ground_fields_list_span_all_or_nothing_and_whitespace_tolerant():
    ok = llm_extract.ground_fields(
        {"fields": {"a": [1, 2]}, "citations": {"a": ["Rs. 850   crore", "NSE and BSE"]}},
        {"a": "x"},
        TEXT,
    )
    assert ok["_grounding"]["a"]["grounded"] is True
    bad = llm_extract.ground_fields(
        {"fields": {"a": [1, 2]}, "citations": {"a": ["Rs. 850 crore", "invented"]}},
        {"a": "x"},
        TEXT,
    )
    assert bad["_grounding"]["a"]["grounded"] is False
    empty = llm_extract.ground_fields(
        {"fields": {"a": [1]}, "citations": {"a": []}}, {"a": "x"}, TEXT
    )
    assert empty["_grounding"]["a"]["grounded"] is False


def test_ground_fields_only_schema_keys_are_graded_but_extra_fields_are_kept():
    parsed = {
        "fields": {"a": 1, "extra": 2},
        "citations": {"a": "Alpha Corp", "extra": "Alpha Corp"},
    }
    out = llm_extract.ground_fields(parsed, {"a": "x"}, TEXT)
    assert out["extra"] == 2 and set(out["_grounding"]) == {"a"}


def test_ground_fields_does_not_mutate_its_input():
    parsed = {"fields": {"a": 1}, "citations": {"a": "Alpha Corp"}}
    llm_extract.ground_fields(parsed, {"a": "x"}, TEXT)
    assert parsed == {"fields": {"a": 1}, "citations": {"a": "Alpha Corp"}}


@pytest.mark.parametrize(
    "span,expected",
    [
        ("alpha corp", False),  # matching is case-sensitive
        ("Alpha  Corp", True),
        ("\nAlpha\tCorp\n", True),
        ("Alpha", True),
        ("  ", False),
        ("", False),
        (None, False),
        (5, False),
        ({"x": 1}, False),
        ([], False),
        ([""], False),
        (["Alpha", ""], False),
        ([["Alpha"]], True),
    ],
)
def test_verify_citation_matrix(span, expected):
    assert llm_extract._verify_citation(span, TEXT) is expected


# ---- build_prompt specifics --------------------------------------------------


def test_build_prompt_lists_schema_lines_in_order_and_embeds_text_last():
    p = llm_extract.build_prompt("DOC", {"z": "last?", "a": "first?"}, "Be careful.", True)
    assert p.index('  "z": last?') < p.index('  "a": first?')
    assert "Be careful." in p and p.rstrip().endswith("Document text:\nDOC")
    assert '"fields" and "citations"' in p and "verbatim substring" in p


def test_build_prompt_without_citations_has_no_citation_instructions():
    p = llm_extract.build_prompt("DOC", {"a": "x"}, "", False)
    assert (
        "citations" not in p
        and 'Fields:\n  "a": x' in p
        and p.rstrip().endswith("Document text:\nDOC")
    )


# ---- extract_structured wiring -----------------------------------------------


class Client:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, model, messages, temperature):
        self.calls.append((model, messages[0]["content"], temperature))
        if isinstance(self.reply, Exception):
            raise self.reply
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self.reply))]
        )


def test_instructions_and_text_reach_the_model_prompt(monkeypatch):
    c = Client(json.dumps({"fields": {"a": 1}, "citations": {"a": "Alpha Corp"}}))
    monkeypatch.setattr(llm_extract, "_client_and_model", lambda p: (c, f"{p}-m"))
    out = llm_extract.extract_structured(
        TEXT, {"a": "x"}, instructions="USE INR", fallback_provider=None
    )
    assert out["a"] == 1 and out["_grounding"]["a"]["grounded"] is True
    model, prompt, temp = c.calls[0]
    assert model == "groq-m" and temp == 0 and "USE INR" in prompt and TEXT in prompt


def test_fallback_provider_is_used_only_after_the_primary_fails(monkeypatch):
    order = []
    good = Client(json.dumps({"fields": {"a": 1}, "citations": {"a": "Alpha Corp"}}))
    bad = Client(RuntimeError("429"))

    def pick(p):
        order.append(p)
        return (bad if p == "groq" else good), f"{p}-m"

    monkeypatch.setattr(llm_extract, "_client_and_model", pick)
    out = llm_extract.extract_structured(
        TEXT, {"a": "x"}, provider="groq", fallback_provider="gemini"
    )
    assert order == ["groq", "gemini"] and out["a"] == 1
    order.clear()
    llm_extract.extract_structured(TEXT, {"a": "x"}, provider="gemini", fallback_provider="groq")
    assert order == ["gemini"]  # primary succeeded: fallback untouched


def test_final_error_lists_providers_and_the_last_failure(monkeypatch):
    monkeypatch.setattr(
        llm_extract, "_client_and_model", lambda p: (Client(RuntimeError(f"{p} down")), "m")
    )
    with pytest.raises(RuntimeError) as exc:
        llm_extract.extract_structured(TEXT, {"a": "x"})
    assert "['groq', 'gemini']" in str(exc.value) and str(exc.value).endswith("gemini down")


def test_a_reply_that_is_not_valid_json_triggers_the_fallback(monkeypatch):
    clients = {"groq": Client("sorry, I cannot"), "gemini": Client('{"a": 5}')}
    monkeypatch.setattr(llm_extract, "_client_and_model", lambda p: (clients[p], "m"))
    assert llm_extract.extract_structured(TEXT, {"a": "x"}, require_citations=False) == {"a": 5}


def test_main_without_argv_uses_sys_argv_and_default_demo(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(
        llm_extract, "extract_structured", lambda t, s: seen.update(t=t, s=s) or {"ok": True}
    )
    monkeypatch.setattr(sys, "argv", ["llm_extract.py"])
    llm_extract.main()
    assert seen["t"] == llm_extract.DEMO_DRHP_TEXT and seen["s"] is llm_extract.DEMO_DRHP_SCHEMA
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    monkeypatch.setattr(sys, "argv", ["llm_extract.py", "--demo", "portfolio"])
    llm_extract.main()
    assert seen["s"] is llm_extract.DEMO_PORTFOLIO_SCHEMA


def test_demos_registry_is_consistent():
    assert set(llm_extract.DEMOS) == {"drhp", "portfolio"}
    for text, schema in llm_extract.DEMOS.values():
        assert isinstance(text, str) and text and isinstance(schema, dict) and schema


def test_extract_json_multiline_fenced_block_wins_over_brace_noise_before_it():
    reply = 'Note {this is not json}\n```json\n{\n  "a": 1,\n  "b": {"c": 2}\n}\n```\ntrailing {x}'
    assert llm_extract._extract_json(reply) == {"a": 1, "b": {"c": 2}}


def test_extract_json_takes_the_first_of_several_fenced_blocks():
    reply = '```json\n{"a": 1}\n```\nand\n```json\n{"a": 2}\n```'
    assert llm_extract._extract_json(reply) == {"a": 1}


def test_complete_returns_the_first_choice():
    def create(**kw):
        choices = [
            types.SimpleNamespace(message=types.SimpleNamespace(content="first")),
            types.SimpleNamespace(message=types.SimpleNamespace(content="second")),
        ]
        return types.SimpleNamespace(choices=choices)

    completions = types.SimpleNamespace(create=create)
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    assert llm_extract._complete(client, "m", "p") == "first"


def test_main_prints_indented_json(monkeypatch, capsys):
    monkeypatch.setattr(llm_extract, "extract_structured", lambda t, s: {"a": 1, "b": [2]})
    llm_extract.main(["--demo", "drhp"])
    assert capsys.readouterr().out == json.dumps({"a": 1, "b": [2]}, indent=2) + "\n"
