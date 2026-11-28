# llm-extract-citation-fix

<!-- how-to-use-with-ai -->
> **New here?** See [How to use this repo with AI (Claude, Gemini, ChatGPT)](HOW_TO_USE_WITH_AI.md) — clone it, paste it into an AI tool, and ask in plain English.

A single-module Python helper, `extract_structured()`, that turns unstructured
text (a SEBI DRHP filing, a broker portfolio statement) into a flat JSON
object matching a caller-supplied field schema, using an OpenAI-compatible LLM
(Groq by default, Gemini as fallback). Every extracted value is
*citation-grounded*: the model must also return the verbatim source span
backing it, and the span is checked against the document before the value is
trusted. The repo exists to publish one bug fix to that grounding check
(array-typed fields were always flagged ungrounded) together with the tests
that pin it down.

> Status: maintained snapshot. `llm_extract.py` is copied from a private
> multi-market research platform; the extraction technique follows patents
> MY269019 and MY269128 (IIIT-Hyderabad). Educational use only; not investment
> advice.

## In plain English

*This section is for readers who do not write software.*

**What problem does it solve?** AI models are good at pulling facts out of long documents, for example the
price range in a share-sale prospectus, but they sometimes state things the document never said. This tool
asks the AI to **show its evidence**: for each fact, the exact words copied from the document. The program
then checks that those words really are there.

**An example**

| Step | What happens |
|---|---|
| The document says | "...with a price band of Rs. 210 to Rs. 222 per equity share..." |
| The AI reports | lowest price = 210, evidence = "Rs. 210 to Rs. 222" |
| The program checks | Is that exact phrase in the document? Yes, so it is marked **grounded** |
| If the phrase were not there | The value is still shown, but marked **not grounded** for a person to check |

The evidence must also cover **whole words and whole numbers**. If the document says "Rs 15,230.75",
evidence reading "Rs 1" or "Rs 15" is rejected, even though those characters appear inside the longer number.

**What "grounded" does and does not mean**
- It **does** mean the quoted words exist in the document.
- It does **not** mean the answer is right. The AI could quote a real sentence and misread it, link it to the
  wrong item, or get a calculation wrong. For example, it could quote a real price range and report the wrong midpoint.
- So treat "not grounded" as a warning, and "grounded" as "worth checking", not as proof.

**Where does your document go?** Its text is sent to an outside AI service (Groq, or Google's Gemini as the
backup). Do not use confidential or personal documents unless you are allowed to share them with those services.

**What you need:** Python and an access key for one of those services. Their prices and free limits are set by
them. This is a building block for programmers, not an app with buttons.

## What it does
- Builds a schema-driven extraction prompt and returns the model's JSON object (`extract_structured`).
- Asks for and verifies a verbatim citation per field, attaching `_grounding: {field: {grounded, span}}` (`ground_fields`, `_verify_citation`).
- Accepts a *list* of per-item citations for array-typed fields — the bug fix this repo is named for.
- Tolerates model replies wrapped in prose or ```` ```json ```` fences (`_extract_json`).
- Falls back from the primary provider to a second one on any call/parse failure, logging the reason.
- Reads API keys from `~/.env.local` or the environment without a `python-dotenv` dependency.

## Quick start
```bash
git clone https://github.com/herrrickshaw/llm-extract-citation-fix.git
cd llm-extract-citation-fix
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export GROQ_API_KEY=...        # or put it in ~/.env.local
python llm_extract.py --demo portfolio
```

Library use:

```python
from llm_extract import extract_structured

schema = {
    "company_name": "string",
    "issue_size_cr": "number, issue size in INR crore",
    "price_band_low": "number",
    "price_band_high": "number",
    "listing_exchange": "string, e.g. NSE/BSE/both",
}
result = extract_structured(drhp_text, schema)
# result["_grounding"]["issue_size_cr"] -> {"grounded": True, "span": "Rs. 850 crore"}
```

## The bug fix

`_verify_citation()` originally accepted only a single string span. For an
array-typed field — e.g. `holdings: "array of {symbol, quantity, avg_price}"`
— the model correctly returns *one citation per item* (a list), which the old
code rejected outright, silently marking a perfectly grounded extraction as
`grounded: false`.

```diff
 def _verify_citation(span: object, source_text: str) -> bool:
+    if isinstance(span, list):
+        return bool(span) and all(_verify_citation(item, source_text) for item in span)
     if not isinstance(span, str) or not span.strip():
         return False
     return _normalize_whitespace(span) in _normalize_whitespace(source_text)
```

(The final line shown is as it was at the time of that fix; matching has since been tightened to whole words and numbers.)

It was found via a [promptfoo](https://www.promptfoo.dev/) pilot against this
function's real test cases (a SEBI DRHP filing and a portfolio holdings
statement): the array-holdings case failed on `_grounding.holdings` even though
every extracted value was correct and verbatim from the source. The pilot
config, plus a second bug it caught (two deprecated model names silently
failing on every call), is documented in
[prompt-efficiency-tracker](https://github.com/herrrickshaw/prompt-efficiency-tracker).

## How it works

| Stage | Function | Output |
|---|---|---|
| Render prompt from schema + text | `build_prompt` | prompt string (pure) |
| Pick provider, load key/model | `_client_and_model` via `PROVIDERS` registry | OpenAI-compatible client |
| Call the model at temperature 0 | `_complete` | raw reply text |
| Pull the JSON object out of the reply | `_extract_json` | `dict` |
| Verify each citation against the source | `ground_fields` → `_verify_citation` | result dict + `_grounding` |

Design notes:
- Providers are rows in a `PROVIDERS` dict (key env var, model env var, default model, base URL); adding one needs no caller change.
- Prompt rendering and grounding are pure functions so they are unit-tested without a network.
- Any provider, parse or grounding failure moves on to the fallback provider; if all fail, a `RuntimeError` carries the last error. Callers are expected to catch it and degrade (skip the field) rather than crash a report.
- Ungrounded values are **kept** in the result and only flagged, so the caller decides whether to drop, review or log them.

## Repository layout
| Path | Purpose |
|---|---|
| `llm_extract.py` | The whole library + `--demo {drhp,portfolio}` CLI |
| `tests/` | pytest suite; all LLM calls stubbed, no network |
| `pyproject.toml` | dependencies, ruff and pytest config |
| `.github/workflows/ci.yml` | ruff + pytest on push/PR, Python 3.9 and 3.12 |
| `HOW_TO_USE_WITH_AI.md` | plain-language guide for non-programmers |

## Configuration
Read from `~/.env.local` (simple `KEY=VALUE` lines) overlaid by the process environment:

| Variable | Default | Used by |
|---|---|---|
| `GROQ_API_KEY` | required for `groq` | primary provider |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | primary provider |
| `GEMINI_API_KEY` | required for `gemini` | fallback provider |
| `GEMINI_MODEL` | `gemini-2.0-flash` | fallback provider |

Keys are never committed; `.env.local` lives in the home directory, outside the repo.

## Testing
```bash
pytest -q        # 128 tests, no network
ruff check .
```

## Limitations and known issues
- Grounding only proves the cited span is real text from the document. A *derived* value (a midpoint, a total) is checked only if you give the formula:
  `extract_structured(text, schema, derived={"price_band_midpoint": (["price_band_low", "price_band_high"], lambda lo, hi: (lo + hi) / 2)})`.
  Each such field gets `_grounding[field]["derived_ok"]`: `True` (matches), `False` (differs) or `None` (an input is missing or ungrounded, or the formula failed). It checks the arithmetic, not whether the inputs were read from the right place. Values are never changed.
- A citation must start and end on whole words and numbers (`Rs 1` no longer matches `Rs 15,230.75`). It can still match a real but irrelevant phrase elsewhere in the document. Text in scripts written without spaces between words may be rejected when quoted mid-sentence.
- The schema is a plain field → description map used verbatim in the prompt; it is not validated as JSON Schema.
- The default model names may be deprecated by the providers over time (the promptfoo pilot caught exactly this). Override via `GROQ_MODEL` / `GEMINI_MODEL`.
- The literature this guards against is sobering: on FinanceBench, GPT-4-Turbo with retrieval got 81% of questions wrong; FinTagging found the best model only 17% accurate at linking facts to XBRL concepts. Single-shot "trust the JSON" extraction is not a safe default for financial documents.

## Related repositories
- [prompt-efficiency-tracker](https://github.com/herrrickshaw/prompt-efficiency-tracker) — the promptfoo pilot that found this bug.

## Licence
No licence file yet — all rights reserved.
