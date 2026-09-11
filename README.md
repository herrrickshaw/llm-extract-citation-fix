# llm-extract-citation-fix

`extract_structured()` (structured-JSON extraction from unstructured text via
an LLM, with citation-grounding) with one bug fixed: `_verify_citation()`
only accepted a single string span. For an array-typed field — e.g.
`holdings: "array of {symbol, quantity, avg_price}"` — the model correctly
returns *one citation per item* (a list), which the old code rejected
outright, silently marking a perfectly-grounded extraction as
`grounded: false`.

Found via a [promptfoo](https://www.promptfoo.dev/) pilot against this
function's real test cases (a SEBI DRHP filing and a portfolio holdings
statement) — the array-holdings test case failed on `_grounding.holdings`
even though every extracted value was correct and verbatim from the source.

## The fix

```diff
 def _verify_citation(span: object, source_text: str) -> bool:
-    """True if *span* appears verbatim (allowing whitespace differences) in source_text."""
+    """True if *span* appears verbatim (allowing whitespace differences) in
+    source_text. Array-typed fields (e.g. a list of holdings) legitimately
+    come back as a list of per-item citations rather than one string --
+    found via the promptfoo pilot, where a correctly-extracted `holdings`
+    array was silently flagged grounded=False because this only accepted
+    str. Every element must verify for the field to count as grounded."""
+    if isinstance(span, list):
+        return bool(span) and all(_verify_citation(item, source_text) for item in span)
     if not isinstance(span, str) or not span.strip():
         return False
     return _normalize_whitespace(span) in _normalize_whitespace(source_text)
```

`llm_extract.py` here is the fixed version, copied from its source repo
(a private multi-market research platform) with only this bug — nothing
else in the file changed.

The promptfoo pilot config that caught this, plus a second bug it also
caught (two deprecated model names silently failing on every call), is
documented in [prompt-efficiency-tracker](https://github.com/herrrickshaw/prompt-efficiency-tracker).
