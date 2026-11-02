# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## Layout and commands

- `llm_extract.py` is the whole library: `extract_structured()` orchestrates `build_prompt` → `_client_and_model`
  (`PROVIDERS` registry) → `_complete` → `_extract_json` → `ground_fields` / `_verify_citation`.
  CLI: `python llm_extract.py --demo {drhp,portfolio}` (needs `GROQ_API_KEY` or `GEMINI_API_KEY`).
- Tests: `pytest -q` (all LLM calls stubbed, no network). Lint: `ruff check .`. Install: `pip install -e ".[dev]"`.
- Keep `_verify_citation`'s list branch — it is the bug fix this repo exists for; README and tests document it.
- Never commit API keys; they come from `~/.env.local` or the environment.

<!-- source-fetch-toolkit:v1 (managed by ~/bin/add_source_fetch_section.py — edit there, not here) -->
## Source fetching for data collection & literature surveys

When a task needs a primary document (statute, regulator data, government release, paper, filing),
fetch the document itself — never cite a search snippet as if it had been read.

- **On the owner's Mac:** `fetch_source.py "<url | doi:… | celex:… | arxiv:…>" --out ./sources`
  (in `~/bin`; full guide `~/bin/SOURCE_FETCH_TOOLKIT.md`). Scheduled/unattended runs use only
  `bash /Users/umashankar/bin/fetch_source.sh "<target>"` (allowlisted) and then Read the `.txt` it reports.
- **Elsewhere (cloud sessions, CI):** follow the same ladder by hand — direct GET with a browser User-Agent →
  headless browser only if the page needs JavaScript → official alternates (DOI → OpenAlex
  `api.openalex.org/works/doi:<doi>` locations → repository copy; EUR-Lex → EU Publications Office
  `publications.europa.eu/resource/celex/<CELEX>`; arXiv pdf/abs) → Wayback Machine `id_` snapshot.
- **Hard rule:** a CAPTCHA, robot check, Cloudflare/Akamai/WAF challenge or login wall means stop on that
  host. Do not bypass it (no TLS impersonation, stealth browsers, identity rotation or challenge solving);
  look for the same document via another route, or report it as unavailable.
- **Record** for every figure: URL read, title/date, page/table, fetch route; mark abstract-only or
  snippet-only sources. Fetch politely (~1 request/second per host, no bulk crawling).
<!-- /source-fetch-toolkit -->
