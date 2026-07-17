# Second Brain — Architecture & Build Plan

Status: concept / phase-1 design. Nothing here is built yet — this is the blueprint to review, adjust, and then implement (phase 2), module by module.

---

## 0. Assumptions I'm making (confirm or correct these before phase 2)

1. **"MM-DD-YYYY" naming applies to the daily source note**, not to every atomic note the pipeline creates. Obsidian can't have two files with the same name in the same folder, so if every distilled note were literally titled `07-16-2026.md` you'd get one atomic note per day, not per topic. My read: your **daily journal note** is `01-Daily/07-16-2026.md` (matches Obsidian's Daily Notes date-format setting), and the **atomic notes distilled from it** get descriptive titles (`Docker Networking Basics.md`) with a `source:: [[07-16-2026]]` frontmatter link back to the day they came from. If you actually want every new note dated in the filename (e.g. `07-16-2026 - Docker Networking.md`), that's a one-line config change — flag it and I'll adjust.
2. **You want a trust ramp, not day-one autonomy.** Anything that renames, merges, or rewrites your own notes runs in **dry-run** mode first (writes proposals to `_staging/`, changes nothing live) until you've reviewed a few nightly reports and are comfortable flipping to auto-apply.
3. **Privacy matters for a personal vault** that includes career, financial, and possibly emotional journaling content — so the default routing keeps content on-device wherever quality allows, and only sends specific text to a cloud API when the task genuinely needs it (this is a knob you control, see §7).

---

## 1. Five architectural decisions, and why

| Decision | What I'd do instead of the "obvious" choice | Why |
|---|---|---|
| **Batch, not live-watch** | No file-watcher, no inotify, no real-time sync | Your requirement is "run by night." Docker Desktop on macOS uses VirtioFS for bind mounts, and while it's improved a lot, it still has documented edge cases with large-file writes and event propagation, especially with editors (like Obsidian) that save via temp-file-then-rename. A nightly scan that hashes every file and diffs against yesterday's manifest sidesteps that whole failure class, is trivially resumable, and is *simpler* — which matters for a system that's allowed to edit your notes unsupervised. |
| **Embedded vector store, not a DB server** | LanceDB (or Chroma) running in-process inside the worker container, backed by a mounted volume | You have one user and, realistically, a few thousand notes. A server-based vector DB (Qdrant, Milvus, Postgres+pgvector) is a second service to run, patch, and lose sleep over for no benefit at this scale. Embedded stores handle personal-vault sizes fine and add zero operational overhead. |
| **Ollama runs natively on macOS, not inside Docker** | `brew install ollama`, runs as its own background service on the host | Docker Desktop on Apple Silicon cannot pass the GPU through to a Linux container — Metal acceleration is a macOS-native-process thing only. Containerized Ollama silently falls back to CPU and runs 3–5x slower. The worker container reaches host Ollama over `http://host.docker.internal:11434`. |
| **Reuse Obsidian's existing AI plugins for the chat agent (§3), don't build a bespoke chat UI** | Copilot for Obsidian (chat) + Smart Connections (semantic linking), pointed at either Claude or local Ollama, loaded with your generated `AGENTS.md`/`alma.md` as system prompt | These plugins already do RAG-over-vault chat well and are actively maintained. Building your own chat interface would be reinventing a solved problem. Your custom code should focus on the part nobody else builds: **unsupervised nightly curation with an audit trail** (§2, §4, §5) — that's the actual novel work here. |
| **Git as the real safety net, the report as the human-readable layer** | `git init` the vault (or at least run automated changes through git), auto-commit before/after each nightly run with a descriptive message | An LLM editing your personal knowledge base unsupervised needs an instant, boring rollback mechanism. The morning report (§6) is for understanding *why* something changed; git is for undoing it if the "why" doesn't hold up. |

---

## 2. Repo / folder layout

```
second-brain/
├── docker-compose.yml
├── .env                          # API keys, vault path — gitignored
├── config.yaml                  # from config.example.yaml
├── worker/                       # phase 2: Dockerfile + Python package
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── src/
│       ├── manifest.py           # hash/mtime scan + diff against last run
│       ├── llm_router.py         # picks provider+model per task, from config.yaml
│       ├── reorganizer.py        # req. 1: titles, grammar, taxonomy, links
│       ├── digestor.py           # req. 2: daily note → atomic notes
│       ├── agents_md_builder.py  # req. 3: vault analysis → AGENTS.md
│       ├── report.py             # req. 5: nightly change report
│       └── jobs/
│           └── nightly_run.py    # orchestrates the above in order
├── data/                         # mounted volume: manifest.sqlite, vector_store/
├── logs/
└── launchd/
    └── com.alejandrogarcia.secondbrain.nightly.plist

<your-obsidian-vault>/
├── 00-Inbox/
├── 01-Daily/                    # 07-16-2026.md, matches Obsidian's date format setting
├── 02-Areas/
├── 03-Projects/
├── 04-Reference/
├── _staging/                    # dry-run proposals live here until approved
├── _reports/                    # Review-07-16-2026.md, one per run
├── AGENTS.md                    # req. 3 — regenerated after each taxonomy pass
└── alma.md                      # req. 4 — hand-written by you, read by every agent
```

The vault itself is bind-mounted read/write into the worker container; `data/` and `logs/` stay outside the vault so they never show up as "notes" inside Obsidian.

---

## 3. Component walkthrough, mapped to your five requirements

### Requirement 1 — reorganize existing notes, fix grammar, check clarity

Nightly (or on first run, once as a backfill):

1. **`manifest.py`** walks the vault, hashes every `.md` file, and diffs against the previous run's manifest (SQLite). Only new/changed files enter the pipeline — this keeps nightly runs fast as the vault grows into the thousands of notes.
2. For each changed note, `reorganizer.py` runs a staged pipeline:
   - **Title check** — if missing, generic ("Untitled"), or misleading relative to the body, propose a better one.
   - **Grammar & clarity pass** — fix genuine errors; don't rewrite your voice. If a note's explanation is incomplete or the concept isn't clearly landed, **flag it, don't auto-expand it** — the pipeline shouldn't invent content you didn't write to make a note "look" finished.
   - **Taxonomy placement** — compare the note's embedding against existing organized notes to suggest a folder; only move a note if confidence is high, otherwise flag for your review.
   - **Link suggestions** — semantic search against the vault's embeddings for related notes, propose `[[wikilinks]]` rather than silently inserting them.
3. Every proposed change is written as a diff to `_staging/` (dry-run) or applied + git-committed (once you trust it), and logged to the day's change manifest with a one-line rationale per change.

### Requirement 2 — digest the daily note into new, linked notes

1. Load `01-Daily/{today MM-DD-YYYY}.md`.
2. Segment into atomic ideas (heading-based split first; LLM-assisted split as a fallback for unstructured stream-of-consciousness entries).
3. For each atomic chunk, embed it and search the vector store:
   - **High similarity to an existing note** → append/merge into that note with a link back to today's daily note, rather than creating a near-duplicate.
   - **No match** → create a new note with a descriptive title in the right folder, frontmatter `source:: [[MM-DD-YYYY]]`, `created:: MM-DD-YYYY`, suggested tags.
4. Update the daily note itself with a short "Notes generated" section linking out to whatever was distilled from it — the daily note becomes a map-of-content for that day, the atomic notes carry the durable knowledge.

### Requirement 3 — `AGENTS.md` for a repo-connected AI agent

Note on naming: you wrote `Agent.md` — I'd actually call it **`AGENTS.md`** (plural). It's an open, Linux-Foundation-stewarded convention that 60,000+ repos now use, natively read by Claude Code (via import), Cursor, Gemini CLI, and 20+ other tools. Using the standard name means any AI tool you point at this repo later picks it up for free, instead of a name only you recognize.

- `agents_md_builder.py` runs after a reorganization pass (once initially, then whenever the taxonomy changes materially — not every night, per the research on this format: files over ~150 lines or rebuilt too often see diminishing returns and can actually *increase* inference cost without helping).
- It inventories: folder taxonomy, tag vocabulary, frontmatter schema, linking conventions — then drafts `AGENTS.md` documenting all of it, plus your tone rules (professional, no filler, cite `[[note]]` not "according to my search", say "the vault doesn't cover this" instead of guessing).
- See `AGENTS.md.template` — this is what gets generated and refined, not hand-maintained from scratch.
- **How it gets used day to day**: point Copilot for Obsidian's system prompt (or Claude Code, if you ever open the vault as a repo) at this file. You don't need to build a chat UI — the existing plugin ecosystem already does vault-RAG chat well; `AGENTS.md` is what makes it *your* agent instead of a generic one.

### Requirement 4 — `alma.md`, the soul document

This one you write, not the pipeline — it's your call what the assistant's purpose, tone, and working relationship with you should be. See `alma.md.template` for a scaffold following the same logic AGENTS.md uses (be specific, give examples, avoid vague adjectives like "friendly" with nothing behind them). The pipeline never edits this file; it only reads it.

### Requirement 5 — nightly run + morning report

`jobs/nightly_run.py` runs the full sequence (manifest diff → reorganize → digest → AGENTS.md rebuild if needed) and finishes by calling `report.py`, which writes `_reports/Review-{MM-DD-YYYY}.md`. Format and materiality logic are in §6.

---

## 4. Docker Compose & host services

The worker is invoked on-demand by launchd (§5) — it's not a long-running service, so `docker compose up` intentionally does nothing.

```yaml
services:
  worker:
    build: ./worker
    image: second-brain-worker:latest
    volumes:
      - ${VAULT_PATH}:/vault
      - ./data:/data
      - ./logs:/logs
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:-}
      - OLLAMA_HOST=http://host.docker.internal:11434
      - VAULT_PATH=/vault
      - DATA_PATH=/data
      - DRY_RUN=${DRY_RUN:-true}
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Host-side (not in Docker):

- **Ollama** — installed via `brew install ollama`, runs as its own background service, serves at `localhost:11434` with Metal acceleration automatically.
- **Obsidian.app** with **Copilot** (chat UI) and **Smart Connections** (semantic linking sidebar) — both configurable to call Ollama locally or a cloud API.

This is `worker/Dockerfile` + the Python package (`manifest.py`, `reorganizer.py`, etc.) — the actual implementation — which I'd build in phase 2 once you've signed off on the routing and folder assumptions above, since getting those wrong means re-doing real code.

---

## 5. Scheduling: launchd, not cron

macOS deprecated cron in favor of launchd years ago, and for this use case launchd is genuinely better, not just "the Apple way": `StartCalendarInterval` jobs run when the Mac wakes from sleep if they were missed, where a cron job would just skip that day entirely on a laptop that was closed at 2am.

See `com.alejandrogarcia.secondbrain.nightly.plist` — runs the full pipeline at 02:00. By the time you're up, `_reports/Review-{today}.md` is sitting in the vault. Load it with:

```
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.alejandrogarcia.secondbrain.nightly.plist
```

Optional extra: a second, tiny plist at 07:00 that just fires an `osascript` notification ("Second Brain: N changes, report ready") if you want a nudge rather than checking manually.

---

## 6. The morning report

`_reports/Review-MM-DD-YYYY.md`, generated as the last step of the nightly run:

```markdown
# Review — July 16, 2026

Run: 02:00–02:14 · 1,842 notes scanned · 6 changed · 3 new · est. cost $0.11

## Significant changes
(materiality threshold: renames, merges, taxonomy moves, deletions — configurable in config.yaml)

### Merged "Docker Networking" into "Docker Fundamentals"
Both notes covered the same bridge-network explanation with ~80% overlap.
Merged under the existing note; the newer note's diagram was kept, its
duplicate prose was not. [[Docker Fundamentals]]

### Moved 2 notes from 00-Inbox to 02-Areas/Cybersecurity
High embedding similarity (>0.86) to your existing Security+ study notes.

## New notes from today's daily digestion
- [[Qwen3 Local Model Notes]] ← from [[07-15-2026]]
- [[ScotiaTech Interview Followups]] ← from [[07-15-2026]]

## Minor changes (23 total — see full log)
Typo fixes, tag additions, link suggestions accepted at high confidence.
[View full diff log](_reports/07-16-2026-full-diff.json)

## Flagged for your review
- [[Cloud Cost Optimization Draft]] — clarity score low, left untouched;
  the note starts three different arguments and doesn't land any of them
```

The "estimated cost" line is a natural byproduct of the model router logging tokens per call — cheap to add, and it keeps the pricing conversation from §7 grounded in what you're actually spending, not just the sticker rate.

---

## 7. Model routing — five options, mapped to pricing vs. quality

The right move isn't "pick one model," it's route each pipeline stage to the cheapest model that clears the quality bar for that specific job. Prices below are current as of mid-July 2026 — verify at each provider's pricing page before budgeting, since this moves fast.

| # | Model | Price (input/output per MTok) | Where it fits in this pipeline |
|---|---|---|---|
| 1 | **Qwen3-30B-A3B (via Ollama, native)** | **$0 marginal cost** | Bulk grammar pass, embedding-adjacent classification, first-pass daily digestion. Fits ~16GB unified memory at Q4, 256K context. Zero cost and nothing leaves your Mac — the right default for a personal journal. |
| 2 | **DeepSeek V4 Flash** | $0.14 / $0.28 (cache hits ~$0.003) | Ultra-budget cloud fallback if local quality isn't cutting it for a specific task but you don't want Claude-tier pricing. ~20–35x cheaper than Claude Haiku. OpenAI-compatible API, one-line swap in the router. |
| 3 | **Claude Haiku 4.5** | $1 / $5 | Title suggestions, tagging, manifest-diff summarization — high-volume, low-complexity nightly tasks where you want Anthropic's instruction-following without Sonnet pricing. |
| 4 | **Claude Sonnet 5** | $2 / $10 intro through Aug 31, 2026 (then $3 / $15) | The workhorse for anything that needs real judgment: daily-note digestion/synthesis, taxonomy decisions, drafting `AGENTS.md` and helping you draft `alma.md`, and the interactive chat agent via Copilot for Obsidian. This is my default pick if you want one model for "the hard stuff." |
| 5 | **Claude Opus 4.8** | $5 / $25 | Reserve for the one-time (or rare) deep vault-taxonomy analysis that produces the first `AGENTS.md` — a "run once, get it right" job where quality matters more than per-token cost, unlike the nightly recurring tasks above. |

Both the Batch API (flat 50% off, works well since nothing here needs a same-second response) and prompt caching (90% off repeated context, e.g. your taxonomy description sent with every note) apply on the Anthropic tiers and stack — worth wiring in from day one rather than retrofitting later.

Honorable mentions if you want to swap providers: **Gemini 3.5 Flash** ($1.50/$9, currently the best per-token coding/analysis value in Google's lineup) and **GPT-5.6 Terra** ($2.50/$15, OpenAI's mid-tier as of its July 9, 2026 release) both sit in roughly the same slot as Sonnet 5 above — reasonable alternates, not clearly better for this use case.

**Embeddings**: default to local (`nomic-embed-text` via Ollama, 768-dim — the balance point most Obsidian-focused guides converge on) so vault content never has to leave the Mac just to build the search index. If you later want a quality bump and don't mind the vault touching a cloud API, `voyage-3-large` is Anthropic's recommended pairing for Claude-based retrieval — but at these vault sizes the difference is a few dollars a year either way; privacy is the more relevant deciding factor than price here.

---

## 8. Safety & trust model (read this before flipping `DRY_RUN` off)

- Vault must be a **git repo**. Every apply run does `git commit` before touching anything and after, so any bad night is one `git revert` away from gone.
- **Dry-run by default**: proposed edits land in `_staging/`, nothing in the real vault changes, until you've read a few morning reports and trust the judgment calls.
- **Materiality threshold** (`config.yaml`) controls what the morning report elaborates on vs. lumps into a count — you can turn this up or down.
- The pipeline **never invents content** to fill a gap in an underdeveloped note — low-clarity notes get flagged for you, not auto-expanded. This is a deliberate constraint, not a missing feature.

---

## 9. Build roadmap (phase 2, once you confirm §0)

1. `manifest.py` — the diff scanner. No LLM calls yet; just prove the hash/diff loop is fast and correct on your real vault.
2. `llm_router.py` + `config.yaml` — wire up the provider/model routing table from §7 as actual code, so swapping models later is a config change, not a rewrite.
3. `reorganizer.py` in dry-run only — get titles/grammar/taxonomy proposals landing in `_staging/` and readable.
4. `report.py` — even a rough version early gives you visibility into what steps 3 onward are actually doing.
5. `digestor.py` — daily note → atomic notes, once you trust the reorganizer's judgment on placement/linking.
6. `agents_md_builder.py` — run once against your real, reorganized vault.
7. launchd + Docker wiring, then flip `DRY_RUN=false` once a few weeks of reports look right.

Companion files in this delivery: `docker-compose.yml`, `config.example.yaml`, `AGENTS.md.template`, `alma.md.template`, `com.alejandrogarcia.secondbrain.nightly.plist`.
