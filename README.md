# Second Brain

A nightly pipeline that reads an Obsidian vault, cleans up notes, digests the day's
journal entry into atomic notes, and writes a report explaining what changed and why.
It's built to run unsupervised against a personal vault, so it defaults to dry-run:
nothing gets written to your real notes until you've read a few reports and trust it.

Full design notes and reasoning live in `SECOND_BRAIN_ARCHITECTURE.md`.

## What it actually does

Each run:

1. Scans the vault and hashes every note, comparing against the last run so only
   changed files get reprocessed.
2. For each changed note: checks the title, fixes grammar without rewriting your
   voice, suggests a better folder if a very similar note already lives elsewhere,
   and suggests related notes to link — none of this is applied blindly, low-confidence
   or unclear cases get flagged for you instead.
3. Splits today's daily note into atomic ideas. Each one either gets merged into an
   existing note it's clearly related to, or becomes a new note with proper frontmatter.
4. Regenerates `AGENTS.md` (a description of your vault's structure, for other AI
   tools) if the folder structure changed.
5. Writes a report to `_reports/` summarizing what happened, and commits everything
   to git so any night's changes can be reverted with one command if they're wrong.

All of the actual language-model work — grammar, titles, digesting the daily note,
and the embeddings used for similarity search — runs through Gemini's API.

## Before you start

You'll need:

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Gemini API key (free tier works fine) — get one at
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- Your Obsidian vault as a local git repository (`git init` inside it if it isn't
  one yet — this is how the pipeline gives you a safety net for anything it changes)

## Setup

Clone the repo, then install dependencies:

```bash
cd worker
uv sync --extra dev
```

Copy the environment template and fill it in:

```bash
cp .env.example .env
```

Open `.env` and set:

- `VAULT_PATH` — the absolute path to your vault on disk
- `GEMINI_API_KEY` — your API key

Everything else in `.env` is optional. Leave `DATA_PATH` blank unless you want the
local cache (embeddings index, change-tracking database) somewhere other than
`worker/data/`.

Take a look at `worker/config.yaml` too. The defaults assume daily notes and new
notes sit at your vault's root — if your vault uses a dated subfolder like
`01-Daily/` instead, or a different daily-note filename format, update
`vault.daily_notes_folder` and `vault.daily_note_date_format` to match.

## Running it

From the `worker/` directory, or using the `make` shortcuts from the repo root
(they just `cd worker` for you):

```bash
# confirm your vault, git repo, and API key are all actually working
uv run second-brain check

# see what's configured — safety mode, which model handles which task
uv run second-brain status

# do a safe first run — nothing gets written to your real notes,
# proposals land in a _staging/ folder inside the vault instead
uv run second-brain run --dry-run --no-git-review
```

Read the report it prints, and check what landed in `_staging/`. Once you're
comfortable with what it's proposing, you can let it write for real:

```bash
uv run second-brain run --apply --no-git-review
```

There's also a review-branch mode, if you'd rather approve changes as a GitHub
pull request instead of reading a local report:

```bash
uv run second-brain run --apply --git-review
```

This writes to a separate branch and pushes it — your vault's live checkout is
never touched until you open the PR and merge it yourself. `config.yaml`'s
`git_review` section controls the branch name and where it pushes to.

To look back at a previous night's report without running anything:

```bash
uv run second-brain report --date 2026-07-19
```

(That date has to match whatever `vault.daily_note_date_format` is set to in
`config.yaml` — `YYYY-MM-DD` by default here.)

Run `second-brain --help`, or `second-brain <command> --help`, for the full list
of options on any command.

## Configuration

`worker/config.yaml` is the main configuration file. The pieces worth knowing about:

- **`vault`** — where your notes live, how daily notes are named, which folders to
  skip (`_staging`, `_reports`, `.obsidian` by default).
- **`safety`** — `mode: dry_run` or `mode: apply`, whether a git repo is required
  (it is, by default), and how much detail the report includes.
- **`models`** — which model handles each task. Everything points at Gemini, a
  virtual/commercial LLM, out of the box - no local model to install or keep running.
  `llm_router.py` also has a working Ollama adapter for local inference if you ever
  want it, but it isn't the recommended or default path here.
- **`git_review`** — the PR-based review workflow described above.

Two safety switches have to agree before anything gets written for real: `config.yaml`'s
`safety.mode` and the `DRY_RUN` environment variable. Passing `--apply` or `--dry-run`
directly on the command line overrides both for that one run.

## What it won't do

- It won't write to your vault outside of `_staging/` unless you're in apply mode.
- It won't invent content to pad out a thin note — if a note's idea isn't clearly
  landed, it gets flagged for you to finish, not auto-expanded.
- It won't delete a note. Moving a note to a better folder is a real git-tracked
  rename, not a delete-and-recreate.
- Every apply-mode run is wrapped in a git commit before and after, so a bad night
  is one `git revert` away from gone.

## Project layout

```
worker/
  src/
    cli.py              the second-brain command
    jobs/nightly_run.py  orchestrates a full run
    manifest.py           tracks which notes changed since last run
    reorganizer.py         title, grammar, folder, and link suggestions
    digestor.py             daily note -> atomic notes
    agents_md_builder.py     regenerates AGENTS.md
    llm_router.py            talks to Gemini (and optionally Ollama)
    vector_store.py           embeddings and similarity search
    vault_io.py                safe reads/writes into the vault
    git_safety.py               the commit-before/after safety net
    worktree.py                  the git-review branch workflow
  tests/
  config.yaml
```

## Running the tests

```bash
cd worker
uv run pytest -q
```

All of it runs against temporary vaults and fakes for the model calls — nothing
in the test suite touches a real vault or makes a real API call.

## Docker

A `Dockerfile` and `docker-compose.yml` are included for running this on a
schedule (see `launchd/` for a macOS example). Docker wasn't the focus of this
build and hasn't been verified end to end the way the local `uv` path has — if
you go that route, expect to double-check the volume mounts and environment
variables in `docker-compose.yml` against your own setup.

## CLI Commands

Command reference

| **Command**              | **Flag / Option**        | **Description**                                                                                                                                                                   |
| ------------------------ | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Global (any command)** | `-v`, `--verbose`        | Enable debug-level logging.                                                                                                                                                       |
| **Global (any command)** | `-q`, `--quiet`          | Show only warnings and errors.                                                                                                                                                    |
| **Global (any command)** | `--help`                 | Display help for the selected command.                                                                                                                                            |
| **run**                  | *(no flags)*             | Run the entire pipeline once: **manifest → diff → reorganize → digest → AGENTS.md → report**.                                                                                     |
| **run**                  | `-c`, `--config PATH`    | Specify the path to `config.yaml`. Default: `$CONFIG_PATH` or `./config.yaml`.                                                                                                    |
| **run**                  | `--vault PATH`           | Override the vault path. Falls back to `$VAULT_PATH`, then the config file.                                                                                                       |
| **run**                  | `--apply`                | Force changes to be applied, overriding the `DRY_RUN` configuration.                                                                                                              |
| **run**                  | `--dry-run`              | Simulate execution without making changes, overriding the `DRY_RUN` configuration.                                                                                                |
| **run**                  | `--git-review`           | Enable Git review regardless of configuration.                                                                                                                                    |
| **run**                  | `--no-git-review`        | Disable Git review regardless of configuration.                                                                                                                                   |
| **run**                  | `--full`                 | Process every note in the vault instead of only notes changed since the previous run.                                                                                             |
| **run**                  | `--show-report`          | Print the generated report to the terminal (default behavior).                                                                                                                    |
| **run**                  | `--no-show-report`       | Do not print the generated report to the terminal.                                                                                                                                |
| **status**               | `-c`, `--config PATH`    | Specify the path to `config.yaml`.                                                                                                                                                |
| **status**               | `--vault PATH`           | Override the vault path.                                                                                                                                                          |
| **status**               | *(output)*               | Display the current configuration, materiality threshold, embeddings status, Git review settings, and the complete model routing table.                                           |
| **check**                | `-c`, `--config PATH`    | Specify the path to `config.yaml`.                                                                                                                                                |
| **check**                | `--vault PATH`           | Override the vault path.                                                                                                                                                          |
| **check**                | *(output)*               | Run preflight checks: verify the vault exists, is a Git repository, confirm Ollama is reachable (if configured), validate Gemini connectivity, and check configuration integrity. |
| **report**               | `-c`, `--config PATH`    | Specify the path to `config.yaml`.                                                                                                                                                |
| **report**               | `--vault PATH`           | Override the vault path.                                                                                                                                                          |
| **report**               | `--date STR`             | Generate a report for a specific date. Uses the configured date format. Default: **today**.                                                                                       |
| **report**               | `--type review \| recap` | Select the report type. `review` generates the morning reinforcement summary (default); `recap` generates a recap report.                                                         |
