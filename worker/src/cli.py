"""CLI module: a Rich-powered command-line utility for running the Second Brain pipeline."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.table import Table

from .config import Config, load_config
from .git_safety import GitSafety, VaultNotAGitRepoError
from .jobs.nightly_run import NightlyRun, resolve_dry_run, strftime_pattern
from .vault_io import VaultIO
from .worktree import NightlyWorktree

app = typer.Typer(help="Second Brain: nightly Obsidian vault curation pipeline.", no_args_is_help=True)
console = Console()


def _configure_logging(verbose: bool, quiet: bool) -> None:
    """Wire Python's standard logging module to Rich for readable, leveled CLI output."""
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=verbose, rich_tracebacks=True)],
        force=True,
    )
    if not verbose:
        # httpx logs every request at INFO by default - too noisy for this CLI's own INFO level.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug-level logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only warnings and errors."),
) -> None:
    """Second Brain: nightly Obsidian vault curation pipeline."""
    _configure_logging(verbose, quiet)


def _load_config_or_exit(config_path: Optional[Path]) -> Config:
    """Load config.yaml, printing a clean error and exiting instead of a raw traceback."""
    try:
        return load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        raise typer.Exit(code=1)


@app.command()
def run(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.yaml (defaults to $CONFIG_PATH or ./config.yaml)."),
    vault: Optional[Path] = typer.Option(None, "--vault", help="Override the vault path from config.yaml for this run."),
    apply: bool = typer.Option(False, "--apply", help="Write changes live for this run, overriding config.yaml/DRY_RUN."),
    dry_run_flag: bool = typer.Option(False, "--dry-run", help="Force dry-run for this run, overriding config.yaml/DRY_RUN."),
    git_review: Optional[bool] = typer.Option(None, "--git-review/--no-git-review", help="Override config.yaml's git_review.enabled for this run."),
    show_report: bool = typer.Option(True, "--show-report/--no-show-report", help="Print the generated report to the terminal when done."),
) -> None:
    """Run the nightly pipeline once: manifest diff -> reorganize -> digest -> AGENTS.md -> report."""
    if apply and dry_run_flag:
        console.print("[bold red]Error:[/bold red] --apply and --dry-run are mutually exclusive.")
        raise typer.Exit(code=1)

    config = _load_config_or_exit(config_path)

    if apply:
        dry_run = False
    elif dry_run_flag:
        dry_run = True
    else:
        dry_run = resolve_dry_run(config)

    use_git_review = config.git_review.enabled if git_review is None else git_review
    vault_root = vault or Path(config.vault.path)

    worktree_obj: Optional[NightlyWorktree] = None
    if use_git_review:
        worktree_obj = NightlyWorktree(
            vault_root=vault_root,
            worktree_path=Path(config.git_review.worktree_path),
            branch=config.git_review.branch,
            base_branch=config.git_review.base_branch,
            remote=config.git_review.remote,
        )
        with console.status("Syncing review worktree..."):
            vault_root = worktree_obj.sync()
        dry_run = False
        console.print(f"[cyan]Review mode:[/cyan] writing to branch [bold]{config.git_review.branch}[/bold], vault checkout untouched.")

    try:
        with console.status("Running nightly pipeline..."):
            report = NightlyRun(config, dry_run=dry_run, vault_root=vault_root).run()
    except VaultNotAGitRepoError as exc:
        console.print(f"[bold red]Git error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if worktree_obj is not None:
        with console.status("Pushing review branch..."):
            worktree_obj.push()
        console.print(f"[green]Pushed[/green] {config.git_review.branch} - review on GitHub and merge when ready.")

    mode_label = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]APPLIED[/green]"
    console.print(f"{mode_label} - pipeline run complete.")
    if show_report:
        console.print(Markdown(report))


@app.command()
def status(config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.yaml.")) -> None:
    """Show the current configuration: vault, safety mode, model routing, git review."""
    config = _load_config_or_exit(config_path)

    settings = Table(title="Second Brain configuration")
    settings.add_column("Setting")
    settings.add_column("Value")
    settings.add_row("Vault path", config.vault.path)
    settings.add_row("Daily notes folder", config.vault.daily_notes_folder or "(vault root)")
    settings.add_row("Default new-note folder", config.vault.default_new_note_folder or "(vault root)")
    settings.add_row("Safety mode", config.safety.mode)
    settings.add_row("Require git", str(config.safety.require_git))
    settings.add_row("Materiality threshold", config.safety.materiality_threshold)
    settings.add_row("Embeddings", f"{config.embeddings.provider} / {config.embeddings.model}")
    settings.add_row("Git review enabled", str(config.git_review.enabled))
    if config.git_review.enabled:
        settings.add_row("Review branch", f"{config.git_review.branch} <- {config.git_review.remote}/{config.git_review.base_branch}")
    console.print(settings)

    routing = Table(title="Model routing")
    routing.add_column("Task")
    routing.add_column("Provider")
    routing.add_column("Model")
    for task_key, task_cfg in config.models.items():
        routing.add_row(task_key, task_cfg.provider, task_cfg.model)
    console.print(routing)


@app.command()
def check(config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.yaml.")) -> None:
    """Preflight checks: vault exists and is a git repo, Ollama reachable, Gemini key present if needed."""
    config = _load_config_or_exit(config_path)
    table = Table(title="Preflight checks")
    table.add_column("Check")
    table.add_column("Result")
    all_ok = True

    vault_path = Path(config.vault.path)
    vault_ok = vault_path.is_dir()
    all_ok &= vault_ok
    table.add_row("Vault directory exists", "[green]OK[/green]" if vault_ok else f"[red]MISSING[/red] ({vault_path})")

    if vault_ok:
        try:
            GitSafety(vault_path, require_git=True)
            table.add_row("Vault is a git repo", "[green]OK[/green]")
        except VaultNotAGitRepoError:
            all_ok = False
            table.add_row("Vault is a git repo", "[red]NOT A GIT REPO[/red]")

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        httpx.get(f"{ollama_host}/api/tags", timeout=3.0).raise_for_status()
        table.add_row("Ollama reachable", f"[green]OK[/green] ({ollama_host})")
    except httpx.HTTPError:
        all_ok = False
        table.add_row("Ollama reachable", f"[red]UNREACHABLE[/red] ({ollama_host})")

    if any(task_cfg.provider == "gemini" for task_cfg in config.models.values()):
        has_key = bool(os.environ.get("GEMINI_API_KEY"))
        all_ok &= has_key
        table.add_row("GEMINI_API_KEY set", "[green]OK[/green]" if has_key else "[red]MISSING[/red]")

    console.print(table)
    if not all_ok:
        raise typer.Exit(code=1)


@app.command()
def report(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.yaml."),
    date: Optional[str] = typer.Option(None, "--date", help="Report date matching daily_note_date_format; defaults to today."),
) -> None:
    """Show a nightly report from the vault's _reports/ folder."""
    config = _load_config_or_exit(config_path)
    vault = VaultIO(Path(config.vault.path))
    date_str = date or datetime.now().strftime(strftime_pattern(config.vault.daily_note_date_format))
    rel_path = config.report.path.format(date=date_str)

    if not vault.exists(rel_path):
        console.print(f"[yellow]No report found for {date_str}[/yellow] ({rel_path})")
        raise typer.Exit(code=1)
    console.print(Markdown(vault.read_raw(rel_path)))


if __name__ == "__main__":
    app()
