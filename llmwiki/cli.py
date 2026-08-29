"""Command-line interface for the RAG core.

The CLI is intentionally small. Subcommands:

- ``llmwiki index``    — run one indexing pass over a vault.
- ``llmwiki status``   — show database state.
- ``llmwiki search``   — (Phase 5+) search the indexed vault.

Phase 1 implements ``index`` and ``status`` only. The plugin is the
intended long-term entry point; the CLI is for testing, scripting,
and one-off queries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NoReturn

import click

from .config import Settings
from .indexer import Indexer, summarise_database
from .logging import setup_logging


def _resolve_settings(
    *,
    vault: Path | None,
    db: Path | None,
    watch: bool,
) -> Settings:
    """Build Settings from CLI flags, falling back to env / defaults."""
    from .config import Settings as _Settings

    base = _Settings.from_env()
    return _Settings(
        vault_path=(vault or base.vault_path).expanduser().resolve(),
        db_path=(db or base.db_path).expanduser().resolve(),
        file_watch=watch or base.file_watch,
        log_level=base.log_level,
        log_format=base.log_format,
    )


@click.group()
@click.option("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR (overrides env)")
@click.option(
    "--log-format", default=None, type=click.Choice(["text", "json"]), help="Log output format"
)
def main(log_level: str | None, log_format: str | None) -> None:
    """Local-first hybrid RAG over an Obsidian vault."""
    setup_logging(level=log_level, fmt=log_format)


@main.command()
@click.option(
    "--vault",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Path to the Obsidian vault (default: $LLMWIKI_VAULT or cwd)",
)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite database file (default: $LLMWIKI_DB or ./.data/llmwiki.sqlite)",
)
@click.option(
    "--mode",
    type=click.Choice(["incremental", "full"]),
    default="incremental",
    help="Indexing mode (default: incremental)",
)
@click.option(
    "--watch/--no-watch",
    default=False,
    help="Watch the vault for changes and re-index (Phase 1: not yet implemented)",
)
def index(vault: Path | None, db: Path | None, mode: str, watch: bool) -> None:
    """Index a vault into the local SQLite database."""
    if watch:
        click.echo("--watch is not yet implemented (Phase 1); running one pass.", err=True)
    settings = _resolve_settings(vault=vault, db=db, watch=False)
    click.echo(f"indexing: vault={settings.vault_path} db={settings.db_path} mode={mode}")
    indexer = Indexer(settings)
    stats = indexer.run(mode=mode)
    click.echo(
        f"done: seen={stats.documents_seen} added={stats.documents_added} "
        f"updated={stats.documents_updated} removed={stats.documents_removed} "
        f"skipped={stats.documents_skipped} "
        f"chunks: +{stats.chunks_added} ~{stats.chunks_updated} -{stats.chunks_removed} "
        f"errors={len(stats.errors)}"
    )
    if stats.errors:
        sys.exit(1)


@main.command()
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite database file (default: $LLMWIKI_DB or ./.data/llmwiki.sqlite)",
)
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON")
def status(db: Path | None, as_json: bool) -> None:
    """Show database state."""
    from .config import Settings as _Settings

    base = _Settings.from_env()
    db_path = (db or base.db_path).expanduser().resolve()
    summary = summarise_database(db_path)
    if as_json:
        click.echo(json.dumps(summary, indent=2, default=str))
        return
    if not summary.get("exists"):
        click.echo(f"database does not exist: {db_path}")
        return
    click.echo(f"database: {summary['path']}")
    click.echo(f"documents: {summary['documents']}")
    click.echo(f"chunks: {summary['chunks']}")
    click.echo(f"index runs: {summary['runs']}")
    last_run = summary.get("last_run")
    if last_run is not None:
        # last_run is dict[str, object]; cast to dict[str, Any] for safe subscripting.
        from typing import Any, cast

        lr = cast("dict[str, Any]", last_run)
        click.echo(
            f"last run: mode={lr['mode']} added={lr['added']} updated={lr['updated']} "
            f"removed={lr['removed']} skipped={lr['skipped']}"
        )


@main.command()
@click.option(
    "--vault", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None
)
@click.option("--db", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--query", required=True, help="Search query")
@click.option("--top-k", default=10, show_default=True, type=int)
def search(vault: Path | None, db: Path | None, query: str, top_k: int) -> NoReturn:
    """Search the indexed vault (Phase 5+; currently a stub)."""
    raise click.ClickException("search is not yet implemented (Phase 5+)")


if __name__ == "__main__":  # pragma: no cover
    main()
