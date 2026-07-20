"""Markers module: the shared Obsidian-callout format for AI-added content in notes.

Every AI-added passage (fact-check flags, and any future enhancement-style addition) is
appended as a callout, never merged into the user's own prose - so it's always visually
and structurally distinguishable from what the user actually wrote.
"""


def format_callout(kind: str, date: str, body: str) -> str:
    """Format an AI-added passage as an Obsidian callout block, e.g. `> [!ai-fact-check] ...`."""
    lines = body.strip().splitlines()
    quoted_body = "\n".join(f"> {line}" if line else ">" for line in lines)
    return f"> [!{kind}] Second Brain — {date}\n{quoted_body}"
