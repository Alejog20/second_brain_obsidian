"""Recap module: a morning reinforcement summary of yesterday's newly digested notes."""

from dataclasses import dataclass
from typing import Optional

from .llm_router import Router

RECAP_SYSTEM_PROMPT = (
    "You write a short morning reinforcement summary for someone reviewing what they wrote "
    "and learned the day before. Given the notes below, write a brief recap (a few sentences) "
    "of the key ideas, then 2-3 short recall questions to test whether they remember the "
    "material - don't answer the questions yourself. Keep the whole thing readable in under a "
    "minute; this is a memory jog, not a full re-explanation of the material."
)


@dataclass(frozen=True)
class RecapEntry:
    """One digested note's title and content, feeding into the recap."""

    title: str
    content: str


class RecapGenerator:
    """Builds a short reinforcement recap from a day's newly digested notes."""

    def __init__(self, router: Router) -> None:
        self._router = router

    def build(self, entries: list[RecapEntry], date_str: str) -> Optional[str]:
        """Generate the recap markdown, or None if there's nothing to recap."""
        if not entries:
            return None
        material = "\n\n".join(f"## {entry.title}\n{entry.content}" for entry in entries)
        response = self._router.generate("daily_recap", system=RECAP_SYSTEM_PROMPT, prompt=material)
        return "\n".join([f"# Recap — {date_str}", "", response.text.strip()])
