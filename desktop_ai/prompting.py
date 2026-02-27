"""Prompt-building helpers."""

from __future__ import annotations

from collections.abc import Mapping


def build_context_block(context: Mapping[str, str]) -> str:
    """Render context entries as deterministic key-value lines."""
    if not context:
        return "No structured context was collected."
    lines: list[str] = [f"- {key}: {value}" for key, value in sorted(context.items())]
    return "\n".join(lines)


def build_user_prompt(context: Mapping[str, str], user_note: str | None = None) -> str:
    """Build the user message that accompanies the screenshot."""
    note: str = user_note.strip() if user_note else "No additional user note."
    context_block: str = build_context_block(context)
    return (
        "Analyze my current desktop activity using both screenshot and structured context.\n\n"
        f"User note:\n{note}\n\n"
        f"Structured context:\n{context_block}\n\n"
        "Provide:\n"
        "1. A short description of what I seem to be doing.\n"
        "2. One practical suggestion I can do next.\n"
        "3. A concise spoken-style response (max 3 sentences)."
    )
