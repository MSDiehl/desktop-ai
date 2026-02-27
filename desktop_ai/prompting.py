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
    note: str = user_note.strip() if user_note else ""
    context_block: str = build_context_block(context)
    if note:
        return (
            "You are speaking to the user in real time as their buddy, Sophie.\n"
            "Primary objective: respond to the user note directly.\n\n"
            "Behavior rules:\n"
            "- Answer the user note first.\n"
            "- Use screenshot/context only if it improves the answer.\n"
            "- If the note asks for help, give clear practical guidance.\n"
            "- If the note is casual, keep a friendly buddy tone.\n"
            "- Do not give unsolicited productivity coaching.\n"
            "- For recommendation questions, give one top pick plus 2 to 4 options.\n"
            "- Keep replies short (1 to 4 sentences) unless asked for more.\n\n"
            f"User note:\n{note}\n\n"
            f"Structured context:\n{context_block}\n\n"
            "Return only the spoken reply."
        )

    return (
        "No explicit user request was provided.\n"
        "Use the screenshot and structured context to give a short buddy-style check-in.\n\n"
        "Behavior rules:\n"
        "- Sound human and conversational, not robotic.\n"
        "- Mention one relevant observation.\n"
        "- Optionally suggest one useful next step.\n"
        "- Avoid judgmental productivity advice.\n"
        "- Keep it to 1 to 2 sentences.\n\n"
        f"Structured context:\n{context_block}\n\n"
        "Return only the spoken reply."
    )
