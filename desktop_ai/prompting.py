"""Prompt-building helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timezone

from desktop_ai.types import MemoryRecord


def build_context_block(context: Mapping[str, str]) -> str:
    """Render context entries as deterministic key-value lines."""
    if not context:
        return "No structured context was collected."
    lines: list[str] = [f"- {key}: {value}" for key, value in sorted(context.items())]
    return "\n".join(lines)


def _truncate_text(value: str, *, max_chars: int) -> str:
    """Trim whitespace and enforce a hard character limit."""
    collapsed: str = " ".join(value.split()).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    if max_chars <= 3:
        return collapsed[:max_chars]
    return collapsed[: max_chars - 3].rstrip() + "..."


def build_memory_block(
    memories: Sequence[MemoryRecord],
    *,
    max_entry_chars: int,
) -> str:
    """Render recalled memories in a concise bullet list."""
    if not memories:
        return "No relevant prior memories were recalled."

    lines: list[str] = []
    for memory in memories:
        timestamp_text: str = memory.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        note_text: str = _truncate_text(memory.user_note or "(no user note)", max_chars=max_entry_chars)
        reply_text: str = _truncate_text(memory.assistant_reply, max_chars=max_entry_chars)
        if memory.action_summary:
            action_text: str = _truncate_text(memory.action_summary, max_chars=max_entry_chars)
            lines.append(
                f"- [{timestamp_text}] user={note_text} | assistant={reply_text} | actions={action_text}"
            )
            continue
        lines.append(f"- [{timestamp_text}] user={note_text} | assistant={reply_text}")
    return "\n".join(lines)


def build_user_prompt(
    context: Mapping[str, str],
    user_note: str | None = None,
    memories: Sequence[MemoryRecord] = (),
    *,
    desktop_control_enabled: bool = False,
    max_actions_per_turn: int = 5,
    allowed_launch_commands: tuple[str, ...] = (),
    memory_entry_chars: int = 240,
) -> str:
    """Build the user message that accompanies the screenshot."""
    note: str = user_note.strip() if user_note else ""
    context_block: str = build_context_block(context)
    memory_block: str = build_memory_block(memories, max_entry_chars=max(80, memory_entry_chars))
    if desktop_control_enabled:
        launch_scope: str = (
            ", ".join(allowed_launch_commands)
            if allowed_launch_commands
            else "(none; launch commands will be rejected)"
        )
        if note:
            note_block: str = f"User note:\n{note}\n"
        else:
            note_block = (
                "No explicit user note was captured.\n"
                "Do not execute any desktop actions unless a direct user request is clear.\n"
            )
        return (
            "You are speaking to the user in real time as their buddy, Sophie.\n"
            "Desktop control mode is enabled.\n\n"
            "Return exactly one JSON object with this schema:\n"
            "{\n"
            '  "spoken_reply": "<what you will say aloud>",\n'
            '  "actions": [\n'
            '    {"type": "<action type>", "...": "action args"}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- `spoken_reply` must always be present and conversational.\n"
            "- Do not narrate low-level desktop steps in `spoken_reply` (avoid lists like click/type/press).\n"
            "- Keep the JSON compact and valid.\n"
            "- For `type_text`, keep each `text` value under 120 characters.\n"
            "- If long writing is requested, ask to continue in chunks instead of outputting one huge block.\n"
            "- `actions` must be present; use [] when no action is needed.\n"
            f"- Emit at most {max(1, max_actions_per_turn)} actions.\n"
            "- If a request is ambiguous, ask a clarifying question in `spoken_reply` and use `actions: []`.\n"
            "- Never invent action types.\n"
            "- Keep `spoken_reply` short (1 to 3 sentences).\n"
            "- Launch allowlist prefixes: "
            f"{launch_scope}\n\n"
            "Supported action types and args:\n"
            '- `move_mouse`: {"type":"move_mouse","x":int,"y":int,"duration":float?}\n'
            '- `drag_mouse`: {"type":"drag_mouse","x":int,"y":int,"duration":float?,"button":"left|middle|right"?}\n'
            '- `click`: {"type":"click","x":int?,"y":int?,"button":"left|middle|right"?,"clicks":int?,"interval":float?}\n'
            '- `type_text`: {"type":"type_text","text":str,"interval":float?}\n'
            '- `press`: {"type":"press","key":str,"presses":int?,"interval":float?}\n'
            '- `hotkey`: {"type":"hotkey","keys":[str,str,...]}  # at least 2 keys\n'
            '- `scroll`: {"type":"scroll","amount":int}\n'
            '- `launch`: {"type":"launch","command":str}\n'
            '- `wait`: {"type":"wait","seconds":float}\n\n'
            f"{note_block}\n"
            f"Relevant prior memories:\n{memory_block}\n\n"
            f"Structured context:\n{context_block}\n"
        )
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
            f"Relevant prior memories:\n{memory_block}\n\n"
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
        f"Relevant prior memories:\n{memory_block}\n\n"
        f"Structured context:\n{context_block}\n\n"
        "Return only the spoken reply."
    )


def build_action_repair_prompt(
    *,
    raw_response: str,
    user_note: str | None,
    max_actions_per_turn: int,
    allowed_launch_commands: tuple[str, ...] = (),
) -> str:
    """Build a focused retry prompt that repairs invalid desktop-control output."""
    launch_scope: str = (
        ", ".join(allowed_launch_commands)
        if allowed_launch_commands
        else "(none; launch commands will be rejected)"
    )
    note: str = user_note.strip() if user_note else ""
    note_block: str = note if note else "(none captured)"
    return (
        "Repair the draft assistant output into a valid desktop action plan.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "spoken_reply": "<what you will say aloud>",\n'
        '  "actions": [\n'
        '    {"type": "<action type>", "...": "action args"}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Return JSON only; no markdown and no extra text.\n"
        "- `spoken_reply` must be short and conversational (1 to 3 sentences).\n"
        "- Do not narrate low-level steps in `spoken_reply`.\n"
        "- Keep the JSON compact and valid.\n"
        "- For `type_text`, keep each `text` value under 120 characters.\n"
        "- If long writing is requested, ask to continue in chunks instead of outputting one huge block.\n"
        "- `actions` must always be present; use [] if unclear.\n"
        f"- Emit at most {max(1, max_actions_per_turn)} actions.\n"
        "- If details are ambiguous, ask a clarifying question in `spoken_reply` and set `actions` to [].\n"
        "- Never invent action types.\n"
        "- Launch allowlist prefixes: "
        f"{launch_scope}\n\n"
        "Supported action types and args:\n"
        '- `move_mouse`: {"type":"move_mouse","x":int,"y":int,"duration":float?}\n'
        '- `drag_mouse`: {"type":"drag_mouse","x":int,"y":int,"duration":float?,"button":"left|middle|right"?}\n'
        '- `click`: {"type":"click","x":int?,"y":int?,"button":"left|middle|right"?,"clicks":int?,"interval":float?}\n'
        '- `type_text`: {"type":"type_text","text":str,"interval":float?}\n'
        '- `press`: {"type":"press","key":str,"presses":int?,"interval":float?}\n'
        '- `hotkey`: {"type":"hotkey","keys":[str,str,...]}  # at least 2 keys\n'
        '- `scroll`: {"type":"scroll","amount":int}\n'
        '- `launch`: {"type":"launch","command":str}\n'
        '- `wait`: {"type":"wait","seconds":float}\n\n'
        f"User note:\n{note_block}\n\n"
        "Draft output to repair:\n"
        f"{raw_response.strip()}\n"
    )
