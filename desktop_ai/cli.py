"""Command-line interface for the desktop assistant."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from collections.abc import Callable, Sequence

from dotenv import load_dotenv

from desktop_ai.assistant import DesktopAssistant
from desktop_ai.audio import LocalAudioOutput
from desktop_ai.config import AssistantConfig, AvatarConfig, DesktopControlConfig, ElevenLabsConfig
from desktop_ai.context import CompositeContextCollector, build_default_context_registry
from desktop_ai.desktop_control import DesktopController, format_action
from desktop_ai.elevenlabs_client import ElevenLabsSpeechSynthesizer
from desktop_ai.memory import SQLiteMemoryStore
from desktop_ai.openai_client import OpenAITextGenerator
from desktop_ai.screen import MSSScreenCapturer
from desktop_ai.types import DesktopActionPlan
from desktop_ai.voice_activation import OpenAIWakeWordListener


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Context-aware desktop AI assistant.")
    parser.add_argument("--once", action="store_true", help="Run one assistant turn and exit.")
    parser.add_argument("--interval", type=float, help="Override assistant interval in seconds.")
    parser.add_argument("--max-turns", type=int, default=None, help="Maximum loop turns before exit.")
    parser.add_argument("--note", type=str, default=None, help="Optional note to include in prompts.")
    parser.add_argument(
        "--context-providers",
        type=str,
        default=None,
        help="Comma-separated provider names (for example: timestamp,environment,active_window).",
    )
    parser.add_argument("--monitor-index", type=int, default=None, help="mss monitor index (default from env).")
    parser.add_argument("--no-speech", action="store_true", help="Disable ElevenLabs synthesis and playback.")
    parser.add_argument("--no-autoplay", action="store_true", help="Store audio file but do not play it.")
    voice_group = parser.add_mutually_exclusive_group()
    voice_group.add_argument("--voice-trigger", action="store_true", help="Enable wake-word voice activation.")
    voice_group.add_argument("--no-voice-trigger", action="store_true", help="Disable wake-word voice activation.")
    parser.add_argument("--wake-word", type=str, default=None, help="Override wake word (default from env).")
    avatar_group = parser.add_mutually_exclusive_group()
    avatar_group.add_argument("--avatar", action="store_true", help="Enable always-on-top desktop avatar overlay.")
    avatar_group.add_argument("--no-avatar", action="store_true", help="Disable desktop avatar overlay.")
    parser.add_argument(
        "--no-avatar-auto-move",
        action="store_true",
        help="Disable autonomous movement for the desktop avatar.",
    )
    parser.add_argument("--avatar-size", type=int, default=None, help="Avatar window size in pixels.")
    parser.add_argument(
        "--avatar-opacity",
        type=float,
        default=None,
        help="Avatar opacity from 0.4 to 1.0.",
    )
    desktop_control_group = parser.add_mutually_exclusive_group()
    desktop_control_group.add_argument(
        "--desktop-control",
        action="store_true",
        help="Enable keyboard/mouse/application control actions.",
    )
    desktop_control_group.add_argument(
        "--no-desktop-control",
        action="store_true",
        help="Disable keyboard/mouse/application control actions.",
    )
    parser.add_argument(
        "--auto-approve-actions",
        action="store_true",
        help="Execute desktop actions without interactive confirmation.",
    )
    parser.add_argument(
        "--allow-launch",
        type=str,
        default=None,
        help="Comma-separated command prefixes allowed for launch actions (for example: notepad,code).",
    )
    parser.add_argument(
        "--max-actions-per-turn",
        type=int,
        default=None,
        help="Upper bound for desktop actions executed in a single assistant turn.",
    )
    return parser


def configure_logging(level: str) -> None:
    """Configure process logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _split_cli_csv(value: str) -> tuple[str, ...]:
    """Split comma-separated CLI values."""
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _build_action_approval_callback() -> Callable[[DesktopActionPlan], bool]:
    """Build an interactive callback for approving action plans."""

    def callback(plan: DesktopActionPlan) -> bool:
        stdin = sys.stdin
        if stdin is None or not hasattr(stdin, "isatty") or not stdin.isatty():
            logging.getLogger(__name__).warning(
                "Desktop actions were requested but stdin is unavailable/non-interactive; denying by default."
            )
            return False
        print("Desktop action plan proposed:")
        for index, action in enumerate(plan.actions, start=1):
            print(f"{index}. {format_action(action)}")
        try:
            decision: str = input("Approve these actions? [y/N]: ").strip().lower()
        except EOFError:
            logging.getLogger(__name__).warning(
                "Desktop action approval prompt failed due to EOF; denying by default."
            )
            return False
        return decision in {"y", "yes"}

    return callback


def build_assistant(args: argparse.Namespace) -> tuple[DesktopAssistant, AssistantConfig]:
    """Construct fully wired assistant from environment config + CLI overrides."""
    config: AssistantConfig = AssistantConfig.from_env()
    configure_logging(config.log_level)

    provider_names: tuple[str, ...] = (
        _split_cli_csv(args.context_providers)
        if args.context_providers is not None
        else config.context_provider_names
    )
    registry = build_default_context_registry()
    context_providers = registry.create_many(provider_names)
    context_collector = CompositeContextCollector(providers=context_providers)

    monitor_index: int = args.monitor_index if args.monitor_index is not None else config.monitor_index
    screen_capturer = MSSScreenCapturer(monitor_index=monitor_index)

    text_generator = OpenAITextGenerator(config=config.openai, system_prompt=config.system_prompt)

    voice_config = config.voice_trigger
    if args.voice_trigger:
        voice_config = replace(voice_config, enabled=True)
    if args.no_voice_trigger:
        voice_config = replace(voice_config, enabled=False)
    if args.wake_word is not None:
        wake_word: str = args.wake_word.strip()
        voice_config = replace(voice_config, wake_word=wake_word or voice_config.wake_word)

    voice_trigger_listener = None
    if voice_config.enabled:
        voice_trigger_listener = OpenAIWakeWordListener(
            openai_config=config.openai,
            voice_config=voice_config,
            logger=logging.getLogger("desktop_ai.voice_activation"),
        )

    speech_enabled: bool = config.enable_speech and not args.no_speech
    speech_synthesizer = None
    audio_output = None
    if speech_enabled:
        elevenlabs_config: ElevenLabsConfig = ElevenLabsConfig.from_env()
        speech_synthesizer = ElevenLabsSpeechSynthesizer(elevenlabs_config)
        audio_output = LocalAudioOutput(
            output_dir=config.artifacts_dir / "audio",
            autoplay=not args.no_autoplay,
        )

    avatar_config: AvatarConfig = config.avatar
    if args.avatar:
        avatar_config = replace(avatar_config, enabled=True)
    if args.no_avatar:
        avatar_config = replace(avatar_config, enabled=False)
    if args.no_avatar_auto_move:
        avatar_config = replace(avatar_config, auto_move=False)
    if args.avatar_size is not None:
        avatar_config = replace(avatar_config, size=max(100, args.avatar_size))
    if args.avatar_opacity is not None:
        opacity: float = max(0.4, min(1.0, args.avatar_opacity))
        avatar_config = replace(avatar_config, opacity=opacity)

    desktop_control_config: DesktopControlConfig = config.desktop_control
    if args.desktop_control:
        desktop_control_config = replace(desktop_control_config, enabled=True)
    if args.no_desktop_control:
        desktop_control_config = replace(desktop_control_config, enabled=False)
    if args.auto_approve_actions:
        desktop_control_config = replace(desktop_control_config, require_approval=False)
    if args.allow_launch is not None:
        desktop_control_config = replace(
            desktop_control_config,
            allowed_launch_commands=_split_cli_csv(args.allow_launch),
        )
    if args.max_actions_per_turn is not None:
        desktop_control_config = replace(
            desktop_control_config,
            max_actions_per_turn=max(1, args.max_actions_per_turn),
        )

    presence_overlay = None
    if avatar_config.enabled:
        try:
            from desktop_ai.avatar import TkAvatarOverlay

            presence_overlay = TkAvatarOverlay(
                auto_move=avatar_config.auto_move,
                size=avatar_config.size,
                opacity=avatar_config.opacity,
                logger=logging.getLogger("desktop_ai.avatar"),
            )
            presence_overlay.start()
        except Exception as error:
            logging.getLogger(__name__).warning("Avatar overlay disabled: %s", error)

    desktop_controller = None
    action_approval_callback: Callable[[DesktopActionPlan], bool] | None = None
    if desktop_control_config.enabled:
        try:
            desktop_controller = DesktopController(
                require_approval=desktop_control_config.require_approval,
                allowed_launch_commands=desktop_control_config.allowed_launch_commands,
                max_actions_per_turn=desktop_control_config.max_actions_per_turn,
                action_delay_seconds=desktop_control_config.action_delay_seconds,
                log_path=desktop_control_config.action_log_path,
                logger=logging.getLogger("desktop_ai.desktop_control"),
            )
            if desktop_control_config.require_approval:
                action_approval_callback = _build_action_approval_callback()
        except Exception as error:
            logging.getLogger(__name__).warning("Desktop control disabled: %s", error)

    memory_store = None
    if config.memory.enabled:
        try:
            memory_store = SQLiteMemoryStore(
                config=config.memory,
                logger=logging.getLogger("desktop_ai.memory"),
            )
        except Exception as error:
            logging.getLogger(__name__).warning("Memory store disabled: %s", error)

    assistant = DesktopAssistant(
        context_collector=context_collector,
        screen_capturer=screen_capturer,
        text_generator=text_generator,
        speech_synthesizer=speech_synthesizer,
        audio_output=audio_output,
        voice_trigger_listener=voice_trigger_listener,
        presence_overlay=presence_overlay,
        desktop_controller=desktop_controller,
        memory_store=memory_store,
        memory_recall_limit=config.memory.recall_limit,
        memory_prompt_chars=config.memory.prompt_entry_chars,
        action_approval_callback=action_approval_callback,
        enable_speech=speech_enabled,
        logger=logging.getLogger("desktop_ai.assistant"),
    )
    return assistant, config


def run(args: argparse.Namespace) -> int:
    """Run the assistant command and return exit code."""
    load_dotenv()
    assistant, config = build_assistant(args)
    interval_seconds: float = args.interval if args.interval is not None else config.interval_seconds

    try:
        if args.once:
            result = assistant.run_once(user_note=args.note)
            print(result.response_text)
            if result.action_results:
                for index, action_result in enumerate(result.action_results, start=1):
                    status: str = "ok" if action_result.success else "failed"
                    print(
                        f"Action {index} [{status}]: {format_action(action_result.action)} | "
                        f"{action_result.detail}"
                    )
            if result.audio_path is not None:
                print(f"Audio saved: {result.audio_path}")
            return 0

        assistant.run_loop(
            interval_seconds=interval_seconds,
            user_note=args.note,
            max_turns=args.max_turns,
        )
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Stopped by user.")
    finally:
        assistant.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)
