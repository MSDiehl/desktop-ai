"""Command-line interface for the desktop assistant."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from collections.abc import Sequence

from dotenv import load_dotenv

from desktop_ai.assistant import DesktopAssistant
from desktop_ai.audio import LocalAudioOutput
from desktop_ai.config import AssistantConfig, ElevenLabsConfig
from desktop_ai.context import CompositeContextCollector, build_default_context_registry
from desktop_ai.elevenlabs_client import ElevenLabsSpeechSynthesizer
from desktop_ai.openai_client import OpenAITextGenerator
from desktop_ai.screen import MSSScreenCapturer
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

    assistant = DesktopAssistant(
        context_collector=context_collector,
        screen_capturer=screen_capturer,
        text_generator=text_generator,
        speech_synthesizer=speech_synthesizer,
        audio_output=audio_output,
        voice_trigger_listener=voice_trigger_listener,
        enable_speech=speech_enabled,
        logger=logging.getLogger("desktop_ai.assistant"),
    )
    return assistant, config


def run(args: argparse.Namespace) -> int:
    """Run the assistant command and return exit code."""
    load_dotenv()
    assistant, config = build_assistant(args)
    interval_seconds: float = args.interval if args.interval is not None else config.interval_seconds

    if args.once:
        result = assistant.run_once(user_note=args.note)
        print(result.response_text)
        if result.audio_path is not None:
            print(f"Audio saved: {result.audio_path}")
        return 0

    try:
        assistant.run_loop(
            interval_seconds=interval_seconds,
            user_note=args.note,
            max_turns=args.max_turns,
        )
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Stopped by user.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)
