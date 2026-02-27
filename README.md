# desktop-ai

Context-aware desktop AI assistant in Python.

It captures:
- A screenshot of your desktop.
- Structured runtime context (timestamp, environment, active window).

It then:
- Uses OpenAI for multimodal reasoning (text + screenshot).
- Uses OpenAI transcription for wake-word voice activation.
- Uses ElevenLabs for TTS.
- Saves audio output and optionally plays it locally.

## Features

- Strongly typed architecture with Protocol-based components.
- Docstrings on all functions and methods.
- Dynamic context-provider registry for easy expansion.
- CLI-first workflow for one-shot or continuous assistant loops.
- Optional wake-word trigger (for example: `Lune, what do you think of this?`).

## Project layout

`desktop_ai/`
- `assistant.py`: orchestration loop.
- `config.py`: typed env config.
- `context.py`: pluggable context providers.
- `screen.py`: screenshot capture (mss).
- `openai_client.py`: OpenAI Responses API integration.
- `voice_activation.py`: microphone capture + wake-word transcription.
- `elevenlabs_client.py`: ElevenLabs TTS integration.
- `audio.py`: local WAV storage/playback.
- `cli.py`: command-line entrypoint.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -e .
```

3. Copy `.env.example` to `.env` and fill required values:
- `OPENAI_API_KEY`
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID`

Optional for wake-word mode:
- `ASSISTANT_ENABLE_VOICE_TRIGGER=true`
- `ASSISTANT_WAKE_WORD=Lune`

## Usage

Single run:

```bash
python -m desktop_ai --once
```

Continuous loop:

```bash
python -m desktop_ai
```

Continuous loop with wake-word activation:

```bash
python -m desktop_ai --voice-trigger --wake-word Lune
```

Useful options:

```bash
python -m desktop_ai --interval 6 --note "Focus on gaming commentary"
python -m desktop_ai --once --no-speech
python -m desktop_ai --context-providers timestamp,active_window --once
python -m desktop_ai --no-voice-trigger
```

## Dynamic extension points

To add more context:
1. Implement a class in `context.py` matching `ContextProvider`.
2. Register it in `build_default_context_registry()`.
3. Add it to `ASSISTANT_CONTEXT_PROVIDERS`.

To swap AI/TTS providers:
1. Implement methods compatible with `TextGenerator` and `SpeechSynthesizer`.
2. Wire your classes in `cli.py`.

## Notes

- Artifacts are written to `./artifacts/audio` by default.
- Response length can be reduced with `OPENAI_MAX_OUTPUT_TOKENS` in `.env`.
- On Unix, playback uses `afplay`, `aplay`, or `paplay` if installed.
- On Windows, playback uses `winsound` with WAV output.
