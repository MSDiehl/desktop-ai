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
- Persists long-term memory of turns and recalls relevant past events each turn.
- Optionally executes desktop actions (keyboard, mouse, launch commands).

## Features

- Strongly typed architecture with Protocol-based components.
- Docstrings on all functions and methods.
- Dynamic context-provider registry for easy expansion.
- Tkinter desktop launcher for editing `.env` settings and starting/stopping AI.
- CLI-first workflow for one-shot or continuous assistant loops.
- Optional wake-word trigger (for example: `Lune, what do you think of this?`).
- Optional always-on-top avatar overlay with `idle/listening/thinking/speaking` states.
- Optional desktop-control mode with bounded actions, approval prompts, and action logs.
- SQLite-backed memory store with bounded retention + relevance recall.

## Project layout

`desktop_ai/`

- `assistant.py`: orchestration loop.
- `launcher.py`: settings UI + start/stop launcher window.
- `config.py`: typed env config.
- `context.py`: pluggable context providers.
- `screen.py`: screenshot capture (mss).
- `openai_client.py`: OpenAI Responses API integration.
- `voice_activation.py`: microphone capture + wake-word transcription.
- `elevenlabs_client.py`: ElevenLabs TTS integration.
- `audio.py`: local WAV storage/playback.
- `avatar.py`: desktop avatar overlay and state animation.
- `desktop_control.py`: action-plan parsing + keyboard/mouse/app execution.
- `memory.py`: persistent memory storage + recall ranking.
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
- `ASSISTANT_VOICE_START_SILENCE_SECONDS=3.0` (max silence to wait for you to start speaking after wake word)
- `ASSISTANT_VOICE_END_SILENCE_SECONDS=1.0` (stop follow-up capture after trailing silence)
- `ASSISTANT_VOICE_FOLLOWUP_LISTEN_SECONDS=12.0` (max follow-up capture length safety cap)
- `ASSISTANT_VOICE_ACTIVITY_THRESHOLD=450` (raise if background noise causes premature triggers)

Optional for avatar overlay:

- `ASSISTANT_ENABLE_AVATAR=true`
- `ASSISTANT_AVATAR_AUTO_MOVE=true`

Optional for desktop control:

- `ASSISTANT_ENABLE_DESKTOP_CONTROL=true`
- `ASSISTANT_DESKTOP_REQUIRE_APPROVAL=true` (recommended safety default; launcher/.exe uses a GUI Yes/No approval prompt)
- `ASSISTANT_DESKTOP_ALLOWED_LAUNCH=notepad,code,calc` (prefix allowlist for launch actions)
- `ASSISTANT_DESKTOP_MAX_ACTIONS_PER_TURN=5`
- `ASSISTANT_DESKTOP_ACTION_LOG=actions.log` (written under `ASSISTANT_ARTIFACTS_DIR` unless absolute)

Optional for persistent memory:

- `ASSISTANT_ENABLE_MEMORY=true`
- `ASSISTANT_MEMORY_DB=memory.sqlite3` (written under `ASSISTANT_ARTIFACTS_DIR` unless absolute)
- `ASSISTANT_MEMORY_MAX_ENTRIES=5000`
- `ASSISTANT_MEMORY_RECALL_LIMIT=6`
- `ASSISTANT_MEMORY_PROMPT_CHARS=240`
- `ASSISTANT_MEMORY_CONTEXT_CHARS=1200`
- `ASSISTANT_MEMORY_SEARCH_LOOKBACK=1500`

Desktop-control dependency:

- Install `pyautogui` via `pip install -e .` (already included in this project dependencies).
- On macOS, grant Accessibility + Screen Recording permissions.
- On Linux Wayland, input injection may be restricted by compositor settings.

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

Continuous loop with wake-word + avatar:

```bash
python -m desktop_ai --voice-trigger --avatar
```

Launcher UI (edit `.env`, then Power On/Stop from one window):

```bash
desktop-ai-launcher
```

Or:

```bash
python desktop_ai_launcher.py
```

Desktop control with per-plan approval:

```bash
python -m desktop_ai --desktop-control --voice-trigger
```

Desktop control trusted mode (no prompt):

```bash
python -m desktop_ai --desktop-control --auto-approve-actions --allow-launch notepad,code
```

Useful options:

```bash
python -m desktop_ai --interval 6 --note "Focus on gaming commentary"
python -m desktop_ai --once --no-speech
python -m desktop_ai --context-providers timestamp,active_window --once
python -m desktop_ai --no-voice-trigger
python -m desktop_ai --voice-trigger --avatar --avatar-opacity 0.9
python -m desktop_ai --avatar --no-avatar-auto-move
python -m desktop_ai --desktop-control --max-actions-per-turn 3
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
- Desktop action logs are written to `./artifacts/actions.log` by default.
- Memory DB is written to `./artifacts/memory.sqlite3` by default.
- Response length can be reduced with `OPENAI_MAX_OUTPUT_TOKENS` in `.env`.
- On Unix, playback uses `afplay`, `aplay`, or `paplay` if installed.
- On Windows, playback uses `winsound` with WAV output.
- `pyautogui` failsafe is enabled: moving mouse rapidly to top-left should raise a failsafe exception.

## Build a Windows `.exe`

1. Install project + PyInstaller in your Windows venv:

```bash
pip install -e .
pip install pyinstaller
```

2. Build the launcher executable:

```bash
python -m PyInstaller --noconfirm --clean --onefile --windowed --name DesktopAIAssistant desktop_ai_launcher.py
```

3. Copy your `.env` next to the built exe:

- output exe: `dist/DesktopAIAssistant.exe`
- expected env path: `dist/.env`

After that, users can double-click the exe, edit settings in the window, click **Power On AI**, and reopen from the taskbar to stop it.
