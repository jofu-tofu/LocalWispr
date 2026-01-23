# LocalWispr

Local voice-to-text for Windows. Press a hotkey to capture speech and insert text into any focused application.

## Features

- **Hotkey Activation**: Configurable keybind starts/stops voice capture
- **Universal Input**: Text types into the currently focused window
- **Local Processing**: Privacy-focused speech recognition via faster-whisper
- **Visual Feedback**: Floating overlay indicates recording status
- **Mode Support**: Multiple transcription modes (dictation, chat, email)

## Tech Stack

- **Runtime**: Python 3.11+ with uv package manager
- **Speech Recognition**: faster-whisper (local Whisper implementation)
- **Hotkey System**: pynput for global keyboard hooks
- **UI**: PySide6 for system tray and settings window
- **Build**: PyInstaller for standalone EXE

## Quick Start

```bash
# Install dependencies
uv sync

# Run from source
uv run python -m localwispr

# Build standalone EXE
build.bat
```

## Configuration

Settings stored in `config.toml`:

| Setting | Description | Default |
|---------|-------------|---------|
| `hotkey` | Recording activation keybind | `win+ctrl+shift` |
| `language` | Speech recognition language | `en` |
| `model_size` | Whisper model (tiny, base, small, medium, large) | `base` |

## Development

See `CLAUDE.md` for build instructions and testing guidelines.

## License

MIT
