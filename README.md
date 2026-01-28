# LocalWispr

**Privacy-first voice-to-text for Windows.** Press a hotkey to capture speech and insert text into any application. All processing happens locally - no cloud APIs, no internet required after setup.

## Quick Install

1. Download the installer from [latest release](https://github.com/jofu-tofu/LocalWispr/releases/latest)
2. Run `LocalWispr-Setup-X.X.X.exe`
3. Open Settings (tray icon) to download a speech model (Base recommended)
4. Press **Win + Ctrl + Shift** to record and transcribe

See [INSTALLATION.md](INSTALLATION.md) for detailed instructions and troubleshooting.

## Features

- **Hotkey Activation**: Configurable keybind starts/stops voice capture
- **Universal Input**: Works in any application (Word, browsers, chat apps, etc.)
- **Local Processing**: Privacy-focused speech recognition via faster-whisper (no cloud, no internet)
- **Visual Feedback**: Floating overlay shows recording status
- **Multiple Models**: Choose speed vs accuracy (Base, Small, Medium)
- **Streaming Mode**: See transcription in real-time as you speak
- **Flexible Input**: Paste or type text into applications

## For Developers

### Building from Source

LocalWispr uses Python 3.11+ with the UV package manager and PyInstaller for building standalone executables.

**Quick start:**
```bash
# Install dependencies
uv sync

# Run from source
uv run python -m localwispr

# Run tests
uv run pytest tests/ -v --ignore=tests/gui

# Build standalone EXE
build.bat test
```

### Build System

Two build variants exist for development and production:

| Command | Output | Use Case |
|---------|--------|----------|
| `build.bat test` | `dist/LocalWispr-Test/` | Development iteration (orange icon, Ctrl+Alt+Shift) |
| `build.bat stable` | `dist/LocalWispr/` | Production release (blue icon, Win+Ctrl+Shift) |
| `build.bat installer` | `dist/LocalWispr-Setup-X.X.X.exe` | Windows installer |

Both versions can run simultaneously for testing changes without affecting your daily workflow.

See [CLAUDE.md](CLAUDE.md) for complete build instructions, testing guidelines, and development workflows.

### Tech Stack

- **Runtime**: Python 3.11+ with uv package manager
- **Speech Recognition**: faster-whisper (local Whisper implementation)
- **Hotkey System**: pynput for global keyboard hooks
- **UI**: PySide6 for system tray and settings window
- **Build**: PyInstaller for standalone EXE, Inno Setup for installer

### Configuration

Settings stored in `config.toml` (or `config-test.toml` for test builds):

| Setting | Description | Default |
|---------|-------------|---------|
| `hotkey` | Recording activation keybind | `win+ctrl+shift` |
| `language` | Speech recognition language | `en` |
| `model_size` | Whisper model (tiny, base, small, medium, large) | `base` |
| `streaming` | Real-time transcription | `false` |

## License

MIT
