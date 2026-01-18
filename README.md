# LocalWispr

A lightweight local voice-to-text tool for Windows. Press a keybind to capture speech and insert text into any focused application.

## Features

- **Hotkey Activation**: Press a configurable keybind to start/stop voice capture
- **Universal Input**: Text is typed into whatever window is currently focused
- **Local Processing**: Privacy-focused with local speech recognition
- **Customization UI**: Configure hotkeys, language, and behavior

## Tech Stack

- **Runtime**: Bun
- **UI Framework**: Electron (lightweight)
- **Speech Recognition**: Whisper.cpp (local) or Web Speech API
- **Hotkey System**: Native Windows API bindings

## Getting Started

```bash
# Install dependencies
bun install

# Run in development
bun run dev

# Build for production
bun run build
```

## Configuration

Settings are stored in `config.json`:

- `hotkey`: The keybind to activate recording (default: `Ctrl+Shift+Space`)
- `language`: Speech recognition language (default: `en-US`)
- `model`: Whisper model size (default: `base`)

## License

MIT
