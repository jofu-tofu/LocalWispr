# Installing LocalWispr

## System Requirements
- Windows 10 (build 17763) or later
- 4 GB RAM minimum (8 GB recommended)
- 500 MB disk space for application
- Additional 150-3000 MB for speech recognition model

## Installation Steps

### 1. Download the Installer
Go to [Releases](https://github.com/jofu-tofu/LocalWispr/releases/latest) and download `LocalWispr-Setup-X.X.X.exe`.

### 2. Run the Installer
1. Double-click the downloaded file
2. Windows may show a security warning:
   - Click "More info"
   - Click "Run anyway"
   - This warning appears because the installer isn't digitally signed (signing certificates cost money)
3. Choose installation location (default: `C:\Program Files\LocalWispr`)
4. Optional: Check "Start LocalWispr when Windows starts"
5. Click "Install"

### 3. First Launch
1. LocalWispr runs in your system tray (look for the blue wave icon near the clock)
2. Right-click the tray icon → Settings
3. Go to the Model tab
4. Click "Download" next to a model:
   - **Base (recommended)**: 145 MB, good accuracy, fast
   - **Small**: 488 MB, better accuracy, slower
   - **Medium**: 1.5 GB, best accuracy, slowest
5. Wait for the download to complete (progress bar will show status)
6. Close Settings

### 4. Start Using LocalWispr
1. Click in any text field (Notepad, Word, browser, etc.)
2. Press **Win + Ctrl + Shift** (hold all three keys)
3. Speak while holding the keys
4. Release the keys when done speaking
5. Your speech will be transcribed and inserted at the cursor

## Tips

- **Change the hotkey**: Settings → Hotkeys tab
- **Enable streaming mode**: Transcribe as you speak (Settings → Features)
- **Paste vs Type**: Choose how text is inserted (Settings → Features)
- **Multiple models**: Download different models for different use cases

## Troubleshooting

### "Windows protected your PC" warning
This is expected. LocalWispr isn't signed with a code signing certificate.
- Click "More info" → "Run anyway"

### Hotkey doesn't work
- **No model downloaded**: Go to Settings → Model tab → Download a model
- **Hotkey conflict**: Another app may use the same shortcut. Change it in Settings → Hotkeys
- **App not running**: Check system tray for the blue wave icon

### No text appears after recording
- **Check the model**: Settings → Model tab → Ensure a model is downloaded and selected
- **Check paste method**: Settings → Features → Try switching between "Paste" and "Type"
- **Check logs**: `%LOCALAPPDATA%\LocalWispr\localwispr.log`

### "This app can't run on your PC"
- **Windows version**: LocalWispr requires Windows 10 build 17763 or later
- **Architecture**: LocalWispr requires 64-bit Windows

### Model download fails
- **Check internet connection**: Models are downloaded from HuggingFace
- **Check disk space**: Base model needs 150 MB, Small needs 500 MB, Medium needs 1.5 GB
- **Firewall**: Ensure LocalWispr can access the internet

### High CPU usage
- **Large model**: Medium model uses more CPU. Try Base or Small model instead
- **Streaming mode**: Real-time transcription uses more CPU than processing after release

## Uninstalling

1. Close LocalWispr (right-click tray icon → Exit)
2. Go to Settings → Apps → Installed apps
3. Find "LocalWispr" in the list
4. Click the three dots → Uninstall
5. Follow the uninstaller prompts

## Getting Help

- **GitHub Issues**: https://github.com/jofu-tofu/LocalWispr/issues
- **Check logs**: `%LOCALAPPDATA%\LocalWispr\localwispr.log`
- **Settings diagnostics**: Settings → About → View diagnostic log (saved to Desktop)
