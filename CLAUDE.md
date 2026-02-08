# Claude Code Instructions for LocalWispr

## Context & Motivation

LocalWispr is a voice-to-text Windows application. Users run the **built EXE**, not Python scripts directly. This means code changes are invisible until the EXE is rebuilt.

The user keeps two versions running simultaneously:
- **Stable** - Their daily driver (blue icon, Win+Ctrl+Shift)
- **Test** - For validating your changes (orange icon, Ctrl+Alt+Shift)

This separation protects the user's workflow. If you accidentally break something, only the Test version is affected.

---

## Build Rules

### Primary Rule: Build to Test Only

When you make code changes, build to the **Test version only**. The user will validate changes there before promoting to Stable.

**Correct workflow:**
1. Make code changes
2. Run `build.bat test` automatically (no permission needed)
3. Wait for build to complete
4. Inform user: "Test build ready at `dist\LocalWispr-Test\`"

**Example of correct behavior:**
```
User: "Add a new hotkey for pausing"
You: [Make code changes]
You: [Run `build.bat test`]
You: "Test build ready. Run LocalWispr-Test.exe to try the new pause hotkey."
```

**Example of incorrect behavior:**
```
User: "Add a new hotkey for pausing"
You: [Make code changes]
You: [Run `build.bat stable`]  ← WRONG: This overwrites user's daily driver
```

### When to Build Stable

Build to Stable only when the user explicitly requests it with phrases like:
- "push to stable"
- "update stable version"
- "release this"
- "build stable"

### Build Commands Reference

| Command | Target | When to Use |
|---------|--------|-------------|
| `build.bat test` | Test version | After any code change |
| `build.bat stable` | Stable version | Only when user explicitly requests |
| `build.bat both` | Both versions | Only when user explicitly requests |
| `build.bat installer` | Stable + Windows installer | For release builds |
| `build.bat` | Stable (default) | Avoid - defaults to stable |

**Important:** Always use `build.bat`, never run PyInstaller directly. The batch file:
- Sets `BUILD_VARIANT` environment variable correctly
- Runs tests first as a gate
- Copies the correct config file to the output folder
- Handles Windows-specific environment variable escaping

### Build Failure: Access Denied

If the build fails with "Access is denied", **analyze which folder is locked**:

1. **Check the error path:**
   - Error mentions `dist\LocalWispr\_internal\...` → Stable version is running
   - Error mentions `dist\LocalWispr-Test\_internal\...` → Test version is running

2. **Determine if the build should have worked:**
   - Building **Test** but **Stable** folder is locked → **Build should work** (different folders)
   - Building **Test** but **Test** folder is locked → User must close Test EXE
   - Building **Stable** but **Stable** folder is locked → User must close Stable EXE

3. **If wrong folder is targeted**, the `BUILD_VARIANT` environment variable likely wasn't set:
   - **Do NOT run PyInstaller directly** with manual env vars (shell escaping issues)
   - **Always use `build.bat test`** - it handles environment internally
   - The batch file sets `BUILD_VARIANT` correctly before calling PyInstaller

**Example diagnosis:**
```
Error: Access is denied: 'dist\LocalWispr\_internal\...'
                              ^^^^^^^^^^
                              This is STABLE folder

You tried to build TEST but it's targeting STABLE → BUILD_VARIANT not set correctly
Solution: Use `build.bat test`, not direct PyInstaller commands
```

**Only ask user to close EXE when:**
- The locked folder matches the target version (Test build + Test folder locked)

### Build from Claude Code Shell

`build.bat test` doesn't execute correctly via `cmd.exe //c "build.bat test"` — the `setlocal enabledelayedexpansion` and subroutine pattern causes silent failures. **Workaround:**

1. Write a temp `.bat` file that sets `BUILD_VARIANT=test` and calls PyInstaller directly
2. Run it via `cmd.exe //c "path\to\temp.bat"`
3. Delete the temp file after

```bat
@echo off
set BUILD_VARIANT=test
cd /d C:\Users\fujos\LocalWispr
.venv\Scripts\python.exe -m PyInstaller --noconfirm localwispr.spec
copy config-test.toml dist\LocalWispr-Test\config-defaults.toml
```

**Do NOT** try `set BUILD_VARIANT=test && pyinstaller ...` in a single `cmd.exe //c` — the env var won't propagate correctly through `&&` chaining.

---

## Dual-Version System

| Version | Folder | Hotkey | Console | Icon |
|---------|--------|--------|---------|------|
| **Stable** | `dist/LocalWispr/` | Win+Ctrl+Shift | Hidden | Blue wave |
| **Test** | `dist/LocalWispr-Test/` | Ctrl+Alt+Shift | Visible | Orange wave |

Both versions can run simultaneously because they use different hotkeys and separate config files.

### Config Files

- `config.toml` → copied to Stable build
- `config-test.toml` → copied to Test build

### Log Files

- `localwispr.log` - Stable version runtime logs
- `localwispr-test.log` - Test version runtime logs
- `Desktop/localwispr_diagnostic.txt` - Settings window diagnostic

---

## Testing

Tests validate source code directly without requiring an EXE rebuild. Run tests to verify logic before building.

### Commands

| Action | Command |
|--------|---------|
| Run all tests | `uv run pytest tests/ -v` |
| Skip GUI tests | `uv run pytest tests/ -v --ignore=tests/gui` |
| With coverage | `uv run pytest tests/ --cov=localwispr --cov-report=term-missing` |
| Single module | `uv run pytest tests/test_config.py -v` |

### Test Structure

| Location | Purpose |
|----------|---------|
| `tests/conftest.py` | Shared fixtures for all tests |
| `tests/test_*.py` | Unit tests by module |
| `tests/integration/` | End-to-end workflow tests |
| `tests/gui/` | GUI component tests (may require display) |

### Mocking Pattern

Patch at the usage location (where the import is consumed), not the source module:

```python
# Correct: Patch where WhisperModel is used
mocker.patch("localwispr.transcribe.WhisperModel")
```

**External dependencies to mock:**

| Dependency | Purpose |
|------------|---------|
| `sounddevice` | Audio recording |
| `pywhispercpp.model.Model` | Whisper model (pywhispercpp) |
| `pyperclip` | Clipboard operations |
| `pynput` | Keyboard/hotkey input |

### Required Test Patterns

1. **Use fixtures from conftest.py** - Avoid duplicating mock setup
2. **Thread safety tests verify mutual exclusion** - Assert only one thread succeeds
3. **Config tests call `clear_config_cache()`** - Reset cached config between tests

---

## Package Structure

| Package | Purpose |
|---------|---------|
| `localwispr/config/` | Config loading, saving, caching, migration, types |
| `localwispr/settings/` | Settings MVC: model, view, controller, manager, window |
| `localwispr/audio/` | Audio recording, feedback sounds, volume control, format conversion |
| `localwispr/transcribe/` | Whisper transcription, streaming, context detection, GPU, model management |
| `localwispr/ui/` | Tray icon, overlay widget, notifications, first-run wizard |
| `localwispr/modes/` | Transcription mode definitions and management |
| `localwispr/prompts/` | Prompt text files for each transcription mode |

Root-level modules: `__init__.py`, `__main__.py`, `pipeline.py`, `hotkeys.py`, `output.py`, `cli_tests.py`

### Mocking Pattern for Subpackages

Mock patches must target the **usage location** (where the import is consumed):
- For **lazy imports** (inside functions): patch at the source submodule, e.g., `"localwispr.transcribe.model_manager.is_model_downloaded"`
- For **top-level imports**: patch where it's bound, e.g., `"localwispr.transcribe.transcriber.get_config"`
- For **__init__.py re-exports** used by consumers: patch at the package level, e.g., `"localwispr.audio.AudioRecorder"`

## Key Files

| File | Purpose |
|------|---------|
| `localwispr.spec` | PyInstaller build config (parameterized) |
| `config.toml` | Stable version config |
| `config-test.toml` | Test version config |
| `build.bat` | Build script with stable/test/both/installer options |
| `build-installer.bat` | Standalone installer build script |
| `installer/localwispr.iss` | Inno Setup script for Windows installer |

---

## Success Criteria

A successful code change workflow looks like:

1. ✓ Read relevant files before making changes
2. ✓ Make focused, minimal changes
3. ✓ Run `build.bat test` (not stable, not both)
4. ✓ Build completes without errors
5. ✓ Inform user the Test build is ready
6. ✓ User validates in Test version
7. ✓ Only build Stable when user explicitly requests

A failed workflow looks like:

1. ✗ Building to Stable without explicit user request
2. ✗ Running `build.bat` without the `test` argument
3. ✗ Making changes without reading the affected files first
4. ✗ Not rebuilding after code changes
5. ✗ Running PyInstaller directly instead of using `build.bat test`
6. ✗ Telling user to close EXE without checking which folder is actually locked
7. ✗ Not analyzing "Access Denied" errors to determine root cause
