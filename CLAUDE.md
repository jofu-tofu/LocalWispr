# Claude Code Instructions for LocalWispr

## Context

LocalWispr is a voice-to-text Windows application. Users run the **built EXE**, not Python scripts directly. This means code changes are invisible until rebuilt.

## Building the EXE

**CRITICAL: Always rebuild BEFORE asking the user to test.** Code changes have no effect until the EXE is rebuilt. After making changes:

1. Run `build.bat test` (or the manual build commands below)
2. Wait for build to complete successfully
3. THEN ask the user to test

If build fails with "Access is denied", the user needs to close the running EXE first.

### Dual-Version Build System

LocalWispr supports two build variants that can run simultaneously:

| Version | Folder | Hotkey | Console | Icon |
|---------|--------|--------|---------|------|
| **Stable** | `dist/LocalWispr/` | Win+Ctrl+Shift | Hidden | Blue wave |
| **Test** | `dist/LocalWispr-Test/` | Ctrl+Alt+Shift | Visible | Orange wave |

### Build Commands

```bash
# Build stable version (default)
build.bat
build.bat stable

# Build test version (for iterating on changes)
build.bat test

# Build both versions
build.bat both
```

### Typical Workflow

1. Make code changes
2. Run `build.bat test` to build test version
3. User runs `dist\LocalWispr-Test\LocalWispr-Test.exe` alongside stable
4. Iterate until working
5. Run `build.bat stable` when ready to release

### What Each Build Does

1. Runs all tests (fails build if tests fail)
2. Runs PyInstaller to create EXE
3. Copies appropriate config file:
   - Stable: `config.toml` → `dist/LocalWispr/config.toml`
   - Test: `config-test.toml` → `dist/LocalWispr-Test/config.toml`

### Build Behavior

- User runs the built EXE, so changes require rebuild to take effect
- Test version shows visible console for debugging
- Both versions run simultaneously with different hotkeys
- Separate log files: `localwispr.log` (stable) vs `localwispr-test.log` (test)

### Key Files

| File | Purpose |
|------|---------|
| `localwispr/*.py` | Source code |
| `localwispr.spec` | PyInstaller build config (parameterized) |
| `config.toml` | Stable version config (Win+Ctrl+Shift) |
| `config-test.toml` | Test version config (Ctrl+Alt+Shift) |
| `build.bat` | Build script with stable/test/both options |
| `dist/LocalWispr/` | Stable build output |
| `dist/LocalWispr-Test/` | Test build output |

### Diagnostic Locations

- `Desktop/localwispr_diagnostic.txt` - Settings window diagnostic
- `localwispr.log` - Stable version runtime logs
- `localwispr-test.log` - Test version runtime logs

---

## Testing

Tests validate source code directly without requiring an EXE rebuild.

### Commands

| Action | Command |
|--------|---------|
| Run all tests | `uv run pytest tests/ -v` |
| With coverage | `uv run pytest tests/ --cov=localwispr --cov-report=term-missing` |
| Single module | `uv run pytest tests/test_config.py -v` |

### Test Structure

| Location | Purpose |
|----------|---------|
| `tests/conftest.py` | Shared fixtures for all tests |
| `tests/test_*.py` | Unit tests by module |
| `tests/integration/` | End-to-end workflow tests |
| `tests/gui/` | GUI component tests (may require display) |

### Mocking

Patch at usage location (where the import is consumed) rather than the source module. This ensures patches affect already-imported references.

```python
# Patch where WhisperModel is used
mocker.patch("localwispr.transcribe.WhisperModel")
```

**External dependencies to mock:**

| Dependency | Purpose |
|------------|---------|
| `sounddevice` | Audio recording |
| `faster_whisper` | Whisper model |
| `pyperclip` | Clipboard operations |
| `pynput` | Keyboard/hotkey input |

### Required Patterns

1. **Use fixtures from conftest.py** - Avoid duplicating mock setup
2. **Thread safety tests verify mutual exclusion** - Assert only one thread succeeds, not just that all complete
3. **Config tests call `clear_config_cache()`** - Reset cached config between tests

### When to Run

| Trigger | Action |
|---------|--------|
| Before committing | Run full test suite |
| After modifying core modules | Run relevant test file |
