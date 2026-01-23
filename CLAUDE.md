# Claude Code Instructions for LocalWispr

## Building the EXE

**IMPORTANT**: When testing ANY code changes, you MUST rebuild the EXE before asking the user to test.

### Build Command
```bash
cd /c/Users/fujos/LocalWispr && .venv/Scripts/python.exe -m PyInstaller --noconfirm localwispr.spec && cp config.toml dist/LocalWispr/
```

### What This Does
1. Runs PyInstaller to create `dist/LocalWispr/LocalWispr.exe`
2. Copies `config.toml` to `dist/LocalWispr/` (next to EXE)

### Why This Matters
- The user runs the **built EXE**, not the Python script
- Code changes don't take effect until rebuilt
- The EXE reads `config.toml` from its own directory (`dist/LocalWispr/`)

### Testing Workflow
1. Make code changes
2. **Rebuild the EXE** (command above)
3. Ask user to run `dist\LocalWispr\LocalWispr.exe`
4. Check logs/diagnostics

### Key Files
| File | Purpose |
|------|---------|
| `localwispr/*.py` | Source code |
| `localwispr.spec` | PyInstaller build config |
| `config.toml` | Source config (copied to dist on build) |
| `dist/LocalWispr/` | Built application folder |
| `dist/LocalWispr/config.toml` | Config the EXE actually reads |

### Diagnostic Locations
- `Desktop/localwispr_diagnostic.txt` - Settings window diagnostic
- `localwispr.log` - Runtime logs (in CWD or next to EXE)

---

## Testing

Tests run against source code directly and do NOT require an EXE rebuild.

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

**Patch at usage location, not source.** Patching at source doesn't affect already-imported references.

```python
# Correct
mocker.patch("localwispr.transcribe.WhisperModel")

# Wrong
mocker.patch("faster_whisper.WhisperModel")
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
