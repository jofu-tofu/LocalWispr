# Test Review for test_pipeline.py

## Executive Summary

The test file contains several issues where tests don't adequately verify their stated behavior. These range from trivial passes to misaligned mock configurations that allow bugs to slip through.

---

## Issues Found

### 1. ✗ `test_pipeline_start_recording` - Missing Actual Start Verification

**Line:** 26-37

**Test Name Claims:** "Test starting recording"

**What It Actually Tests:**
- Only verifies that `mock_audio_recorder.start_recording()` is called
- Does NOT verify that `pipeline.is_recording` becomes `True`
- Does NOT verify that the recorder instance is stored in `pipeline._recorder`

**Problem:**
The test passes trivially because the fixture automatically patches `AudioRecorder` globally. The test never asserts on the key outcome: that the pipeline's `is_recording` property returns `True`. This means a bug where `start_recording()` succeeds but doesn't set internal state would not be caught.

**Fix Required:**
```python
result = pipeline.start_recording()
assert result is True
mock_audio_recorder.start_recording.assert_called_once()
assert pipeline.is_recording is True  # ← MISSING
```

---

### 2. ✗ `test_pipeline_stop_and_transcribe_no_audio` - Contradictory Mock Setup

**Line:** 77-96

**Test Name Claims:** "Test stop_and_transcribe with no audio captured"

**What It Actually Tests:**
- Sets `mock_recorder.is_recording = False` (line 80)
- Then expects error "No audio" OR "not recording" (line 96)

**Problem:**
The test setup contradicts the test name. If the test is about "no audio captured" (audio is empty), the recorder should be `is_recording = True` with empty audio returned. Instead, the test is actually verifying the "not recording" error path, not the "no audio" path.

The code path at pipeline.py:387-389 checks `if self._recorder is None or not self._recorder.is_recording`, which is what this test actually exercises, not audio duration checking.

**Fix Required:**
Either rename the test or change the setup:
```python
# Option A: Test the actual "no audio" path
mock_recorder.is_recording = True  # Still recording
mock_recorder.get_whisper_audio.return_value = np.zeros(10, dtype=np.float32)  # < 0.1s
result = pipeline.stop_and_transcribe()

# Option B: Rename and clarify
# def test_pipeline_stop_and_transcribe_not_recording
```

---

### 3. ✗ `test_pipeline_preload_model_async` - No Verification of Model Actually Loading

**Line:** 55-75

**Test Name Claims:** "Test async model preloading"

**What It Actually Tests:**
- Mocks `WhisperTranscriber` globally
- Calls `preload_model_async()`
- Waits for the event and checks `is_model_ready is True`

**Problem:**
The test doesn't verify that the model was actually loaded or accessed. It only checks that an event was set. The mock setup doesn't even expose what the transcriber did. If `_preload_model()` had a bug and never actually instantiated `WhisperTranscriber`, this test would still pass.

**Real-World Implication:**
A regression where the background thread fails silently (exception handling is generic) would not be caught. The test assumes the happy path without validating the actual work.

**Fix Required:**
```python
mock_transcriber = MagicMock()
mock_transcriber.model = MagicMock()  # Simulate model property access
mocker.patch(
    "localwispr.transcribe.WhisperTranscriber",
    return_value=mock_transcriber,
)

pipeline.preload_model_async()
pipeline._model_preload_complete.wait(timeout=2.0)

assert pipeline.is_model_ready is True
assert pipeline._transcriber is not None  # ← MISSING
mock_transcriber.model  # Verify model property was accessed  # ← MISSING
```

---

### 4. ✗ `test_pipeline_model_timeout` - Invalid Timeout Behavior

**Line:** 167-188

**Test Name Claims:** "Test pipeline handles model load timeout"

**What It Actually Tests:**
- Sets `MODEL_LOAD_TIMEOUT = 0.01` (10ms) - absurdly short
- Calls `stop_and_transcribe()` WITHOUT calling `preload_model_async()`
- Expects a timeout error

**Problem:**
The timeout is set on the pipeline instance, but the model preload is never initiated. The code path being tested (pipeline.py:407-411) waits for `_model_preload_complete.is_set()`. Since preload was never started, the event is never set, so the code hits the timeout as expected.

However, this is an **unrealistic scenario**. In normal operation:
1. `preload_model_async()` is called at startup
2. `stop_and_transcribe()` waits for it using the normal timeout (60s)

The test validates timeout mechanics but not how timeouts integrate with the real preload flow.

**Real Issue:**
If the model takes exactly 61 seconds to load, the real app would fail, but this test wouldn't catch regressions in timeout handling for realistic scenarios.

**Fix Required:**
```python
# Test the real scenario: preload is slow
import threading

def slow_preload(*args, **kwargs):
    time.sleep(2)  # Preload takes 2 seconds

mocker.patch("localwispr.transcribe.WhisperTranscriber", side_effect=slow_preload)

pipeline.MODEL_LOAD_TIMEOUT = 0.01  # 10ms timeout
pipeline.preload_model_async()

result = pipeline.stop_and_transcribe()
assert result.success is False
assert "timeout" in result.error.lower()
```

---

### 5. ✗ `test_pipeline_on_error_callback` - Wrong Exception Type

**Line:** 190-210

**Test Name Claims:** "Test that on_error callback is invoked on failure"

**What It Actually Tests:**
- Patches `AudioRecorder` to raise `Exception("device error")`
- Calls `start_recording()`
- Verifies `on_error` callback was called

**Problem:**
The patch is incorrect. The test patches `AudioRecorder` constructor globally to raise, but pipeline.py:209-220 instantiates `AudioRecorder` inside a try-except that catches `Exception`. The mock exception gets caught and `on_error("Failed to start recording")` is called correctly.

However, the test description says "on failure" but doesn't verify WHICH failure case it's testing. It passes by accident because exception handling is broad. A more specific test would verify the callback receives the correct error message or that it's called with specific context.

**Real Issue:**
The test validates the mechanism but not the specifics. It's testing exception handling generally, not "device error handling specifically."

**Better Test:**
```python
# Make it clear: testing exception during recorder init
mocker.patch(
    "localwispr.audio.AudioRecorder",
    side_effect=Exception("device error"),
)

error_callback = MagicMock()
pipeline = RecordingPipeline(mode_manager=mode_manager, on_error=error_callback)

result = pipeline.start_recording()

assert result is False
error_callback.assert_called_once_with("Failed to start recording")
# Optionally assert on the exact message, not just that it was called
```

---

### 6. ⚠ `test_stale_generation_skips_callback` - Executor Implementation Leaks Details

**Line:** 397-454

**Test Name Claims:** "Verify stale transcriptions don't invoke callbacks"

**What It Actually Tests:**
- Creates a custom `StaleExecutor` that increments generation before completing
- Submits async work
- Verifies callbacks are NOT invoked

**Problem:**
The test is deeply coupled to implementation details. The custom executor (line 415-436) intrudes into internal pipeline state (`_generation_lock`, `_current_generation`) to simulate staleness. While the test does validate the desired behavior, it requires detailed knowledge of the pipeline's internal generation tracking mechanism.

A better approach would test via the public API (e.g., calling `stop_and_transcribe_async()` twice rapidly), but the current approach works and is acceptable for testing the generation mechanism itself.

**Verdict:** This test is actually OK but fragile. It works because it exercises the intended code path, but any refactoring of generation tracking would break it unnecessarily.

---

### 7. ✓ `test_pipeline_thread_safety` - Good Test

**Line:** 212-271

This test is well-designed. It verifies mutual exclusion by:
- Running 5 threads calling `start_recording()` simultaneously
- Checking only 1 succeeds and 4 fail
- Verifying only 1 `AudioRecorder` instance is created

This is a proper concurrency test with assertions that matter.

---

### 8. ⚠ `test_pipeline_cancel_recording` - Incomplete Verification

**Line:** 130-144

**Test Name Claims:** "Test canceling a recording"

**What It Actually Tests:**
- Sets `mock_audio_recorder.is_recording = True`
- Calls `cancel_recording()`
- Verifies `stop_recording()` was called

**Problem:**
The test doesn't verify that the audio is actually discarded or that the pipeline state is reset. In particular:
- Doesn't check `pipeline.is_recording` becomes `False` after cancel
- Doesn't verify recording state is properly cleared

**Fix Required:**
```python
mock_audio_recorder.is_recording = True
pipeline._recorder = mock_audio_recorder

pipeline.cancel_recording()

mock_audio_recorder.stop_recording.assert_called_once()
assert pipeline.is_recording is False  # ← MISSING
```

---

### 9. ✓ `test_pipeline_mute_system_audio` - Good Test

**Line:** 273-295

This test properly validates that `mute_system()` is called when the flag is set. Clear and focused.

---

### 10. ✓ `test_stop_and_transcribe_async_success` - Good Test

**Line:** 301-361

Properly tests async flow with callbacks. Verifies:
- Callbacks are invoked with correct data
- Generation ID is returned
- Result contains expected transcription

---

### 11. ⚠ `test_no_audio_restores_system_mute` - Suspicious Test Logic

**Line:** 571-609

**Test Name Claims:** "Verify system audio restored even when no audio captured"

**What It Actually Tests:**
- Sets `is_recording = False` (line 576)
- Calls `start_recording(mute_system=True)` (line 596)
- Then sets `is_recording = False` again (line 600)
- Calls `stop_and_transcribe_async(mute_system=True)` (line 603-604)
- Asserts `restore_mute_state` was called (line 609)

**Problem:**
Line 600 is redundant - it's already `False`. The test setup is confusing. The real test should be:
1. Start recording WITH mute
2. Never call stop_recording (leave it in recording state)
3. Call stop_and_transcribe (which detects "not recording" and errors)
4. Verify restore was called despite the error

The current test doesn't clearly demonstrate the "restore on error" path.

**Fix Required:**
```python
mock_recorder = MagicMock()
mock_recorder.is_recording = False  # Not recording
mocker.patch("localwispr.audio.AudioRecorder", return_value=mock_recorder)

mock_mute = mocker.patch("localwispr.volume.mute_system", return_value=True)
mock_restore = mocker.patch("localwispr.volume.restore_mute_state")

pipeline = RecordingPipeline(mode_manager=mode_manager, executor=sync_executor)

# Try to stop with mute flag when not recording
results = []
pipeline.stop_and_transcribe_async(
    mute_system=True,
    on_result=lambda r, g: results.append(r),
)

# Should restore even though no audio
mock_restore.assert_called_once_with(True)
assert len(results) == 1
assert results[0].success is False
```

---

### 12. ✓ `test_is_current_generation` - Good Test

**Line:** 611-635

Properly validates generation tracking across different states. Clear assertions.

---

## Summary Table

| Test | Status | Issue |
|------|--------|-------|
| `test_pipeline_initialization` | ✓ | OK |
| `test_pipeline_start_recording` | ✗ | Missing `pipeline.is_recording` assertion |
| `test_pipeline_start_recording_already_recording` | ✓ | OK |
| `test_pipeline_preload_model_async` | ✗ | No verification of actual model load |
| `test_pipeline_stop_and_transcribe_no_audio` | ✗ | Test name mismatches actual behavior |
| `test_pipeline_stop_and_transcribe_success` | ✓ | OK |
| `test_pipeline_cancel_recording` | ⚠ | Incomplete verification of state reset |
| `test_pipeline_get_rms_level` | ✓ | OK |
| `test_pipeline_model_timeout` | ✗ | Invalid/unrealistic scenario |
| `test_pipeline_on_error_callback` | ⚠ | Vague about which error case |
| `test_pipeline_thread_safety` | ✓ | Good test |
| `test_pipeline_mute_system_audio` | ✓ | Good test |
| `test_stop_and_transcribe_async_success` | ✓ | Good test |
| `test_stop_and_transcribe_async_no_audio` | ✓ | OK |
| `test_stale_generation_skips_callback` | ⚠ | Fragile; implementation-coupled |
| `test_shutdown_skips_callbacks` | ✓ | OK |
| `test_callback_exception_doesnt_crash` | ✓ | OK |
| `test_no_audio_restores_system_mute` | ⚠ | Confusing test setup |
| `test_is_current_generation` | ✓ | Good test |

**Summary:** 5 clear failures/misalignments, 4 warnings (incomplete or fragile), 10 good tests

---

## Recommendations

1. **High Priority:** Fix tests #2, #3, #5 - they have fundamental assertion gaps
2. **Medium Priority:** Improve tests #1, #8, #11 - verify state transitions, not just method calls
3. **Low Priority:** Document fragile tests like #6 and consider refactoring to use public API
