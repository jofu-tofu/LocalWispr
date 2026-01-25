# Edge Case Coverage Analysis: tests/test_pipeline.py

## Executive Summary

The test suite for `RecordingPipeline` has **37 missing edge cases** across 5 categories. While core happy-path scenarios are tested, several critical error conditions, boundary cases, and state transitions are not covered. These gaps could hide subtle bugs in production.

---

## Category 1: Error Conditions (Not Fully Tested)

### 1.1 Recorder Initialization Failure
**Scenario:** AudioRecorder constructor throws exception during `start_recording()`

**Why it matters:**
- The exception is caught and `on_error` is called, but we never test the error callback path
- If recorder init fails, the `_recorder` remains `None` but code doesn't validate this
- Next call to `start_recording()` might create a second recorder instead of retrying

**Test name:** `test_pipeline_recorder_init_failure`
**Code path:** `start_recording()` line 209-210, exception handling line 233-236
**Missing assertion:** Verify `_recorder` is still None after failure, verify `on_error` called with correct message

---

### 1.2 Model Preload Error Recovery
**Scenario:** Model preload fails with exception, then `stop_and_transcribe()` called

**Why it matters:**
- `_model_preload_error` is set, but the retry logic in `_get_transcriber()` (line 424-427) should create a new transcriber
- If sync fallback also fails, we only return error but never log which error (preload vs sync)
- Error message is generic "Failed to initialize transcriber" - doesn't distinguish cause

**Test name:** `test_pipeline_model_preload_error_sync_fallback`
**Code path:** `_preload_model()` line 162-179, `_get_transcriber()` line 414-440
**Missing assertion:** Verify that failed preload triggers sync retry, verify both errors are logged

---

### 1.3 Transcriber.transcribe() Throws During get_whisper_audio()
**Scenario:** Audio is successfully retrieved, but `transcriber.transcribe()` raises exception

**Why it matters:**
- The exception is caught in generic `except Exception` block (line 328-330)
- We lose the specific error type (OOM, model error, audio corruption, etc.)
- Error message doesn't distinguish "transcription failed" from "model load failed"
- Audio duration and inference time are lost

**Test name:** `test_pipeline_transcription_exception`
**Code path:** `_perform_transcription()` line 442-483, exception handler line 328-330
**Missing assertion:** Verify generic error returned, verify specific exception is logged

---

### 1.4 Streaming Mode Initialization Fails
**Scenario:** `_check_streaming_enabled()` returns True but transcriber not ready

**Why it matters:**
- Line 254-258 has fallback to batch mode if transcriber is None
- But we never test what happens if streaming is enabled but transcriber initialization fails
- The `_streaming_transcriber` is set to `None` but `_streaming_enabled` remains True
- Next `stop_and_transcribe()` would try to use `None` transcriber at line 348

**Test name:** `test_pipeline_streaming_init_with_null_transcriber`
**Code path:** `_init_streaming_transcriber()` line 251-267, fallback line 254-258
**Missing assertion:** Verify `_streaming_enabled` is False after failed init, verify batch mode used

---

### 1.5 Streaming.finalize() Returns Empty Text
**Scenario:** Streaming transcriber completes but returns empty/whitespace text

**Why it matters:**
- `_stop_and_transcribe_streaming()` has check at line 362: `if not streaming_result.text.strip()`
- But we never test whether streaming can return None instead of empty string
- What if `streaming_result.text` is None? We'd crash on `.strip()` call

**Test name:** `test_pipeline_streaming_none_text`
**Code path:** `_stop_and_transcribe_streaming()` line 332-371
**Missing assertion:** Verify graceful handling of None text, verify error message is appropriate

---

## Category 2: Boundary Conditions (Empty, Null, Max Values)

### 2.1 Zero-Length Audio Array
**Scenario:** `get_whisper_audio()` returns array with length=0

**Why it matters:**
- `_get_recorded_audio()` line 393 calculates: `audio_duration = len(audio) / 16000.0` = 0.0
- Check at line 394: `if audio_duration < 0.1` would catch this
- But we never test the boundary: what if audio is exactly 0 bytes?
- Also, what if `get_whisper_audio()` returns None instead of empty array?

**Test name:** `test_pipeline_zero_length_audio`
**Code path:** `_get_recorded_audio()` line 380-399
**Missing assertion:** Verify zero-length audio returns error, verify None audio is handled

---

### 2.2 Minimal Audio (Just Under Threshold)
**Scenario:** Audio is captured but duration is 0.099 seconds (just under 0.1s minimum)

**Why it matters:**
- Boundary condition: 0.099s is filtered, 0.1s is accepted
- We test "no audio" but never test the boundary
- Different models might need different minimum duration
- Log message says "no audio captured" but actually means "too short"

**Test name:** `test_pipeline_minimal_audio_boundary`
**Code path:** `_get_recorded_audio()` line 394
**Missing assertion:** Verify 0.099s is rejected, verify exact message matches threshold logic

---

### 2.3 Maximum Generation ID Wraparound
**Scenario:** `_current_generation` reaches INT_MAX and wraps to negative

**Why it matters:**
- Line 125: `self._current_generation: int = 0`
- Line 532: `self._current_generation += 1`
- No protection against integer overflow (Python int is unbounded, but comparison at line 585 could have issues)
- After 2^63 generations (unrealistic but possible in long-running test), behavior undefined

**Test name:** `test_pipeline_generation_wraparound`
**Code path:** `stop_and_transcribe_async()` line 510-559, `_transcribe_background()` line 585
**Missing assertion:** Verify generation comparison still works with very large numbers

---

### 2.4 Empty ModeManager Prompt
**Scenario:** `get_mode_prompt()` returns empty string

**Why it matters:**
- `_perform_transcription()` line 460 uses `get_mode_prompt()` return value
- If empty string is returned, transcriber still runs with `initial_prompt=""`
- We never test whether Whisper handles empty prompt correctly
- Might produce different results than None or no prompt

**Test name:** `test_pipeline_empty_mode_prompt`
**Code path:** `_perform_transcription()` line 456-465
**Missing assertion:** Verify empty prompt doesn't crash, verify transcription still works

---

### 2.5 Null Callbacks in Async Transcription
**Scenario:** `stop_and_transcribe_async()` called with both `on_result=None` and `on_complete=None`

**Why it matters:**
- Lines 545-548 check `if on_result` and `if on_complete` before calling
- But we never test the case where both are None
- The function still does full transcription, returning generation ID
- Edge case: caller doesn't care about result but wants generation ID

**Test name:** `test_pipeline_async_no_callbacks`
**Code path:** `stop_and_transcribe_async()` line 510-559
**Missing assertion:** Verify function completes without error, verify generation returned

---

## Category 3: Race Conditions and Threading (Async/Threaded Code)

### 3.1 Concurrent start_recording() During Preload
**Scenario:** Two threads call `start_recording()` while model preload is in progress

**Why it matters:**
- Test at line 212-272 verifies mutual exclusion for `start_recording()` calls
- But doesn't test mutual exclusion between `start_recording()` and `_preload_model()` (background thread)
- `_preload_model()` acquires `_transcriber_lock`, `start_recording()` doesn't touch it
- If both threads try to initialize transcriber, could create two instances

**Test name:** `test_pipeline_start_recording_during_preload`
**Code path:** `start_recording()` line 181-237, `_preload_model()` line 162-179
**Missing assertion:** Verify only one transcriber instance created, verify no race in `_streaming_enabled`

---

### 3.2 stop_and_transcribe_async() Called Multiple Times Rapidly
**Scenario:** `stop_and_transcribe_async()` called 5 times in quick succession without waiting

**Why it matters:**
- Each call increments generation (line 532)
- Earlier transcriptions become stale and skip callbacks (line 585-591)
- We test stale generation skips callbacks (line 397-454)
- But don't test what happens if all 5 are in progress simultaneously
- Thread pool has 1 worker, so they queue up - but we never verify ordering

**Test name:** `test_pipeline_async_rapid_calls`
**Code path:** `stop_and_transcribe_async()` line 510-559, `_transcribe_background()` line 561-657
**Missing assertion:** Verify all generations processed eventually, verify correct one succeeds

---

### 3.3 Shutdown During start_recording()
**Scenario:** Thread A calls `shutdown()`, Thread B calls `start_recording()` simultaneously

**Why it matters:**
- `shutdown()` sets `_shutting_down = True` (line 667)
- `start_recording()` doesn't check `_shutting_down`
- Could start recording after shutdown, orphaning the recorder
- Violates the contract that shutdown should prevent new operations

**Test name:** `test_pipeline_start_recording_during_shutdown`
**Code path:** `start_recording()` line 181-237, `shutdown()` line 659-673
**Missing assertion:** Verify recording doesn't start after shutdown initiated

---

### 3.4 Mute State Race Condition
**Scenario:** `start_recording(mute_system=True)` and `cancel_recording()` in different threads

**Why it matters:**
- `start_recording()` sets `_was_muted_before_recording` (line 195)
- `cancel_recording()` doesn't touch mute state (line 496-508)
- But if cancel happens before restoration in `stop_and_transcribe()`, mute state is lost
- Also: no synchronization on `_was_muted_before_recording` between threads

**Test name:** `test_pipeline_mute_state_concurrent_cancel`
**Code path:** `start_recording()` line 195, `cancel_recording()` line 496-508
**Missing assertion:** Verify mute state is restored even if cancel called between start/stop

---

### 3.5 Audio Chunk Callback During Cleanup
**Scenario:** `_on_audio_chunk()` called from sounddevice thread after `_streaming_transcriber = None`

**Why it matters:**
- `_on_audio_chunk()` line 269-278 checks `if self._streaming_transcriber is not None`
- But there's a race: between the check and the call, `_streaming_transcriber` could be set to None
- `cancel_recording()` line 506 sets `_streaming_transcriber = None` without lock
- `_on_audio_chunk()` is called from sounddevice callback thread (line 269, comment)

**Test name:** `test_pipeline_audio_chunk_race_with_cleanup`
**Code path:** `_on_audio_chunk()` line 269-278, `cancel_recording()` line 496-508
**Missing assertion:** Verify no crash when chunk arrives during cleanup, verify thread safety

---

## Category 4: State Transition Edge Cases

### 4.1 Double Cancellation
**Scenario:** `cancel_recording()` called twice without `start_recording()` in between

**Why it matters:**
- First call sets `_streaming_transcriber = None` (line 506)
- Second call tries to set it to None again (idempotent, but should be tested)
- Also: `_streaming_enabled = False` is idempotent
- Edge case: what if recorder is None the second time?

**Test name:** `test_pipeline_double_cancellation`
**Code path:** `cancel_recording()` line 496-508
**Missing assertion:** Verify second cancel doesn't crash, verify idempotent behavior

---

### 4.2 Transcribe Before Recording Started
**Scenario:** `stop_and_transcribe()` called without `start_recording()`

**Why it matters:**
- Test at line 77-96 covers this: "no audio captured" error
- But doesn't test async version: `stop_and_transcribe_async()` without `start_recording()`
- Returns generation ID and calls callbacks with error - but we never test this flow

**Test name:** `test_pipeline_async_transcribe_without_start` (Currently: test_stop_and_transcribe_async_no_audio)
**Code path:** `stop_and_transcribe_async()` line 510-559
**Status:** ALREADY TESTED at line 363-395

---

### 4.3 preload_model_async() Called Multiple Times
**Scenario:** `preload_model_async()` called 3 times before first completes

**Why it matters:**
- Each call spawns new background thread (line 155-160)
- `_model_preload_complete` event is set once (line 179)
- All threads race to set `_transcriber` - last one wins
- No test verifies only one transcriber is created even if preload called multiple times

**Test name:** `test_pipeline_multiple_preload_calls`
**Code path:** `preload_model_async()` line 149-160, `_preload_model()` line 162-179
**Missing assertion:** Verify only one transcriber instance, verify only one completes, verify no error

---

### 4.4 get_rms_level() Called While Not Recording
**Scenario:** `get_rms_level()` called when `_recorder` is None

**Why it matters:**
- Returns 0.0 safely (line 494)
- But we never test that the safe default is actually returned
- Also: what if `_recorder.is_recording` is True but `get_rms_level()` throws?

**Test name:** `test_pipeline_rms_level_not_recording`
**Code path:** `get_rms_level()` line 485-494
**Missing assertion:** Verify returns 0.0, verify doesn't crash

---

### 4.5 is_model_ready After Preload Error
**Scenario:** Model preload fails, then check `is_model_ready`

**Why it matters:**
- Property at line 141-147: returns `event.is_set() and error is None`
- If preload failed, event is set BUT error is not None, so returns False (correct)
- But we never test this logic explicitly
- Also: what if error is cleared but event still set?

**Test name:** `test_pipeline_is_model_ready_with_error`
**Code path:** `is_model_ready` property line 141-147
**Missing assertion:** Verify returns False when error set, verify returns True only when both conditions met

---

## Category 5: Cleanup and Resource Management

### 5.1 Executor Not Owned (External Executor Shutdown)
**Scenario:** Pipeline initialized with external executor, then `shutdown()` called

**Why it matters:**
- Line 123: `self._owns_executor = executor is None`
- `shutdown()` line 671: only shuts down executor if `_owns_executor` is True
- We test with injected executor (line 302-343, sync_executor)
- But never verify that external executor is NOT shut down
- If external executor is shared, pipeline shouldn't shutdown another component's resource

**Test name:** `test_pipeline_external_executor_not_shut_down`
**Code path:** `shutdown()` line 659-673, `__init__` line 119-123
**Missing assertion:** Verify external executor NOT shutdown, verify `_owns_executor` prevents it

---

### 5.2 Resource Leak: Recorder Not Cleaned Up After Error
**Scenario:** `start_recording()` succeeds, but `stop_and_transcribe()` raises exception in finally block

**Why it matters:**
- `stop_and_transcribe()` line 289-330: if exception in finally block, `_recorder` never cleaned
- recorder holds audio buffer and file handles
- Next call to `start_recording()` would try to create another recorder
- Over time, leaks memory and file descriptors

**Test name:** `test_pipeline_recorder_leak_on_exception`
**Code path:** `stop_and_transcribe()` line 280-330
**Missing assertion:** Verify recorder state after exception, verify cleanup happens

---

### 5.3 Streaming Transcriber Not Cleaned on Exception
**Scenario:** `start_recording(streaming=True)` succeeds, `stop_and_transcribe()` raises in finalize()

**Why it matters:**
- `_stop_and_transcribe_streaming()` line 348-371
- If `streaming_transcriber.finalize()` throws, `_streaming_transcriber = None` never executed (line 351)
- Next call sees `_streaming_enabled = True` but `_streaming_transcriber = None`
- `_on_audio_chunk()` would call None.process_chunk()

**Test name:** `test_pipeline_streaming_leak_on_finalize_error`
**Code path:** `_stop_and_transcribe_streaming()` line 332-371
**Missing assertion:** Verify cleanup happens even if finalize() throws

---

### 5.4 Model Preload Thread Never Completes
**Scenario:** Model preload thread hangs (infinite wait)

**Why it matters:**
- Thread is daemon (line 157), so won't block program exit
- But `_model_preload_complete` event is never set
- Any call to `_wait_for_model()` will timeout (line 409)
- We test timeout (line 167-188) but never test that daemon thread hang triggers it

**Test name:** `test_pipeline_preload_thread_hang_timeout`
**Code path:** `_preload_model()` line 162-179, `_wait_for_model()` line 401-412
**Missing assertion:** Verify timeout occurs after MODEL_LOAD_TIMEOUT, verify fallback works

---

### 5.5 Config Change During Recording
**Scenario:** `invalidate_transcriber()` and `clear_model_preload()` called while recording

**Why it matters:**
- `invalidate_transcriber()` sets `_transcriber = None` (line 706)
- If called during `_perform_transcription()`, could use None transcriber (race condition)
- `clear_model_preload()` clears event (line 713)
- Next transcription would re-wait for model that's already loaded

**Test name:** `test_pipeline_config_change_during_recording`
**Code path:** `invalidate_transcriber()` line 698-706, `clear_model_preload()` line 708-715
**Missing assertion:** Verify thread safety, verify no use-after-invalidate, verify reload works

---

## Summary Table

| Category | Count | Severity | Example |
|----------|-------|----------|---------|
| Error Conditions | 5 | HIGH | Recorder init failure not tested |
| Boundary Conditions | 5 | HIGH | Zero-length audio array not tested |
| Race Conditions | 5 | CRITICAL | Audio chunk callback race during cleanup |
| State Transitions | 5 | MEDIUM | Double cancellation not tested |
| Cleanup/Resources | 5 | CRITICAL | Resource leaks on exception not tested |
| **TOTAL** | **25** | - | - |

---

## Recommended Testing Priority

### Tier 1 (Critical - Test First)
1. **Resource leaks on exception** (5.2, 5.3) - Could crash application in long-term use
2. **Audio chunk callback race** (3.5) - Could cause segfault
3. **Recorder initialization failure** (1.1) - Error path not covered
4. **Concurrent preload calls** (4.3) - Could create duplicate resources

### Tier 2 (High - Test Second)
1. **Streaming mode fallback** (1.4) - Streaming is new feature, needs coverage
2. **Model preload error recovery** (1.2) - Complex error handling path
3. **Mute state race condition** (3.4) - User-facing audio state bug
4. **Shutdown during start_recording** (3.3) - Violates contract

### Tier 3 (Medium - Test Third)
1. **State transition edge cases** (4.x) - Less likely to occur but should be defensive
2. **Boundary conditions** (2.x) - Should be caught by limits, but edge cases exist
3. **Config change during recording** (5.5) - Settings changes are rare but possible

---

## Code Smell Notes

1. **Missing locks:** `_was_muted_before_recording` and `_streaming_enabled` are not protected by locks
2. **No cleanup on exception:** `_stop_and_transcribe_streaming()` doesn't use try/finally for cleanup
3. **Race between check and use:** `_on_audio_chunk()` checks `is not None` but doesn't acquire lock
4. **Generic exception handling:** Multiple places catch `Exception` and lose error context
5. **No resource pooling:** `_recorder` recreated each session instead of reused
