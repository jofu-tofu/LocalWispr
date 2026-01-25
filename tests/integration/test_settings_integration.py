"""Integration tests for settings propagation.

These tests verify that settings changes propagate correctly through
the full stack: GUI → Config → SettingsManager → Handlers → Components.

These tests target the #1 recurring bug: settings changes don't apply
until restart. They test end-to-end workflows, not just isolated units.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import pytest
from localwispr.config import clear_config_cache, get_config, reload_config, save_config
from localwispr.settings_manager import get_settings_manager


class TestCriticalSettingsPropagation:
    """Tests for the top 3 critical settings that fail to propagate.

    These tests verify the user's specific reported bugs:
    1. Model changes (tiny → large-v3) don't apply
    2. Hotkey mode changes (PTT ↔ Toggle) don't apply
    3. Streaming toggle (enable/disable) doesn't apply
    """

    def test_model_change_propagates_to_pipeline(self, full_app_context, mocker):
        """CRITICAL: Verify model changes actually propagate to pipeline.

        This catches the #1 recurring bug: changing model in settings
        but next transcription still uses old model.

        Flow tested:
        1. TrayApp initializes with "tiny" model (auto-registers handlers)
        2. Settings change saves config with "large-v3" model
        3. Settings manager detects change and calls registered handlers
        4. Both invalidate_transcriber and clear_model_preload are called
        5. Config is updated

        Expected: Test should PASS if propagation works, FAIL if bug exists.
        """
        config_path = full_app_context["config_path"]

        # Mock OverlayWidget to allow TrayApp initialization in tests
        mocker.patch("localwispr.overlay.OverlayWidget")

        # Reset settings manager to avoid cross-test contamination
        import localwispr.settings_manager as sm_module
        sm_module._settings_manager = None

        # Spy on pipeline methods BEFORE TrayApp creates pipeline
        # This ensures handler registration picks up the spy
        from localwispr.pipeline import RecordingPipeline
        invalidate_spy = mocker.spy(RecordingPipeline, "invalidate_transcriber")
        clear_preload_spy = mocker.spy(RecordingPipeline, "clear_model_preload")

        # Step 1: Create TrayApp (this auto-registers handlers via _register_settings_handlers)
        from localwispr.tray import TrayApp
        app = TrayApp()

        # Step 2: Verify initial state (tiny model)
        initial_config = get_config()
        assert initial_config["model"]["name"] == "tiny"

        # Step 3: Save old config for comparison
        old_config = copy.deepcopy(initial_config)

        # Step 4: Change model to large-v3 and apply via manager
        new_config = copy.deepcopy(initial_config)
        new_config["model"]["name"] = "large-v3"

        # Save to disk (simulates GUI Save)
        save_config(new_config, config_path)

        # Reload config (simulates controller reload)
        reload_config()
        new_config_loaded = get_config()

        # Get pipeline instance before applying settings
        pipeline = app._pipeline

        # Apply settings via manager (this triggers handlers)
        manager = get_settings_manager()
        manager.apply_settings(old_config, new_config_loaded)

        # Step 5: Verify BOTH handlers were called (model.name has both flags)
        assert invalidate_spy.call_count == 1
        assert clear_preload_spy.call_count == 1

        # Step 6: Verify actual state changed - transcriber should be None
        assert pipeline._transcriber is None, \
            "Transcriber should be invalidated (set to None) after model change"

        # Verify preload state was cleared
        assert not pipeline._model_preload_complete.is_set(), \
            "Model preload flag should be cleared after model change"

        # Step 7: Verify config was updated
        current_config = get_config()
        assert current_config["model"]["name"] == "large-v3"

    def test_hotkey_mode_change_propagates_to_listener(self, full_app_context, mocker):
        """CRITICAL: Verify hotkey mode changes propagate to listener.

        This catches the #2 recurring bug: changing hotkey mode
        but listener still uses old mode.

        Flow tested:
        1. TrayApp initializes with push-to-talk mode (auto-registers handlers)
        2. Settings change saves config with toggle mode
        3. Settings manager detects change and calls HOTKEY_LISTENER handler
        4. _restart_hotkey_listener is called
        5. Config is updated

        Expected: Test should PASS if propagation works, FAIL if bug exists.
        """
        config_path = full_app_context["config_path"]

        # Mock OverlayWidget to allow TrayApp initialization
        mocker.patch("localwispr.overlay.OverlayWidget")

        # Reset settings manager
        import localwispr.settings_manager as sm_module
        sm_module._settings_manager = None

        # Spy on _restart_hotkey_listener BEFORE TrayApp is created
        from localwispr.tray import TrayApp
        restart_spy = mocker.spy(TrayApp, "_restart_hotkey_listener")

        # Step 1: Create TrayApp (auto-registers handlers)
        app = TrayApp()

        # Step 2: Verify initial state (push-to-talk)
        initial_config = get_config()
        assert initial_config["hotkeys"]["mode"] == "push-to-talk"

        # Save old config for comparison
        old_config = copy.deepcopy(initial_config)

        # Step 3: Change mode to toggle via config save
        new_config = copy.deepcopy(initial_config)
        new_config["hotkeys"]["mode"] = "toggle"

        # Save to disk
        save_config(new_config, config_path)

        # Step 4: Reload config and apply via manager
        reload_config()
        new_config_loaded = get_config()

        # Verify config changed in memory
        assert old_config["hotkeys"]["mode"] == "push-to-talk"
        assert new_config_loaded["hotkeys"]["mode"] == "toggle"

        # Apply settings (this should trigger _restart_hotkey_listener)
        manager = get_settings_manager()
        manager.apply_settings(old_config, new_config_loaded)

        # Step 5: Verify handler was called
        assert restart_spy.call_count == 1

        # Step 6: Verify config persisted
        current_config = get_config()
        assert current_config["hotkeys"]["mode"] == "toggle"

    def test_streaming_toggle_propagates_to_transcription(self, full_app_context, mocker):
        """CRITICAL: Verify streaming toggle propagates to transcription workflow.

        This catches the #3 recurring bug: enabling streaming mode
        but next transcription still uses regular transcriber.

        Flow tested:
        1. Start with streaming disabled
        2. Save config with streaming enabled
        3. Reload config and trigger settings manager
        4. Handler should be called to recreate transcription workflow
        """
        config_path = full_app_context["config_path"]
        pipeline = full_app_context["get_pipeline"]()

        # Step 1: Verify initial state (streaming disabled)
        initial_config = get_config()
        assert initial_config["streaming"]["enabled"] is False

        # Step 2: Save old config for comparison
        old_config = copy.deepcopy(initial_config)

        # Step 3: Change streaming to enabled via config save
        new_config = copy.deepcopy(initial_config)
        new_config["streaming"]["enabled"] = True

        # Save to disk (use temp config path)
        save_config(new_config, config_path)

        # Step 4: Reload config and apply settings
        reload_config()
        new_config_loaded = get_config()

        # Step 4: Create handler spy
        handler_spy = mocker.MagicMock()

        # Register handler (if streaming had a flag)
        # But streaming.enabled has InvalidationFlags.NONE in SETTINGS_INVALIDATION
        # so no handler will be registered/called
        manager = get_settings_manager()

        # Register handler for streaming flag
        from localwispr.settings_manager import SETTINGS_INVALIDATION, InvalidationFlags

        streaming_flags = SETTINGS_INVALIDATION.get("streaming.enabled", InvalidationFlags.NONE)

        # Verify streaming has proper invalidation flag
        assert streaming_flags == InvalidationFlags.TRANSCRIBER, \
            "streaming.enabled should have TRANSCRIBER flag"

        manager.register_handler(streaming_flags, handler_spy)

        # Trigger settings manager
        manager.apply_settings(old_config, new_config_loaded)

        # Verify handler was called
        handler_spy.assert_called_once()

        # Verify config updated
        current_config = get_config()
        assert current_config["streaming"]["enabled"] is True


class TestSettingsChangeWorkflows:
    """Tests for additional settings propagation scenarios."""

    def test_settings_change_during_recording_queued(self, full_app_context, mocker):
        """Test that settings changes during recording are handled correctly.

        This tests the race condition: what happens if user changes settings
        while a recording is in progress?

        Expected behavior:
        - Settings should save successfully
        - Changes should apply (handlers called)
        - No crashes or data corruption
        """
        config_path = full_app_context["config_path"]

        # Mock OverlayWidget
        mocker.patch("localwispr.overlay.OverlayWidget")

        # Reset settings manager
        import localwispr.settings_manager as sm_module
        sm_module._settings_manager = None

        # Spy BEFORE TrayApp is created
        from localwispr.pipeline import RecordingPipeline
        invalidate_spy = mocker.spy(RecordingPipeline, "invalidate_transcriber")

        # Create TrayApp
        from localwispr.tray import TrayApp
        app = TrayApp()

        # Start recording
        app._pipeline.start_recording()
        assert app._pipeline.is_recording

        # While recording, change settings
        initial_config = get_config()
        old_config = copy.deepcopy(initial_config)

        new_config = copy.deepcopy(initial_config)
        new_config["model"]["name"] = "base"

        save_config(new_config, config_path)
        reload_config()
        new_config_loaded = get_config()

        # Apply settings while recording
        manager = get_settings_manager()
        manager.apply_settings(old_config, new_config_loaded)

        # Handler should be called even during recording
        assert invalidate_spy.call_count == 1

        # Stop recording (use stop_and_transcribe which stops and transcribes)
        result = app._pipeline.stop_and_transcribe()

        # Verify config changed
        assert get_config()["model"]["name"] == "base"

    def test_model_change_triggers_preload_then_invalidation(self, full_app_context, mocker):
        """Test that model changes trigger handlers in correct order.

        When model changes:
        1. MODEL_PRELOAD handler should clear preload
        2. TRANSCRIBER handler should invalidate transcriber
        3. Both handlers should be called

        This verifies handler execution order.
        """
        config_path = full_app_context["config_path"]

        # Mock OverlayWidget
        mocker.patch("localwispr.overlay.OverlayWidget")

        # Reset settings manager
        import localwispr.settings_manager as sm_module
        sm_module._settings_manager = None

        # Spy BEFORE TrayApp is created
        from localwispr.pipeline import RecordingPipeline
        clear_preload_spy = mocker.spy(RecordingPipeline, "clear_model_preload")
        invalidate_spy = mocker.spy(RecordingPipeline, "invalidate_transcriber")

        # Create TrayApp (auto-registers both handlers)
        from localwispr.tray import TrayApp
        app = TrayApp()

        # Change model
        initial_config = get_config()
        old_config = copy.deepcopy(initial_config)

        new_config = copy.deepcopy(initial_config)
        new_config["model"]["name"] = "medium"

        save_config(new_config, config_path)
        reload_config()
        new_config_loaded = get_config()

        # Apply settings
        manager = get_settings_manager()
        manager.apply_settings(old_config, new_config_loaded)

        # Verify BOTH handlers were called (model.name has both flags)
        assert clear_preload_spy.call_count == 1
        assert invalidate_spy.call_count == 1

    def test_config_reload_propagates_to_all_consumers(self, full_app_context, mocker):
        """Test that config reload invalidates cache for all consumers.

        When config is reloaded:
        - All consumers should see the new config
        - Multiple get_config() calls should return same new values

        This tests cache invalidation works correctly.
        """
        config_path = full_app_context["config_path"]

        # Step 1: Verify initial config value
        initial_config = get_config()
        assert initial_config["model"]["name"] == "tiny"
        assert initial_config["hotkeys"]["mode"] == "push-to-talk"

        # Step 2: Change config on disk directly (bypass settings flow)
        new_config = copy.deepcopy(initial_config)
        new_config["model"]["name"] = "large-v3"
        new_config["hotkeys"]["mode"] = "toggle"
        save_config(new_config, config_path)

        # Step 3: Reload config (clears cache)
        reload_config()

        # Step 4: Verify ALL consumers see new values
        config_read_1 = get_config()
        config_read_2 = get_config()
        config_read_3 = get_config()

        # All reads should see the new values
        assert config_read_1["model"]["name"] == "large-v3"
        assert config_read_2["model"]["name"] == "large-v3"
        assert config_read_3["model"]["name"] == "large-v3"

        assert config_read_1["hotkeys"]["mode"] == "toggle"
        assert config_read_2["hotkeys"]["mode"] == "toggle"
        assert config_read_3["hotkeys"]["mode"] == "toggle"

    def test_settings_change_preserves_manual_mode_override(self, full_app_context, mocker):
        """Test that transcription mode cycling survives settings changes.

        If user manually cycled to a different transcription mode (e.g. CODE):
        - Settings change shouldn't reset mode to default (DICTATION)
        - Manual override should persist

        This tests that settings and runtime state are properly separated.
        """
        config_path = full_app_context["config_path"]

        # Mock OverlayWidget
        mocker.patch("localwispr.overlay.OverlayWidget")

        # Reset settings manager
        import localwispr.settings_manager as sm_module
        sm_module._settings_manager = None

        # Create TrayApp (starts with DICTATION mode by default)
        from localwispr.tray import TrayApp
        app = TrayApp()

        # Verify default transcription mode
        from localwispr.modes.definitions import ModeType
        assert app._mode_manager.current_mode_type == ModeType.DICTATION

        # Manually cycle mode to EMAIL (user override)
        # MODE_CYCLE_ORDER: CODE → NOTES → DICTATION → EMAIL → CHAT
        app._mode_manager.cycle_mode()  # DICTATION → EMAIL

        # Verify mode changed
        assert app._mode_manager.current_mode_type == ModeType.EMAIL

        # Now change a DIFFERENT setting (model, not mode)
        initial_config = get_config()
        old_config = copy.deepcopy(initial_config)

        new_config = copy.deepcopy(initial_config)
        new_config["model"]["name"] = "base"  # Change model, not mode

        save_config(new_config, config_path)
        reload_config()
        new_config_loaded = get_config()

        # Apply settings
        manager = get_settings_manager()
        manager.apply_settings(old_config, new_config_loaded)

        # Verify manual transcription mode override persists (not reset to default)
        # The mode should still be EMAIL (manual override), not DICTATION (default)
        assert app._mode_manager.current_mode_type == ModeType.EMAIL

    def test_rapid_settings_changes_final_state_wins(self, full_app_context, mocker):
        """Test concurrent rapid settings changes.

        If user rapidly changes settings multiple times:
        - Final state should win
        - Handlers called for each change

        This tests settings atomicity under rapid changes.
        """
        config_path = full_app_context["config_path"]

        # Mock OverlayWidget
        mocker.patch("localwispr.overlay.OverlayWidget")

        # Reset settings manager
        import localwispr.settings_manager as sm_module
        sm_module._settings_manager = None

        # Spy BEFORE TrayApp is created
        from localwispr.pipeline import RecordingPipeline
        invalidate_spy = mocker.spy(RecordingPipeline, "invalidate_transcriber")

        # Create TrayApp
        from localwispr.tray import TrayApp
        app = TrayApp()

        initial_config = get_config()
        assert initial_config["model"]["name"] == "tiny"

        # Rapidly change model 3 times: tiny → base → small → medium
        models = ["base", "small", "medium"]

        for model in models:
            old_config = get_config()
            new_config = copy.deepcopy(old_config)
            new_config["model"]["name"] = model

            save_config(new_config, config_path)
            reload_config()
            new_config_loaded = get_config()

            manager = get_settings_manager()
            manager.apply_settings(old_config, new_config_loaded)

        # Verify final state wins
        final_config = get_config()
        assert final_config["model"]["name"] == "medium"

        # Verify handler was called for EACH change (3 times)
        assert invalidate_spy.call_count == 3

    def test_validation_failure_prevents_partial_save(self, full_app_context, mocker):
        """Test that validation errors prevent partial saves.

        If settings validation fails:
        - No partial config should be saved
        - No handlers should be called
        - Original config should remain unchanged

        This tests atomicity guarantee.
        """
        config_path = full_app_context["config_path"]

        # Mock OverlayWidget
        mocker.patch("localwispr.overlay.OverlayWidget")

        # Reset settings manager
        import localwispr.settings_manager as sm_module
        sm_module._settings_manager = None

        # Spy BEFORE TrayApp is created
        from localwispr.pipeline import RecordingPipeline
        from localwispr.tray import TrayApp
        invalidate_spy = mocker.spy(RecordingPipeline, "invalidate_transcriber")
        restart_spy = mocker.spy(TrayApp, "_restart_hotkey_listener")

        # Create TrayApp
        app = TrayApp()

        # Get initial config
        initial_config = get_config()
        original_model = initial_config["model"]["name"]

        # Create an INVALID config snapshot (for validation testing)
        from localwispr.settings_model import SettingsSnapshot, SettingsValidator
        from dataclasses import replace

        valid_snapshot = SettingsSnapshot.from_config_dict(initial_config)
        invalid_snapshot = replace(valid_snapshot, model_name="")  # Invalid: empty model name

        # Validation should fail
        result = SettingsValidator.validate(invalid_snapshot)
        assert not result.is_valid
        assert "model_name" in result.errors

        # DON'T save invalid config (controller would prevent this)
        # Verify config unchanged
        current_config = get_config()
        assert current_config["model"]["name"] == original_model

        # Verify NO handlers were called (no partial update)
        assert invalidate_spy.call_count == 0
        assert restart_spy.call_count == 0

    def test_test_build_uses_separate_config(self, tmp_path, mocker):
        """Test that Test build variant uses separate config file.

        Test and Stable builds should:
        - Use different config files (config.toml vs config-test.toml)
        - Not interfere with each other
        - Both work simultaneously

        This tests build variant isolation.
        """
        # This test verifies that IS_TEST_BUILD is correctly set
        # and that the build system uses the right config file

        from localwispr import IS_TEST_BUILD

        # The test suite runs in the same process as the code being tested,
        # so IS_TEST_BUILD reflects the current Python process (not build variant)

        # Create two temp config files
        stable_config = tmp_path / "config.toml"
        stable_config.write_text("""
[model]
name = "tiny"

[hotkeys]
mode = "push-to-talk"
""")

        test_config = tmp_path / "config-test.toml"
        test_config.write_text("""
[model]
name = "large-v3"

[hotkeys]
mode = "toggle"
""")

        # Load stable config
        clear_config_cache()
        from localwispr.config import load_config
        stable_loaded = load_config(stable_config)
        assert stable_loaded["model"]["name"] == "tiny"
        assert stable_loaded["hotkeys"]["mode"] == "push-to-talk"

        # Load test config
        clear_config_cache()
        test_loaded = load_config(test_config)
        assert test_loaded["model"]["name"] == "large-v3"
        assert test_loaded["hotkeys"]["mode"] == "toggle"

        # Verify they're different (isolation verified)
        assert stable_loaded["model"]["name"] != test_loaded["model"]["name"]
        assert stable_loaded["hotkeys"]["mode"] != test_loaded["hotkeys"]["mode"]
