"""Smoke tests for LocalWispr end-to-end workflows.

These tests verify the "car drives" - basic functionality works
from app startup through recording and transcription.
"""

from __future__ import annotations

import pytest


class TestApplicationSmoke:
    """Smoke tests for complete application workflows."""

    def test_app_initialization_smoke(self, full_app_context, mocker):
        """Smoke test: App initializes without crashing.

        Verifies:
        - TrayApp can be created
        - All components initialize
        - Handlers are registered
        - No exceptions during startup
        """
        # Mock OverlayWidget
        mocker.patch("localwispr.overlay.OverlayWidget")

        # Reset settings manager
        import localwispr.settings_manager as sm_module
        sm_module._settings_manager = None

        # Create TrayApp (should not crash)
        from localwispr.tray import TrayApp
        app = TrayApp()

        # Verify components exist
        assert app._pipeline is not None
        assert app._mode_manager is not None
        assert app._overlay is not None

        # Verify settings manager has registered handlers
        manager = sm_module.get_settings_manager()
        from localwispr.settings_manager import InvalidationFlags

        # Should have handlers for each flag
        assert InvalidationFlags.HOTKEY_LISTENER in manager._handlers
        assert InvalidationFlags.TRANSCRIBER in manager._handlers
        assert InvalidationFlags.MODEL_PRELOAD in manager._handlers

    def test_full_recording_workflow_smoke(self, full_app_context, mocker):
        """Smoke test: Complete recording workflow works.

        Verifies:
        - Start recording
        - Recording state is tracked
        - Cancel recording
        - No crashes or exceptions

        Note: This is a smoke test - it verifies the workflow executes
        without crashing, not that transcription produces correct output
        (transcription correctness is tested elsewhere).
        """
        # Mock OverlayWidget
        mocker.patch("localwispr.overlay.OverlayWidget")

        # Reset settings manager
        import localwispr.settings_manager as sm_module
        sm_module._settings_manager = None

        # Create TrayApp
        from localwispr.tray import TrayApp
        app = TrayApp()

        # Start recording
        app._pipeline.start_recording()
        assert app._pipeline.is_recording

        # Cancel recording (simpler than stop_and_transcribe for smoke test)
        app._pipeline.cancel_recording()
        assert not app._pipeline.is_recording

        # Verify no crashes occurred

    def test_settings_change_workflow_smoke(self, full_app_context, mocker):
        """Smoke test: Settings changes work end-to-end.

        Verifies:
        - Change a setting
        - Save config
        - Reload and apply
        - Handler is called
        - No crashes or exceptions
        """
        from localwispr.config import get_config, reload_config, save_config

        config_path = full_app_context["config_path"]

        # Mock OverlayWidget
        mocker.patch("localwispr.overlay.OverlayWidget")

        # Reset settings manager
        import localwispr.settings_manager as sm_module
        sm_module._settings_manager = None

        # Spy before creating app
        from localwispr.pipeline import RecordingPipeline
        invalidate_spy = mocker.spy(RecordingPipeline, "invalidate_transcriber")

        # Create TrayApp
        from localwispr.tray import TrayApp
        app = TrayApp()

        # Get initial config
        import copy
        initial_config = get_config()
        old_config = copy.deepcopy(initial_config)

        # Change model
        new_config = copy.deepcopy(initial_config)
        new_config["model"]["name"] = "base"

        # Save, reload, apply
        save_config(new_config, config_path)
        reload_config()
        new_config_loaded = get_config()

        manager = sm_module.get_settings_manager()
        manager.apply_settings(old_config, new_config_loaded)

        # Verify handler was called (no crash)
        assert invalidate_spy.call_count == 1

    def test_mode_cycling_smoke(self, full_app_context, mocker):
        """Smoke test: Mode cycling works.

        Verifies:
        - App starts with default mode
        - Cycling changes mode
        - Mode manager tracks state
        - No crashes
        """
        # Mock OverlayWidget
        mocker.patch("localwispr.overlay.OverlayWidget")

        # Reset mode manager
        import localwispr.modes.manager as manager_module
        manager_module._mode_manager = None

        # Create TrayApp
        from localwispr.tray import TrayApp
        app = TrayApp()

        # Get initial mode
        from localwispr.modes.definitions import ModeType
        initial_mode = app._mode_manager.current_mode_type
        assert initial_mode == ModeType.DICTATION

        # Cycle mode
        app._mode_manager.cycle_mode()

        # Verify mode changed
        new_mode = app._mode_manager.current_mode_type
        assert new_mode != initial_mode
        assert new_mode == ModeType.EMAIL  # DICTATION → EMAIL in cycle order
