"""Tests for the configuration module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_creates_defaults_if_missing(self, tmp_path, mocker):
        """Test that missing config file creates defaults."""
        config_path = tmp_path / "config.toml"

        # Mock _get_config_path to return our test path
        mocker.patch(
            "localwispr.config._get_config_path",
            return_value=config_path,
        )

        from localwispr.config import load_config

        config = load_config(config_path)

        # Should create file with defaults
        assert config_path.exists()
        assert config["model"]["name"] == "large-v3"  # Default

    def test_load_config_merges_with_defaults(self, tmp_path):
        """Test that partial config is merged with defaults."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[model]
name = "tiny"
""")

        from localwispr.config import load_config

        config = load_config(config_path)

        # Custom value
        assert config["model"]["name"] == "tiny"
        # Default values filled in
        assert config["model"]["device"] == "cuda"
        assert config["model"]["compute_type"] == "float16"

    def test_load_config_full_file(self, mock_config_file):
        """Test loading a complete config file."""
        from localwispr.config import load_config

        config = load_config(mock_config_file)

        assert config["model"]["name"] == "tiny"
        assert config["model"]["device"] == "cpu"
        assert config["hotkeys"]["mode"] == "push-to-talk"
        assert config["output"]["auto_paste"] is False

    def test_load_config_deep_merge(self, tmp_path):
        """Test that nested dicts are properly merged."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[hotkeys]
mode = "toggle"
""")

        from localwispr.config import load_config

        config = load_config(config_path)

        # Custom value
        assert config["hotkeys"]["mode"] == "toggle"
        # Defaults preserved
        assert config["hotkeys"]["audio_feedback"] is True
        assert "modifiers" in config["hotkeys"]


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_config_creates_file(self, tmp_path, mock_config):
        """Test that save_config creates a TOML file."""
        config_path = tmp_path / "config.toml"

        from localwispr.config import save_config

        save_config(mock_config, config_path)

        assert config_path.exists()
        content = config_path.read_text()
        assert "[model]" in content
        assert "[hotkeys]" in content
        assert "[output]" in content

    def test_save_config_preserves_values(self, tmp_path, mock_config):
        """Test that saved config can be reloaded with same values."""
        config_path = tmp_path / "config.toml"

        from localwispr.config import load_config, save_config

        save_config(mock_config, config_path)
        reloaded = load_config(config_path)

        assert reloaded["model"]["name"] == mock_config["model"]["name"]
        assert reloaded["model"]["device"] == mock_config["model"]["device"]
        assert reloaded["hotkeys"]["mode"] == mock_config["hotkeys"]["mode"]

    def test_save_config_includes_comments(self, tmp_path, mock_config):
        """Test that saved config includes helpful comments."""
        config_path = tmp_path / "config.toml"

        from localwispr.config import save_config

        save_config(mock_config, config_path)

        content = config_path.read_text()
        assert "# LocalWispr Configuration" in content
        assert "# Whisper model" in content


class TestDeepMerge:
    """Tests for _deep_merge utility function."""

    def test_deep_merge_simple(self):
        """Test merging simple dicts."""
        from localwispr.config import _deep_merge

        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = _deep_merge(base, override)

        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 4

    def test_deep_merge_nested(self):
        """Test merging nested dicts."""
        from localwispr.config import _deep_merge

        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3}}

        result = _deep_merge(base, override)

        assert result["outer"]["a"] == 1
        assert result["outer"]["b"] == 3

    def test_deep_merge_does_not_modify_base(self):
        """Test that base dict is not modified."""
        from localwispr.config import _deep_merge

        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}

        result = _deep_merge(base, override)

        assert base["a"]["b"] == 1
        assert result["a"]["b"] == 2


class TestGetConfig:
    """Tests for get_config caching function."""

    def test_get_config_caches_result(self, mocker, mock_config_file):
        """Test that get_config caches the loaded config."""
        # Clear any existing cache
        import localwispr.config as config_module

        config_module._cached_config = None

        mocker.patch(
            "localwispr.config._get_config_path",
            return_value=mock_config_file,
        )

        from localwispr.config import get_config

        config1 = get_config()
        config2 = get_config()

        # Should be the same object
        assert config1 is config2

    def test_get_config_thread_safe(self, mocker, mock_config_file):
        """Test that get_config is thread-safe."""
        import threading

        import localwispr.config as config_module

        config_module._cached_config = None

        mocker.patch(
            "localwispr.config._get_config_path",
            return_value=mock_config_file,
        )

        from localwispr.config import get_config

        results = []

        def load_config():
            results.append(get_config())

        threads = [threading.Thread(target=load_config) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be the same object
        assert all(r is results[0] for r in results)


class TestReloadConfig:
    """Tests for reload_config function."""

    def test_reload_config_updates_cache(self, mocker, tmp_path):
        """Test that reload_config updates the cached config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[model]
name = "tiny"
""")

        import localwispr.config as config_module

        config_module._cached_config = None

        mocker.patch(
            "localwispr.config._get_config_path",
            return_value=config_path,
        )

        from localwispr.config import get_config, reload_config

        config1 = get_config()
        assert config1["model"]["name"] == "tiny"

        # Update file
        config_path.write_text("""
[model]
name = "large-v3"
""")

        config2 = reload_config()
        assert config2["model"]["name"] == "large-v3"


class TestClearConfigCache:
    """Tests for clear_config_cache function."""

    def test_clear_config_cache(self, mocker, mock_config_file):
        """Test that clear_config_cache resets the cache."""
        import localwispr.config as config_module

        mocker.patch(
            "localwispr.config._get_config_path",
            return_value=mock_config_file,
        )

        from localwispr.config import clear_config_cache, get_config

        config_module._cached_config = None

        config1 = get_config()
        clear_config_cache()

        # Cache should be cleared
        assert config_module._cached_config is None

        config2 = get_config()

        # New load should create different object
        assert config1 is not config2


class TestGetConfigPath:
    """Tests for _get_config_path function."""

    def test_get_config_path_script_mode(self, mocker):
        """Test config path in script mode."""
        import sys

        # Ensure not frozen (default state)
        if hasattr(sys, 'frozen'):
            mocker.patch.object(sys, 'frozen', False)

        from localwispr.config import _get_config_path

        path = _get_config_path()

        # Should be relative to package parent
        assert path.name == "config.toml"
        # Should be in the project root (parent of localwispr package)
        assert path.parent.name == "LocalWispr"

    def test_get_config_path_frozen_mode(self, mocker, tmp_path):
        """Test config path in frozen (PyInstaller) mode."""
        import sys

        # Simulate frozen mode
        mocker.patch.object(sys, "frozen", True, create=True)
        mocker.patch.object(sys, "executable", str(tmp_path / "LocalWispr.exe"))

        from localwispr.config import _get_config_path

        path = _get_config_path()

        assert path == tmp_path / "config.toml"
