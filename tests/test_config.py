"""Tests for the configuration module."""

from __future__ import annotations




class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_with_explicit_path(self, tmp_path):
        """Test that explicit config path works (legacy mode for tests)."""
        config_path = tmp_path / "config.toml"

        from localwispr.config import load_config

        config = load_config(config_path)

        # Should use defaults when file doesn't exist
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
        assert config["model"]["device"] == "auto"
        assert config["model"]["compute_type"] == "auto"

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
        from localwispr.config import clear_config_cache

        clear_config_cache()

        mocker.patch(
            "localwispr.config.loader._get_defaults_path",
            return_value=mock_config_file,
        )
        mocker.patch(
            "localwispr.config.loader._get_appdata_config_path",
            return_value=mock_config_file.parent / "user-settings.toml",
        )

        from localwispr.config import get_config

        config1 = get_config()
        config2 = get_config()

        # Should be the same object
        assert config1 is config2

    def test_get_config_thread_safe(self, mocker, mock_config_file):
        """Test that get_config is thread-safe."""
        import threading

        from localwispr.config import clear_config_cache

        clear_config_cache()

        mocker.patch(
            "localwispr.config.loader._get_defaults_path",
            return_value=mock_config_file,
        )
        mocker.patch(
            "localwispr.config.loader._get_appdata_config_path",
            return_value=mock_config_file.parent / "user-settings.toml",
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
        user_path = tmp_path / "user-settings.toml"
        user_path.write_text("""
[model]
name = "tiny"
""")

        from localwispr.config import clear_config_cache

        clear_config_cache()

        mocker.patch(
            "localwispr.config.loader._get_defaults_path",
            return_value=tmp_path / "config-defaults.toml",
        )
        mocker.patch(
            "localwispr.config.loader._get_appdata_config_path",
            return_value=user_path,
        )

        from localwispr.config import get_config, reload_config

        config1 = get_config()
        assert config1["model"]["name"] == "tiny"

        # Update file
        user_path.write_text("""
[model]
name = "large-v3"
""")

        config2 = reload_config()
        assert config2["model"]["name"] == "large-v3"


class TestClearConfigCache:
    """Tests for clear_config_cache function."""

    def test_clear_config_cache(self, mocker, mock_config_file):
        """Test that clear_config_cache resets the cache."""
        from localwispr.config import cache as config_cache_module

        mocker.patch(
            "localwispr.config.loader._get_defaults_path",
            return_value=mock_config_file,
        )
        mocker.patch(
            "localwispr.config.loader._get_appdata_config_path",
            return_value=mock_config_file.parent / "user-settings.toml",
        )

        from localwispr.config import clear_config_cache, get_config

        clear_config_cache()

        config1 = get_config()
        clear_config_cache()

        # Cache should be cleared
        assert config_cache_module._cached_config is None

        config2 = get_config()

        # New load should create different object
        assert config1 is not config2


class TestConfigThreadSafety:
    """Tests for thread-safe config cache operations."""

    def test_concurrent_get_config_returns_same_object(
        self, mocker, mock_config_file, isolated_config_cache
    ):
        """Test that concurrent get_config calls return same cached object."""
        import threading
        from localwispr.config import get_config

        mocker.patch("localwispr.config.loader._get_defaults_path",
                     return_value=mock_config_file)
        mocker.patch("localwispr.config.loader._get_appdata_config_path",
                     return_value=mock_config_file.parent / "user.toml")

        results = []
        errors = []

        def get_and_store():
            try:
                config = get_config()
                results.append(config)
            except Exception as e:
                errors.append(e)

        # Spawn 20 threads calling get_config simultaneously
        threads = [threading.Thread(target=get_and_store) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20

        # All should get the SAME object (cached)
        first_config = results[0]
        assert all(cfg is first_config for cfg in results), \
            "get_config should return same cached object"

    def test_reload_config_with_concurrent_access(
        self, mocker, tmp_path, isolated_config_cache
    ):
        """Test reload_config handles concurrent access safely."""
        import threading
        from localwispr.config import get_config, reload_config

        config_file = tmp_path / "config-defaults.toml"
        config_file.write_text('[model]\nname = "tiny"\ndevice = "cpu"')

        mocker.patch("localwispr.config.loader._get_defaults_path",
                     return_value=config_file)
        mocker.patch("localwispr.config.loader._get_appdata_config_path",
                     return_value=tmp_path / "user.toml")

        results = []

        def worker(i):
            # Every 5th thread reloads, others read
            if i % 5 == 0:
                config = reload_config()
            else:
                config = get_config()
            results.append(config)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should complete without crashes
        assert len(results) == 30
        # All configs should be valid
        assert all(r is not None for r in results)
        assert all("model" in r for r in results)


class TestGetDefaultsPath:
    """Tests for _get_defaults_path function."""

    def test_get_defaults_path_script_mode(self, mocker):
        """Test defaults path in script mode."""
        import sys

        # Ensure not frozen (default state)
        if hasattr(sys, 'frozen'):
            mocker.patch.object(sys, 'frozen', False)

        from localwispr.config import _get_defaults_path

        path = _get_defaults_path()

        # In script mode, should use config.toml from project root
        assert path.name == "config.toml"
        assert path.parent.name == "LocalWispr"

    def test_get_defaults_path_frozen_mode(self, mocker, tmp_path):
        """Test defaults path in frozen (PyInstaller) mode."""
        import sys

        # Simulate frozen mode
        mocker.patch.object(sys, "frozen", True, create=True)
        mocker.patch.object(sys, "executable", str(tmp_path / "LocalWispr.exe"))

        from localwispr.config import _get_defaults_path

        path = _get_defaults_path()

        assert path == tmp_path / "config-defaults.toml"


class TestGetAppdataConfigPath:
    """Tests for _get_appdata_config_path function."""

    def test_get_appdata_config_path_stable(self, mocker, tmp_path):
        """Test AppData path for Stable variant."""
        import sys

        # Simulate frozen stable build
        mocker.patch.object(sys, "frozen", True, create=True)
        mocker.patch.object(sys, "executable", str(tmp_path / "LocalWispr.exe"))
        mocker.patch.dict("os.environ", {"APPDATA": str(tmp_path / "AppData")})

        from localwispr.config import _get_appdata_config_path

        path = _get_appdata_config_path()

        assert path == tmp_path / "AppData" / "LocalWispr" / "Stable" / "user-settings.toml"

    def test_get_appdata_config_path_test(self, mocker, tmp_path):
        """Test AppData path for Test variant."""
        import sys

        # Simulate frozen test build
        mocker.patch.object(sys, "frozen", True, create=True)
        mocker.patch.object(sys, "executable", str(tmp_path / "LocalWispr-Test.exe"))
        mocker.patch.dict("os.environ", {"APPDATA": str(tmp_path / "AppData")})

        from localwispr.config import _get_appdata_config_path

        path = _get_appdata_config_path()

        assert path == tmp_path / "AppData" / "LocalWispr" / "Test" / "user-settings.toml"

    def test_get_appdata_config_path_fallback(self, mocker, tmp_path):
        """Test AppData path fallback when APPDATA env var missing."""
        import sys

        # Simulate frozen mode without APPDATA env var
        mocker.patch.object(sys, "frozen", True, create=True)
        mocker.patch.object(sys, "executable", str(tmp_path / "LocalWispr.exe"))
        mocker.patch.dict("os.environ", {}, clear=True)
        mocker.patch("pathlib.Path.home", return_value=tmp_path)

        from localwispr.config import _get_appdata_config_path

        path = _get_appdata_config_path()

        assert path == tmp_path / "AppData" / "Roaming" / "LocalWispr" / "Stable" / "user-settings.toml"


class TestTwoTierLoading:
    """Tests for two-tier configuration system."""

    def test_load_config_two_tier_merge(self, tmp_path, mocker):
        """Test that user overrides win over bundled defaults."""
        # Create bundled defaults
        defaults_path = tmp_path / "config-defaults.toml"
        defaults_path.write_text("""
[model]
name = "base"
device = "cpu"

[vocabulary]
words = ["default1", "default2"]
""")

        # Create user overrides
        user_path = tmp_path / "user-settings.toml"
        user_path.write_text("""
[model]
name = "large-v3"

[vocabulary]
words = ["custom1", "custom2", "custom3"]
""")

        mocker.patch("localwispr.config.loader._get_defaults_path", return_value=defaults_path)
        mocker.patch("localwispr.config.loader._get_appdata_config_path", return_value=user_path)

        from localwispr.config import load_config

        config = load_config()

        # User override should win
        assert config["model"]["name"] == "large-v3"
        # But defaults should fill in missing values
        assert config["model"]["device"] == "cpu"
        # User vocabulary should completely override
        assert config["vocabulary"]["words"] == ["custom1", "custom2", "custom3"]

    def test_save_config_creates_appdata_directory(self, tmp_path, mocker, mock_config):
        """Test that save_config creates AppData directory if missing."""
        user_path = tmp_path / "LocalWispr" / "Stable" / "user-settings.toml"

        mocker.patch("localwispr.config.saver._get_appdata_config_path", return_value=user_path)

        from localwispr.config import save_config

        # Directory shouldn't exist yet
        assert not user_path.parent.exists()

        save_config(mock_config)

        # Should create directory and file
        assert user_path.exists()
        assert user_path.parent.exists()

    def test_user_overrides_survive_bundled_defaults_update(
        self, tmp_path, mocker, isolated_config_cache
    ):
        """Test that user settings persist when bundled defaults change."""
        from localwispr.config import get_config, reload_config

        # Initial bundled defaults
        defaults_path = tmp_path / "config-defaults.toml"
        defaults_path.write_text('''
[model]
name = "base"
device = "cpu"
compute_type = "int8"

[hotkeys]
mode = "push-to-talk"
''')

        # User overrides (persisted in AppData)
        user_path = tmp_path / "user-settings.toml"
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text('''
[model]
name = "large-v3"
device = "cuda"
''')

        mocker.patch("localwispr.config.loader._get_defaults_path",
                     return_value=defaults_path)
        mocker.patch("localwispr.config.loader._get_appdata_config_path",
                     return_value=user_path)

        # Initial load
        config1 = get_config()
        assert config1["model"]["name"] == "large-v3"  # User override
        assert config1["model"]["device"] == "cuda"  # User override
        assert config1["model"]["compute_type"] == "int8"  # From defaults

        # Simulate app rebuild: new bundled defaults
        defaults_path.write_text('''
[model]
name = "base"
device = "cpu"
compute_type = "float16"

[hotkeys]
mode = "voice-activity"
''')

        # Reload config
        config2 = reload_config()

        # User overrides still win
        assert config2["model"]["name"] == "large-v3"
        assert config2["model"]["device"] == "cuda"
        # New default fills in where user didn't override
        assert config2["model"]["compute_type"] == "float16"
        assert config2["hotkeys"]["mode"] == "voice-activity"


class TestMigration:
    """Tests for legacy config migration."""

    def test_migrate_legacy_config(self, tmp_path, mocker):
        """Test that legacy config.toml migrates to AppData."""
        import sys

        # Simulate frozen mode
        mocker.patch.object(sys, "frozen", True, create=True)
        mocker.patch.object(sys, "executable", str(tmp_path / "LocalWispr.exe"))

        # Create legacy config next to EXE
        legacy_path = tmp_path / "config.toml"
        legacy_path.write_text("""
[model]
name = "tiny"

[vocabulary]
words = ["migrated1", "migrated2"]
""")

        # AppData path for user settings
        user_path = tmp_path / "AppData" / "LocalWispr" / "Stable" / "user-settings.toml"

        mocker.patch("localwispr.config.loader._get_appdata_config_path", return_value=user_path)
        mocker.patch("localwispr.config.loader._get_defaults_path", return_value=tmp_path / "config-defaults.toml")

        from localwispr.config import load_config

        # First load should trigger migration
        config = load_config()

        # User settings should exist with migrated data
        assert user_path.exists()
        assert config["model"]["name"] == "tiny"
        assert config["vocabulary"]["words"] == ["migrated1", "migrated2"]

        # Legacy file should still exist (kept as backup)
        assert legacy_path.exists()

    def test_migration_skipped_if_already_migrated(self, tmp_path, mocker):
        """Test that migration doesn't run if user settings already exist."""
        import sys

        mocker.patch.object(sys, "frozen", True, create=True)
        mocker.patch.object(sys, "executable", str(tmp_path / "LocalWispr.exe"))

        # Create both legacy and user settings
        legacy_path = tmp_path / "config.toml"
        legacy_path.write_text("""
[model]
name = "tiny"
""")

        user_path = tmp_path / "AppData" / "LocalWispr" / "Stable" / "user-settings.toml"
        user_path.parent.mkdir(parents=True)
        user_path.write_text("""
[model]
name = "large-v3"
""")

        mocker.patch("localwispr.config.loader._get_appdata_config_path", return_value=user_path)
        mocker.patch("localwispr.config.loader._get_defaults_path", return_value=tmp_path / "config-defaults.toml")

        from localwispr.config import load_config

        config = load_config()

        # Should use existing user settings, NOT legacy
        assert config["model"]["name"] == "large-v3"

    def test_migration_handles_errors_gracefully(self, tmp_path, mocker):
        """Test that migration errors don't crash the app."""
        import sys

        mocker.patch.object(sys, "frozen", True, create=True)
        mocker.patch.object(sys, "executable", str(tmp_path / "LocalWispr.exe"))

        # Create legacy config
        legacy_path = tmp_path / "config.toml"
        legacy_path.write_text("""
[model]
name = "tiny"
""")

        # Make user path unwritable (simulate permission error)
        user_path = tmp_path / "AppData" / "LocalWispr" / "Stable" / "user-settings.toml"

        mocker.patch("localwispr.config.loader._get_appdata_config_path", return_value=user_path)
        mocker.patch("localwispr.config.loader._get_defaults_path", return_value=tmp_path / "config-defaults.toml")

        # Mock mkdir to raise permission error
        mocker.patch("pathlib.Path.mkdir", side_effect=PermissionError("Access denied"))

        from localwispr.config import load_config

        # Should not crash, just use defaults
        config = load_config()
        assert config is not None
