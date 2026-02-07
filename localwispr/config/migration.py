"""Legacy configuration migration for LocalWispr."""

import copy
import logging
import sys
import tomllib
from pathlib import Path

from localwispr.config.types import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def _migrate_legacy_config() -> None:
    r"""One-time migration from dist\LocalWispr\config.toml to AppData.

    Note: Legacy config.toml is NOT deleted after migration. It remains
    as a backup and will be ignored in favor of AppData settings.
    """
    if not getattr(sys, 'frozen', False):
        return  # Only for frozen builds

    legacy_path = Path(sys.executable).parent / "config.toml"

    from localwispr.config.loader import _get_appdata_config_path, _deep_merge

    user_path = _get_appdata_config_path()

    # Already migrated or no legacy config
    if user_path.exists() or not legacy_path.exists():
        return

    logger.info("Migrating legacy config from %s to %s", legacy_path, user_path)

    try:
        user_path.parent.mkdir(parents=True, exist_ok=True)
        with open(legacy_path, "rb") as f:
            legacy_config = tomllib.load(f)
        # Merge with defaults to ensure all required fields exist
        full_config = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), legacy_config)

        from localwispr.config.saver import save_config

        save_config(full_config, user_path)
        logger.info("Migration complete - legacy file kept as backup")
    except Exception as e:
        logger.warning("Migration failed (non-fatal): %s", e)
