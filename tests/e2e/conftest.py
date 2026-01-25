"""E2E-specific pytest fixtures.

This module provides fixtures specifically for end-to-end tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def e2e_timeout():
    """Provide a reasonable timeout for E2E tests.

    Returns:
        Timeout in seconds for E2E operations.
    """
    return 10.0  # 10 seconds for full workflows
