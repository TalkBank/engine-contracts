"""Shared test configuration for engine contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).parent.parent / "results"


@pytest.fixture
def results_dir() -> Path:
    """Path to the committed probe results directory."""
    return RESULTS_DIR
