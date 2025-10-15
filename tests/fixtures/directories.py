"""pytest configuration."""

import pytest
from pathlib import Path

@pytest.fixture
def data_directory():
    """Return the directory where various reference files are stored."""
    return Path(__file__).parents[1] / 'data'
