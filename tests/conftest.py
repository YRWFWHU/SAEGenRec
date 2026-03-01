"""Global test fixtures."""

import os

import pytest


@pytest.fixture(autouse=True)
def _change_to_tmp(tmp_path, monkeypatch):
    """Run each test in a temporary directory to avoid polluting the workspace."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def project_root():
    """Return the real project root for reading raw test data if needed."""
    return os.path.dirname(os.path.dirname(__file__))
