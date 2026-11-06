"""Shared pytest fixtures for the client tests."""
import tempfile

import pytest


@pytest.fixture(autouse=True)
def _isolated_tempdir(tmp_path, monkeypatch):
    """Keep test output (e.g. fake stage 2 logs in <tmp>/ecm_stage2_logs) out
    of the real temp dir, where it would mix with logs from real runs."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
