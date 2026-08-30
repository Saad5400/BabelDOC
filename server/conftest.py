"""Fixtures shared by the server test modules.

The three stateless endpoints (/v1/compose, /v1/overlay, /v1/convert) each need
the same app: a configured token, a scratch DATA_DIR, and no job worker. Kept
here so the setup is one story rather than three copies of it that can drift.
"""

import pytest
from fastapi.testclient import TestClient

TOKEN = "test-token"  # noqa: S105 - a test fixture's shared secret


@pytest.fixture()
def anyio_backend():
    """Run every `@pytest.mark.anyio` test on asyncio, and only asyncio.

    anyio's plugin would otherwise parameterise each async test over asyncio
    AND trio; trio is not a dependency of this service and uvicorn does not run
    on it, so the second run would be testing a stack we never ship.
    """
    return "asyncio"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """The app with the token configured and the job worker stood down.

    None of the stateless endpoints touch job state, and starting the worker
    would drag babeldoc's heavy import into every one of these tests.
    """
    from server import app as app_module
    from server import config

    monkeypatch.setattr(config, "DOCTRANSLATE_TOKEN", TOKEN)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(app_module.jobs, "start_worker", lambda: None)

    with TestClient(app_module.app) as test_client:
        yield test_client
