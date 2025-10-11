from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "src" / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))


try:
    import chesscore  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    chesscore = None


@pytest.fixture(scope="session")
def ensure_chesscore() -> object:
    """Skip tests that require the native extension if it is missing."""
    if chesscore is None:
        pytest.skip("chesscore extension is not available; build the C++ core before running this test")
    return chesscore


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
