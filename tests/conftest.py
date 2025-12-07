from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_SRC = REPO_ROOT / "src" / "python"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

try:
    import chesscore
except ImportError:
    chesscore = None


@pytest.fixture(scope="session")
def ensure_chesscore() -> object:
    if chesscore is None:
        pytest.skip("hiányzik a chesscore modul")
    return chesscore


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
