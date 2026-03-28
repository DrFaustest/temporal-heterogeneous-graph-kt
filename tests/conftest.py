from __future__ import annotations

import itertools
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_TMP_ROOT = ROOT / ".tmp_pytest"
_COUNTER = itertools.count()


@pytest.fixture
def tmp_path() -> Path:
    path = _TMP_ROOT / f"case_{next(_COUNTER)}"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
