from __future__ import annotations

from pathlib import Path

from ...data_engine import DataResult
from ...rgfast import run_regmean

def execute(config, workspace: Path) -> DataResult:
    return run_regmean(config, workspace)
