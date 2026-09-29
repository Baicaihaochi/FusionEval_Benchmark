from __future__ import annotations

import gc
import importlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from .checkpoint import inspect_checkpoints
from .config import RunConfig
from .data_runtime import data_inventory
from .engine import (
    copy_metadata,
    roles,
    save_tensors,
    validate_output_tensors,
    validate_saved,
)
from .errors import CheckpointError

@dataclass(frozen=True)
class DataResult:
    tensors: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    artifacts: Mapping[str, Path] = field(default_factory=dict)

def _publish_artifacts(result: DataResult, workspace: Path, staging: Path) -> Dict[str, Any]:
    copied = {}
    workspace = workspace.resolve()
    for relative, source in sorted(result.artifacts.items()):
        destination = Path(relative)
        if destination.is_absolute() or ".." in destination.parts or not destination.parts:
            raise CheckpointError("artifact destination must be a safe relative path")
        source = source.resolve()
        try:
            source.relative_to(workspace)
        except ValueError as exc:
            raise CheckpointError("runner artifact is outside its workspace") from exc
        if not source.exists():
            raise CheckpointError("runner artifact does not exist: {}".format(source))
        target = staging / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            os.rename(source, target)
            copied[str(destination)] = {
                "type": "file",
                "bytes": target.stat().st_size,
            }
        elif source.is_dir():
            os.rename(source, target)
            files = {
                str(path.relative_to(target)): {"bytes": path.stat().st_size}
                for path in sorted(target.rglob("*"))
                if path.is_file()
            }
            if not files:
                raise CheckpointError("runner artifact directory is empty: {}".format(source))
            copied[str(destination)] = {"type": "directory", "files": files}
        else:
            raise CheckpointError("runner artifact is not a file or directory: {}".format(source))
    return copied

def run(config: RunConfig, *, provenance=None):
    import torch

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    inspection = inspect_checkpoints(config)
    if config.output.exists():
        raise CheckpointError("output already exists: {}".format(config.output))
    if config.runtime["device"] == "cuda" and not torch.cuda.is_available():
        raise CheckpointError("runtime.device=cuda but CUDA is unavailable")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    if config.method.data_mode == "per_expert":
        data_paths = [item.data for item in config.experts]
        assert all(path is not None for path in data_paths)
        inventory = data_inventory(data_paths)
    elif config.method.data_mode == "shared":
        assert config.calibration_data is not None
        inventory = data_inventory([config.calibration_data])
    else:
        inventory = []
    if config.method.local_name == "continual_regmean":

        expected_rows = config.method_parameters["examples"]
        if len(inventory) != 1 or inventory[0]["rows"] != expected_rows:
            raise CheckpointError(
                "continual RegMean requires exactly {} rows for the incoming expert".format(
                    expected_rows
                )
            )
    preparation_done = time.perf_counter()
    effective_precision = (
        "float32" if config.method.compute_precision == "float32" else config.runtime["precision"]
    )
    tied = bool(
        json.loads((config.base / "config.json").read_text(encoding="utf-8"))[
            "tie_word_embeddings"
        ]
    )
    staging = Path(tempfile.mkdtemp(prefix=".{}-out.".format(config.run_id), dir=str(config.output.parent)))
    workspace = Path(tempfile.mkdtemp(prefix=".{}-work.".format(config.run_id), dir=str(config.output.parent)))
    try:
        module = importlib.import_module(config.method.runner_module)
        method_started = time.perf_counter()
        result = module.execute(config, workspace)
        method_done = time.perf_counter()
        if not isinstance(result, DataResult):
            raise CheckpointError("data method runner returned an invalid result")
        tensors = dict(result.tensors)
        validate_output_tensors(config, tensors)
        coverage: Dict[str, int] = {}
        for key, value in tensors.items():
            if value.is_floating_point():
                for group in roles(key, tied):
                    coverage[group] = coverage.get(group, 0) + value.numel()
        save_started = time.perf_counter()
        metadata_source = config.initial_model or config.base
        copy_metadata(metadata_source, staging)
        mapping = save_tensors(
            tensors, staging, int(config.runtime["max_shard_size_gib"] * 2**30)
        )
        artifacts = _publish_artifacts(result, workspace, staging)
        validation = validate_saved(staging, tensors, tied)
        files = {
            name: {"bytes": (staging / name).stat().st_size}
            for name in sorted(set(mapping.values()))
        }
        index = staging / "model.safetensors.index.json"
        if index.is_file():
            files[index.name] = {"bytes": index.stat().st_size}
        finished = time.perf_counter()
        finished_at = datetime.now(timezone.utc)
        timing = {
            "protocol": "mergebench-4.3-wall-clock-v1",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "preparation_seconds": preparation_done - started,
            "method_seconds": method_done - method_started,
            "save_validation_seconds": finished - save_started,
            "algorithm_runtime_seconds": finished - started,
            "single_gpu_algorithm_seconds": method_done - method_started,
            "algorithm_device_count": 1,
            "algorithm_runtime_scope": "single_gpu_sequential",
            "visible_cuda_device_count": (
                int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            ),
        }
        contract = config.method.effective_contract(config.method_parameters)
        manifest = {
            "status": "PASS",
            "method": config.method.local_name,
            "contract_id": contract["id"],
            "contract_variant": contract["variant"],
            "contract_tags": contract["tags"],
            "requested_precision": config.runtime["precision"],
            "effective_precision": effective_precision,
            "model_profile": config.model_profile.name if config.model_profile else "auto",
            "parameters": config.method_parameters,
            "source_inspection": inspection,
            "data": inventory,
            "coverage": coverage,
            "diagnostics": dict(result.diagnostics),
            "validation": validation,
            "timing": timing,
            "elapsed_seconds": timing["algorithm_runtime_seconds"],
            "weight_files": files,
            "artifacts": artifacts,
        }
        if provenance is not None:
            manifest["orchestration"] = dict(provenance)
        (staging / "fusioneval_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging, config.output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
