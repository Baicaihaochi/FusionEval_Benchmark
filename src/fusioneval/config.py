from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from .errors import ConfigError
from .methods.base import MethodContext, MethodPlugin
from .model_profile import ModelProfile, load_model_profile
from .registry import resolve_method, validate_parameters

@dataclass(frozen=True)
class ExpertSource:
    id: str
    path: Path
    data: Optional[Path] = None

@dataclass(frozen=True)
class RunConfig:
    path: Path
    common_path: Path
    raw: Dict[str, Any]
    common_raw: Dict[str, Any]
    run_id: str
    base: Path
    experts: List[ExpertSource]
    initial_model: Optional[Path]
    calibration_data: Optional[Path]
    method: MethodPlugin
    method_parameters: Dict[str, Any]
    output: Path
    runtime: Dict[str, Any]
    model_profile: Optional[ModelProfile] = None

_RUN_KEYS = {"schema_version", "common", "method", "parameters", "output"}
_COMMON_KEYS = {
    "schema_version", "base", "experts", "initial_model", "calibration_data",
    "output_root", "runtime", "model_profile"
}
_RUNTIME_DEFAULTS = {
    "device": "cpu",
    "save_dtype": "source",
    "max_shard_size_gib": 5,
    "seed": 42,
    "precision": "mergebench",

    "rng_state": None,
}
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX_STATE = re.compile(r"^[0-9a-f]+$")

def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError("{} must be a mapping".format(location))
    return value

def _read_yaml(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError("{} does not exist: {}".format(label, path)) from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError("cannot read {} {}: {}".format(label, path, exc)) from exc
    return dict(_mapping(value, label))

def _resolve(value: str, directory: Path) -> Path:
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = directory / result
    return result.resolve()

def _within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False

def _overlap(left: Path, right: Path) -> bool:
    return _within(left, right) or _within(right, left)

def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("{} must be numeric".format(location))
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError("{} must be finite".format(location))
    return result

def normalize_runtime(value: Any, location: str = "common.runtime") -> Dict[str, Any]:
    runtime_raw = _mapping(value, location)
    extra_runtime = set(runtime_raw) - set(_RUNTIME_DEFAULTS)
    if extra_runtime:
        raise ConfigError(
            "unsupported runtime key(s): {}".format(", ".join(sorted(extra_runtime)))
        )
    runtime = dict(_RUNTIME_DEFAULTS)
    runtime.update(runtime_raw)
    if runtime["device"] not in ("cpu", "cuda"):
        raise ConfigError("{}.device must be cpu or cuda".format(location))
    if runtime["save_dtype"] not in ("source", "bfloat16", "float16", "float32"):
        raise ConfigError(
            "{}.save_dtype must be source, bfloat16, float16, or float32".format(location)
        )
    if runtime["precision"] not in ("mergebench", "float32"):
        raise ConfigError("{}.precision must be mergebench or float32".format(location))
    max_shard = _finite_number(
        runtime["max_shard_size_gib"], "{}.max_shard_size_gib".format(location)
    )
    if max_shard <= 0:
        raise ConfigError("{}.max_shard_size_gib must be positive".format(location))
    runtime["max_shard_size_gib"] = max_shard
    seed = runtime["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("{}.seed must be a non-negative integer".format(location))
    rng_state = runtime["rng_state"]
    if rng_state is not None and (
        not isinstance(rng_state, str)
        or not rng_state
        or len(rng_state) % 2
        or not _HEX_STATE.fullmatch(rng_state)
    ):
        raise ConfigError(
            "{}.rng_state must be null or an even-length lower-case hexadecimal "
            "torch.Generator state".format(location)
        )
    return runtime

def load_config(path: Path) -> RunConfig:
    path = path.resolve()
    raw = _read_yaml(path, "run config")
    extra = set(raw) - _RUN_KEYS
    if extra:
        raise ConfigError("unsupported run key(s): {}".format(", ".join(sorted(extra))))
    if raw.get("schema_version") != 1:
        raise ConfigError("run schema_version must be 1")

    common_value = raw.get("common")
    if not isinstance(common_value, str) or not common_value:
        raise ConfigError("common must reference a YAML file")
    common_path = _resolve(common_value, path.parent)
    common_raw = _read_yaml(common_path, "common config")
    common_extra = set(common_raw) - _COMMON_KEYS
    if common_extra:
        raise ConfigError("unsupported common key(s): {}".format(", ".join(sorted(common_extra))))
    if common_raw.get("schema_version") != 1:
        raise ConfigError("common schema_version must be 1")

    base_value = common_raw.get("base")
    if not isinstance(base_value, str) or not base_value:
        raise ConfigError("common.base must be a path string")
    base = _resolve(base_value, common_path.parent)
    experts_raw = common_raw.get("experts")
    if not isinstance(experts_raw, list) or not experts_raw:
        raise ConfigError("common.experts must contain at least one checkpoint path")
    experts = []
    ids = set()
    for index, item in enumerate(experts_raw):
        if isinstance(item, str):
            expert_id, expert_value = "e{:02d}".format(index), item
            data_value = None
        else:
            item = _mapping(item, "common.experts[{}]".format(index))
            item_extra = set(item) - {"id", "path", "data"}
            if item_extra:
                raise ConfigError("unsupported expert key(s): {}".format(", ".join(sorted(item_extra))))
            expert_id, expert_value = item.get("id"), item.get("path")
            data_value = item.get("data")
        if not isinstance(expert_id, str) or not _SLUG.fullmatch(expert_id):
            raise ConfigError("each expert id must be a short path-safe slug")
        if expert_id in ids:
            raise ConfigError("duplicate expert id: {}".format(expert_id))
        if not isinstance(expert_value, str) or not expert_value:
            raise ConfigError("each expert path must be a non-empty string")
        if data_value is not None and (not isinstance(data_value, str) or not data_value):
            raise ConfigError("expert data must be a non-empty path string")
        ids.add(expert_id)
        experts.append(
            ExpertSource(
                expert_id,
                _resolve(expert_value, common_path.parent),
                _resolve(data_value, common_path.parent) if data_value else None,
            )
        )
    initial_value = common_raw.get("initial_model")
    if initial_value is not None and (not isinstance(initial_value, str) or not initial_value):
        raise ConfigError("common.initial_model must be a non-empty path string")
    initial_model = (
        _resolve(initial_value, common_path.parent) if initial_value is not None else None
    )
    calibration_value = common_raw.get("calibration_data")
    if calibration_value is not None and (
        not isinstance(calibration_value, str) or not calibration_value
    ):
        raise ConfigError("common.calibration_data must be a non-empty path string")
    calibration_data = (
        _resolve(calibration_value, common_path.parent)
        if calibration_value is not None
        else None
    )

    if len(set([base] + [item.path for item in experts])) != 1 + len(experts):
        raise ConfigError("base and expert checkpoint paths must be distinct")
    if initial_model is not None and initial_model == base:
        raise ConfigError("common.initial_model must differ from common.base")

    method_name = raw.get("method")
    if not isinstance(method_name, str):
        raise ConfigError("method must be a string")
    method = resolve_method(method_name)
    if len(experts) < method.minimum_experts:
        raise ConfigError(
            "method {} requires at least {} expert checkpoint(s)".format(
                method.name, method.minimum_experts
            )
        )
    if method.requires_initial_model and initial_model is None:
        raise ConfigError("method {} requires common.initial_model".format(method.name))
    if method.data_mode == "per_expert":
        missing = [item.id for item in experts if item.data is None]
        if missing:
            raise ConfigError(
                "method {} requires expert data for: {}".format(
                    method.name, ", ".join(missing)
                )
            )
    elif method.data_mode == "shared" and calibration_data is None:
        raise ConfigError("method {} requires common.calibration_data".format(method.name))
    params = validate_parameters(
        method,
        raw.get("parameters", {}),
        MethodContext(tuple(item.id for item in experts)),
    )

    output_name = raw.get("output")
    if not isinstance(output_name, str) or not _SLUG.fullmatch(output_name):
        raise ConfigError("output must be a short path-safe slot name")
    output_root_value = common_raw.get("output_root")
    if not isinstance(output_root_value, str) or not output_root_value:
        raise ConfigError("common.output_root must be a path string")
    output_root = _resolve(output_root_value, common_path.parent)
    output = output_root / output_name
    source_paths = [base] + [item.path for item in experts]
    if initial_model is not None:
        source_paths.append(initial_model)
    if any(_overlap(output, source) for source in source_paths):
        raise ConfigError("output must not overlap a source checkpoint")

    runtime = normalize_runtime(common_raw.get("runtime", {}))

    profile_value = common_raw.get("model_profile")
    if profile_value is not None and (not isinstance(profile_value, str) or not profile_value):
        raise ConfigError("common.model_profile must reference a YAML file")
    model_profile = (
        load_model_profile(_resolve(profile_value, common_path.parent))
        if profile_value is not None
        else None
    )

    return RunConfig(
        path=path,
        common_path=common_path,
        raw=raw,
        common_raw=common_raw,
        run_id=output_name,
        base=base,
        experts=experts,
        initial_model=initial_model,
        calibration_data=calibration_data,
        method=method,
        method_parameters=params,
        output=output,
        runtime=runtime,
        model_profile=model_profile,
    )
