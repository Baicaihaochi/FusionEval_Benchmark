from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from ..errors import ConfigError

def parameters(raw: Mapping[str, Any], names: Sequence[str]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigError("method.parameters must be a mapping")
    value = dict(raw)
    extra = set(value) - set(names)
    missing = set(names) - set(value)
    if missing:
        raise ConfigError("missing method parameter(s): {}".format(", ".join(sorted(missing))))
    if extra:
        raise ConfigError("unsupported method parameter(s): {}".format(", ".join(sorted(extra))))
    return value

def no_parameters(raw: Mapping[str, Any]) -> Dict[str, Any]:
    parameters(raw, ())
    return {}

def optional_mapping(raw: Mapping[str, Any], names: Sequence[str]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigError("method.parameters must be a mapping")
    extra = set(raw) - set(names)
    if extra:
        raise ConfigError("unsupported method parameter(s): {}".format(", ".join(sorted(extra))))
    return dict(raw)

def optional_bool(raw: Mapping[str, Any], name: str, default: bool) -> bool:
    if name not in raw:
        return default
    value = raw[name]
    if not isinstance(value, bool):
        raise ConfigError("method.parameters.{} must be a boolean".format(name))
    return value

def choice(raw: Mapping[str, Any], name: str, default: str, options: Sequence[str]) -> str:
    if name not in raw:
        return default
    value = raw[name]
    if value not in options:
        raise ConfigError(
            "method.parameters.{} must be one of {}".format(name, ", ".join(options))
        )
    return str(value)

def number(value: Any, name: str) -> float:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as exc:
            raise ConfigError("method.parameters.{} must be numeric".format(name)) from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("method.parameters.{} must be numeric".format(name))
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError("method.parameters.{} must be finite".format(name))
    return result

def optional_number(
    raw: Mapping[str, Any],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    closed_min: bool = True,
) -> float:
    if name not in raw:
        return float(default)
    result = number(raw[name], name)
    if minimum is not None:
        allowed = result >= minimum if closed_min else result > minimum
        if not allowed:
            raise ConfigError("method.parameters.{} is outside its valid range".format(name))
    return result

def scale(value: Any) -> float:
    return number(value, "scale")

def unit_interval(value: Any, name: str, *, closed_zero: bool = True) -> float:
    result = number(value, name)
    lower = result >= 0.0 if closed_zero else result > 0.0
    if not lower or result > 1.0:
        raise ConfigError("method.parameters.{} is outside its valid range".format(name))
    return result

def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError("method.parameters.{} must be a positive integer".format(name))
    return value

def optional_positive_int(raw: Mapping[str, Any], name: str, default: int) -> int:
    if name not in raw:
        return int(default)
    return positive_int(raw[name], name)
