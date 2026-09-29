from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .errors import ConfigError
from .methods import discover
from .methods.base import MethodContext, MethodPlugin

SPECS = discover()
OP_CODES = {plugin.op_code: plugin.name for plugin in SPECS.values()}
ALIASES = {}
for plugin in SPECS.values():
    for alias in plugin.aliases:
        if alias in ALIASES:
            raise ConfigError("duplicate method alias: {}".format(alias))
        ALIASES[alias] = plugin.name

if len({plugin.op_code for plugin in SPECS.values()}) != len(SPECS):
    raise ConfigError("method op_code values must be unique")
if set(ALIASES) & set(SPECS):
    raise ConfigError("method alias collides with a canonical name")
if any(plugin.compute_precision not in {"runtime", "float32"} for plugin in SPECS.values()):
    raise ConfigError("method compute_precision must be runtime or float32")
if any(plugin.execution not in {"tensor", "data"} for plugin in SPECS.values()):
    raise ConfigError("method execution must be tensor or data")
if any(plugin.requires_data and plugin.execution != "data" for plugin in SPECS.values()):
    raise ConfigError("data-requiring methods must use the data executor")
if any(plugin.data_mode not in {"none", "per_expert", "shared"} for plugin in SPECS.values()):
    raise ConfigError("method data_mode must be none, per_expert, or shared")
if any(plugin.requires_data != (plugin.data_mode != "none") for plugin in SPECS.values()):
    raise ConfigError("requires_data and data_mode disagree")
if any(plugin.requires_initial_model and plugin.execution != "data" for plugin in SPECS.values()):
    raise ConfigError("initial-model methods must use the data executor")
if any(plugin.minimum_experts < 1 for plugin in SPECS.values()):
    raise ConfigError("method minimum_experts must be positive")
if any(plugin.input_mode not in {"deltas", "experts"} for plugin in SPECS.values()):
    raise ConfigError("method input_mode must be deltas or experts")

def resolve_method(name: str) -> MethodPlugin:
    canonical = OP_CODES.get(name, ALIASES.get(name, name))
    try:
        return SPECS[canonical]
    except KeyError as exc:
        raise ConfigError(
            "unknown method {!r}; expected one of {}".format(
                name, ", ".join(sorted(SPECS))
            )
        ) from exc

def validate_parameters(
    spec: MethodPlugin,
    raw: Mapping[str, Any],
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    return spec.configure(raw, context)

def registry_payload() -> Dict[str, Any]:
    return {name: SPECS[name].to_dict() for name in sorted(SPECS)}
