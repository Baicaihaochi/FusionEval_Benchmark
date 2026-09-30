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

def resolve_method(name: str) -> MethodPlugin:
    canonical = OP_CODES.get(name, ALIASES.get(name, name))
    if canonical not in SPECS:
        raise ConfigError(
            "unknown method {!r}; expected one of {}".format(
                name, ", ".join(sorted(SPECS))
            )
        )
    return SPECS[canonical]

def validate_parameters(
    spec: MethodPlugin,
    raw: Mapping[str, Any],
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    return spec.configure(raw, context)

def registry_payload() -> Dict[str, Any]:
    return {name: SPECS[name].to_dict() for name in sorted(SPECS)}
