from __future__ import annotations

from typing import Any, Mapping

from .fields import positive_int
from ..errors import ConfigError

PARTICIPANT_COUNT = 2

TIES_PARAMETERS = (
    "density", "scale", "merge_func", "exclude_edge", "experts_included",
    "participant_count",
)
DARE_PARAMETERS = (
    "drop_rate", "scale", "seed", "exclude_edge", "experts_included",
    "participant_count",
)
DELLA_PARAMETERS = (
    "drop_rate", "window", "scale", "seed", "exclude_edge", "experts_included",
    "participant_count",
)

RETIRED_STAGE_MESSAGE = (
    "continual {name} no longer accepts 'stage': pass 'experts_included' (the number of "
    "experts already included after this operation, i.e. t+1) and 'participant_count' "
    "(the number of CURRENT participants, always 2 for a two-input continual fusion)."
)

def reject_retired_stage(raw, name: str = "sparse") -> None:
    if isinstance(raw, Mapping) and "stage" in raw:
        raise ConfigError(RETIRED_STAGE_MESSAGE.format(name=name))

def check_counters(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    experts_included = positive_int(value["experts_included"], "experts_included")
    if experts_included < 2:
        raise ConfigError(
            "continual {} fuses only once at least two experts are included "
            "(experts_included >= 2); Stage 0 is the initial expert and runs no "
            "kernel".format(name)
        )
    count = positive_int(value["participant_count"], "participant_count")
    if count != PARTICIPANT_COUNT:
        raise ConfigError(
            "continual {} has exactly {} current participants (previous and incoming); "
            "got participant_count={}".format(name, PARTICIPANT_COUNT, count)
        )
    return value

def two_input_contextualize(name: str):
    def contextualize(value, context):
        if context.expert_ids != ("previous", "incoming"):
            raise ConfigError(
                "continual {} requires experts previous,incoming in order; the fixed "
                "common anchor is common.base, never a third expert".format(name)
            )
        return check_counters(value, name)

    return contextualize

def encode_by_name(parameters, names) -> Mapping[str, Any]:

    values = tuple(parameters)
    if len(values) != len(names):
        raise ValueError(
            "continual kernel received {} parameters, expected {}: {}".format(
                len(values), len(names), ", ".join(names)
            )
        )
    return dict(zip(names, values))
