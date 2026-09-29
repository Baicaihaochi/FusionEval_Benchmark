from ..base import MethodPlugin
from ..fields import optional_bool, parameters, scale, unit_interval
from .._continual_sparse import (
    TIES_PARAMETERS,
    reject_retired_stage,
    two_input_contextualize,
)
from ...errors import ConfigError
from ...protocol import KernelRequirements

def normalize(raw):
    reject_retired_stage(raw, "ties")
    value = parameters(raw, TIES_PARAMETERS)
    if value["merge_func"] != "mean":
        raise ConfigError(
            "continual ties uses the paper-standard disjoint mean; merge_func=sum is the "
            "legacy static variant and needs its own continual protocol"
        )
    if optional_bool(value, "exclude_edge", False):
        raise ConfigError(
            "continual ties requires exclude_edge=false; an edge fallback is undefined "
            "for the fixed-anchor continual protocol"
        )
    return {
        "density": unit_interval(value["density"], "density", closed_zero=False),
        "scale": scale(value["scale"]),
        "merge_func": "mean",
        "exclude_edge": False,
        "experts_included": value["experts_included"],
        "participant_count": value["participant_count"],
    }

PLUGIN = MethodPlugin(
    name="continual_ties",
    aliases=(),
    op_code="c05",

    contract_version=1,
    requirements=KernelRequirements("flat", "global", 2, True),
    parameter_scope="all_floating",
    aggregation="recursive_fixed-anchor_disjoint_mean",
    contract_tags=(
        "continual_standard", "fixed_anchor_sparse", "decoder_full_edge_extension",
    ),
    user_parameters=TIES_PARAMETERS,
    runtime_parameters=TIES_PARAMETERS,
    normalize=normalize,
    contextualize=two_input_contextualize("ties"),
    minimum_experts=2,
    input_mode="experts",
)
