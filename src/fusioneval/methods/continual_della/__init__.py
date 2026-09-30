from ..base import MethodPlugin
from ..fields import number, optional_bool, parameters, scale
from .._continual_sparse import (
    DELLA_PARAMETERS,
    reject_retired_stage,
    two_input_contextualize,
)
from ...errors import ConfigError
from ...protocol import KernelRequirements

def normalize(raw):
    reject_retired_stage(raw, "della")
    value = parameters(raw, DELLA_PARAMETERS)
    if optional_bool(value, "exclude_edge", False):
        raise ConfigError(
            "continual della requires exclude_edge=false; an edge fallback is undefined "
            "for the fixed-anchor continual protocol"
        )
    drop_rate = number(value["drop_rate"], "drop_rate")
    window = number(value["window"], "window")
    low, high = 1.0 - drop_rate - window / 2.0, 1.0 - drop_rate + window / 2.0
    if window < 0.0 or not (0.0 < low <= high <= 1.0):
        raise ConfigError("DELLA survival-probability range is invalid")
    seed = value["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("method.parameters.seed must be a non-negative integer")
    return {
        "drop_rate": drop_rate,
        "window": window,
        "scale": scale(value["scale"]),
        "seed": seed,
        "exclude_edge": False,
        "experts_included": value["experts_included"],
        "participant_count": value["participant_count"],
    }

PLUGIN = MethodPlugin(
    name="continual_della",
    aliases=(),
    op_code="c06",

    contract_version=2,
    requirements=KernelRequirements("full_rows", "none", 1, False, stochastic=True),
    parameter_scope="all_floating",
    aggregation="recursive_fixed-anchor_disjoint_mean",
    contract_tags=(
        "continual_standard", "fixed_anchor_sparse", "official_code_compat",
        "decoder_full_edge_extension",
    ),
    user_parameters=DELLA_PARAMETERS,
    runtime_parameters=DELLA_PARAMETERS,
    normalize=normalize,
    contextualize=two_input_contextualize("della"),
    minimum_experts=2,
    input_mode="experts",
    compute_precision="float32",
)
