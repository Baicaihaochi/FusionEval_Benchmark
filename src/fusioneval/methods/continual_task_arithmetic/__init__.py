from ..base import MethodContext, MethodPlugin
from ..fields import parameters, positive_int, scale
from ...errors import ConfigError
from ...protocol import KernelRequirements

_RETIRED_STAGE_MESSAGE = (
    "continual_task_arithmetic no longer accepts 'stage': pass 'experts_included' (the "
    "number of experts already included after this operation, i.e. t+1). The retired "
    "parameter also selected a common-base initialisation that section 0 removes."
)

def normalize(raw):
    if isinstance(raw, dict) and "stage" in raw:
        raise ConfigError(_RETIRED_STAGE_MESSAGE)
    value = parameters(raw, ("scale", "experts_included"))
    return {
        "scale": scale(value["scale"]),
        "experts_included": positive_int(value["experts_included"], "experts_included"),
    }

def contextualize(value, context: MethodContext):
    if value["experts_included"] < 2:
        raise ConfigError(
            "continual_task_arithmetic requires experts_included >= 2 (the initial expert "
            "plus at least one arrival); Stage 0 is never fused"
        )
    if context.expert_ids != ("anchor", "incoming"):
        raise ConfigError(
            "continual_task_arithmetic requires experts anchor,incoming in order"
        )
    return value

PLUGIN = MethodPlugin(
    name="continual_task_arithmetic",
    aliases=("continual_ta",),
    op_code="c02",

    contract_version=2,
    requirements=KernelRequirements("flat", "none", 1, False),
    parameter_scope="all_floating",
    aggregation="recursive_common-anchor_task-vector-sum",
    contract_tags=("continual_standard", "decoder_full_edge_extension"),
    user_parameters=("scale", "experts_included"),
    runtime_parameters=("scale", "experts_included"),
    normalize=normalize,
    contextualize=contextualize,
    minimum_experts=1,
    input_mode="experts",
)
