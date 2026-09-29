from ..base import MethodContext, MethodPlugin
from ..fields import parameters, positive_int
from ...errors import ConfigError
from ...protocol import KernelRequirements

def normalize(raw):
    value = parameters(raw, ("seen_count",))
    return {"seen_count": positive_int(value["seen_count"], "seen_count")}

def contextualize(value, context: MethodContext):
    if len(context.expert_ids) != 1:
        raise ConfigError("online_average requires exactly one incoming expert")
    return value

PLUGIN = MethodPlugin(
    name="online_average",
    aliases=("continual_average",),
    op_code="c01",
    contract_version=1,
    requirements=KernelRequirements("flat", "none", 1, False),
    parameter_scope="all_floating",
    aggregation="recursive_uniform_expert_mean",
    contract_tags=("continual_standard", "decoder_full_edge_extension"),
    user_parameters=("seen_count",),
    runtime_parameters=("seen_count",),
    normalize=normalize,
    contextualize=contextualize,
    minimum_experts=1,
)
