from ..base import ContractVariant, MethodPlugin
from ..fields import choice, number, optional_bool, optional_mapping, scale
from ...errors import ConfigError
from ...protocol import KernelRequirements

def normalize(raw):
    raw = optional_mapping(raw, ("drop_rate", "scale", "exclude_edge", "leftover_edge"))
    if "drop_rate" not in raw or "scale" not in raw:
        raise ConfigError("missing method parameter(s): drop_rate, scale")
    drop_rate = number(raw["drop_rate"], "drop_rate")
    if not 0.0 <= drop_rate < 1.0:
        raise ConfigError("DARE drop_rate must satisfy 0 <= drop_rate < 1")
    return {
        "drop_rate": drop_rate,
        "scale": scale(raw["scale"]),
        "exclude_edge": optional_bool(raw, "exclude_edge", False),
        "leftover_edge": choice(raw, "leftover_edge", "base", ("base", "mean", "ta")),
    }

def contract(value):
    if not value["exclude_edge"]:
        return ContractVariant()
    return ContractVariant(
        "edge_{}".format(value["leftover_edge"]),
        ("explicit_edge_leftover_variant",),
        ("paper_standard",),
    )

PLUGIN = MethodPlugin(
    name="dare", aliases=(), op_code="o04", contract_version=1,
    requirements=KernelRequirements("flat", "global", 2, True, stochastic=True),
    parameter_scope="all_floating", aggregation="task_vector_sum",
    contract_tags=("paper_standard", "mergebench_numeric_compat", "decoder_full_edge_extension"),
    user_parameters=("drop_rate", "scale", "exclude_edge", "leftover_edge"),
    runtime_parameters=("drop_rate", "scale", "exclude_edge", "leftover_edge"),
    normalize=normalize, contractualize=contract,
)
