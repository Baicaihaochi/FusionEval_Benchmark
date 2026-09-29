from ..base import ContractVariant, MethodPlugin
from ..fields import choice, optional_bool, optional_mapping, scale, unit_interval
from ...errors import ConfigError
from ...protocol import KernelRequirements

def normalize(raw):
    raw = optional_mapping(
        raw, ("density", "scale", "merge_func", "exclude_edge", "leftover_edge")
    )
    if "density" not in raw or "scale" not in raw:
        raise ConfigError("missing method parameter(s): density, scale")
    return {
        "density": unit_interval(raw["density"], "density", closed_zero=False),
        "scale": scale(raw["scale"]),
        "merge_func": choice(raw, "merge_func", "mean", ("mean", "sum")),
        "exclude_edge": optional_bool(raw, "exclude_edge", False),
        "leftover_edge": choice(raw, "leftover_edge", "base", ("base", "mean", "ta")),
    }

def contract(value):
    variants = []
    tags = []
    remove = []
    if value["merge_func"] == "sum":
        variants.append("legacy_sum")
        tags.append("legacy_sum_variant")
        remove.append("paper_standard")
    if value["exclude_edge"]:
        variants.append("edge_{}".format(value["leftover_edge"]))
        tags.append("explicit_edge_leftover_variant")
        remove.append("paper_standard")
    return ContractVariant("_".join(variants) or "standard", tuple(tags), tuple(remove))

PLUGIN = MethodPlugin(
    name="ties", aliases=(), op_code="o03", contract_version=2,
    requirements=KernelRequirements("flat", "global", 2, True),
    parameter_scope="all_floating", aggregation="disjoint_mean",
    contract_tags=("paper_standard", "decoder_full_edge_extension"),
    user_parameters=("density", "scale", "merge_func", "exclude_edge", "leftover_edge"),
    runtime_parameters=("density", "scale", "merge_func", "exclude_edge", "leftover_edge"),
    normalize=normalize, contractualize=contract,
)
