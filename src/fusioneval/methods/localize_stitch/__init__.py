from ..base import ContractVariant, MethodPlugin
from ..fields import optional_mapping, unit_interval, choice
from ...protocol import KernelRequirements

def normalize(raw):
    raw = optional_mapping(raw, ("density", "scope"))
    density = unit_interval(raw.get("density", 0.05), "density", closed_zero=False)
    return {"density": density, "comparison": "strict",
            "scope": choice(raw, "scope", "transformer_blocks", ("transformer_blocks", "all_floating"))}

def contract(value):
    if value["scope"] == "all_floating":
        return ContractVariant("full_decoder", ("decoder_full_edge_extension",), ("paper_standard", "official_code_compat"))
    if value["density"] == 0.05:
        return ContractVariant()
    return ContractVariant(
        "custom_density", ("explicit_density_variant",), ("paper_standard",)
    )

PLUGIN = MethodPlugin(
    name="localize_stitch", aliases=("dls", "dataless_localize_and_stitch"),
    op_code="o07", contract_version=2,
    requirements=KernelRequirements("flat", "global", 2, True),
    parameter_scope="configurable_transformer_blocks", aggregation="overlap_mean",
    contract_tags=("paper_standard", "official_code_compat", "decoder_block_mapping"),
    user_parameters=("density", "scope"), runtime_parameters=("density", "comparison", "scope"),
    normalize=normalize, contractualize=contract,
)
