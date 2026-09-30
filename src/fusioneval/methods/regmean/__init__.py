from ..base import MethodPlugin
from ..fields import choice, optional_mapping, positive_int, unit_interval
from ...errors import ConfigError
from ...protocol import KernelRequirements

def normalize(raw):
    raw = optional_mapping(raw, ("alpha", "examples", "leftover_edge", "leftover_1d",
                                 "variant", "solver", "gram_dtype", "edge_head"))
    if "alpha" not in raw:
        raise ConfigError("missing method parameter(s): alpha")
    return {
        "alpha": unit_interval(raw["alpha"], "alpha"),
        "examples": positive_int(raw["examples"], "examples") if "examples" in raw else 256,
        "leftover_edge": choice(raw, "leftover_edge", "mean", ("base", "mean")),
        "leftover_1d": choice(raw, "leftover_1d", "mean", ("base", "mean")),
        "variant": choice(raw, "variant", "raw", ("raw", "white")),
        "solver": choice(raw, "solver", "exact", ("exact", "pinv", "pinv1e12")),
        "gram_dtype": choice(raw, "gram_dtype", "float32", ("float32", "model")),
        "edge_head": choice(raw, "edge_head", "leftover", ("leftover", "regmean")),
    }

PLUGIN = MethodPlugin(
    name="regmean",
    aliases=("regression_mean",),
    op_code="d02",
    contract_version=1,
    requirements=KernelRequirements("full_rows", "none", 1, False),
    parameter_scope="linear_weights_with_expert_mean_fallback",
    aggregation="activation_gram_regression",
    contract_tags=(
        "paper_standard",
        "reference_numeric_compat",
        "decoder_full_edge_extension",
    ),
    user_parameters=("alpha", "examples", "leftover_edge", "leftover_1d",
                     "variant", "solver", "gram_dtype", "edge_head"),
    runtime_parameters=("alpha", "examples", "leftover_edge", "leftover_1d",
                        "variant", "solver", "gram_dtype", "edge_head"),
    normalize=normalize,
    execution="data",
    requires_data=True,
    data_mode="per_expert",
)
