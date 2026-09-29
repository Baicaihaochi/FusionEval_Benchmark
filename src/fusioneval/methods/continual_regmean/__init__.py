from ..base import MethodContext, MethodPlugin
from ..fields import choice, optional_mapping, positive_int, unit_interval
from ...errors import ConfigError
from ...protocol import KernelRequirements

def normalize(raw):
    if isinstance(raw, dict) and "stage" in raw:
        raise ConfigError(
            "continual_regmean no longer accepts 'stage': pass 'experts_included' (the "
            "number of experts already included after this operation, i.e. t+1, with 1 "
            "meaning the stage-0 statistics initialisation)."
        )
    raw = optional_mapping(
        raw, ("alpha", "examples", "leftover_edge", "leftover_1d", "experts_included")
    )
    missing = [name for name in ("alpha", "experts_included") if name not in raw]
    if missing:
        raise ConfigError("missing method parameter(s): {}".format(", ".join(missing)))
    return {
        "alpha": unit_interval(raw["alpha"], "alpha"),
        "examples": positive_int(raw["examples"], "examples") if "examples" in raw else 256,
        "leftover_edge": choice(raw, "leftover_edge", "mean", ("base", "mean")),
        "leftover_1d": choice(raw, "leftover_1d", "mean", ("base", "mean")),
        "experts_included": positive_int(
            raw["experts_included"], "experts_included"
        ),
    }

def contextualize(value, context: MethodContext):
    if context.expert_ids != ("incoming",):
        raise ConfigError("continual_regmean requires one expert named incoming")
    return value

PLUGIN = MethodPlugin(
    name="continual_regmean",
    aliases=("recursive_regmean",),
    op_code="c04",

    contract_version=2,
    requirements=KernelRequirements("full_rows", "none", 1, False),
    parameter_scope="recursive_linear_weights_with_expert_mean_fallback",
    aggregation="recursive_activation_gram_regression",
    contract_tags=(
        "continual_standard",
        "mergebench_numeric_compat",
        "decoder_full_edge_extension",
    ),
    user_parameters=("alpha", "examples", "leftover_edge", "leftover_1d", "experts_included"),
    runtime_parameters=("alpha", "examples", "leftover_edge", "leftover_1d", "experts_included"),
    normalize=normalize,
    contextualize=contextualize,
    execution="data",
    requires_data=True,
    data_mode="per_expert",
    minimum_experts=1,
    input_mode="experts",

    compute_precision="float32",
)
