from ..base import MethodPlugin
from ..fields import optional_mapping, positive_int, optional_bool, optional_number, choice, number
from ...errors import ConfigError
from ...protocol import KernelRequirements

def normalize(raw):
    raw = optional_mapping(raw, ("examples", "estimator", "samples", "normalize_fishers", "favor_target_model", "fisher_floor", "coefficients"))
    coefficients = raw.get("coefficients")
    if coefficients is not None:
        if not isinstance(coefficients, list) or not coefficients:
            raise ConfigError("coefficients must be a nonempty list in expert order")
        coefficients = [number(x, "coefficients") for x in coefficients]
        if min(coefficients) < 0 or sum(coefficients) <= 0:
            raise ConfigError("coefficients must be nonnegative with positive sum")
    return {
        "examples": positive_int(raw["examples"], "examples") if "examples" in raw else 256,
        "estimator": choice(raw, "estimator", "exact", ("exact", "sampled")),
        "samples": positive_int(raw.get("samples", 1), "samples"),
        "normalize_fishers": optional_bool(raw, "normalize_fishers", True),
        "favor_target_model": optional_bool(raw, "favor_target_model", True),
        "fisher_floor": optional_number(raw, "fisher_floor", 1e-6, minimum=0, closed_min=False),
        "coefficients": coefficients,
    }

def contextualize(parameters, context):
    values = parameters["coefficients"]
    if values is not None and len(values) != len(context.expert_ids):
        raise ConfigError("one coefficient required per expert, in expert order")
    return parameters

PLUGIN = MethodPlugin(
    name="fisher",
    aliases=("fisher_merging",),
    op_code="d01",
    contract_version=2,
    requirements=KernelRequirements("flat", "none", 1, False),
    parameter_scope="all_floating",
    aggregation="predictive_fisher_weighted_mean",
    contract_tags=(
        "paper_equation",
        "decoder_next_token_distribution_extension",
        "target_untied_head_shared_embedding_extension",
    ),
    user_parameters=("examples", "estimator", "samples", "normalize_fishers", "favor_target_model", "fisher_floor", "coefficients"),
    runtime_parameters=("examples", "estimator", "samples", "normalize_fishers", "favor_target_model", "fisher_floor", "coefficients"),
    normalize=normalize,
    contextualize=contextualize,
    execution="data",
    requires_data=True,
    data_mode="per_expert",
)
