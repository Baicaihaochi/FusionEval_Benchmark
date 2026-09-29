from ..base import MethodPlugin
from ..fields import number, optional_mapping, optional_number, optional_positive_int, unit_interval
from ...errors import ConfigError
from ...protocol import KernelRequirements

def _positive(raw, name, default):
    value = number(raw.get(name, default), name)
    if value <= 0.0:
        raise ConfigError("method.parameters.{} must be positive".format(name))
    return value

def normalize(raw):
    raw = optional_mapping(
        raw,
        (
            "ridge_lambda",
            "anchor_rho",
            "teacher_alpha",
            "covariance_eps",
            "examples",
            "prefill_chunk_size",
        ),
    )
    return {
        "ridge_lambda": optional_number(raw, "ridge_lambda", 1e-5, minimum=0.0),
        "anchor_rho": number(raw.get("anchor_rho", 0.5), "anchor_rho"),
        "teacher_alpha": unit_interval(raw.get("teacher_alpha", 0.15), "teacher_alpha"),
        "covariance_eps": _positive(raw, "covariance_eps", 1e-8),
        "examples": optional_positive_int(raw, "examples", 256),
        "prefill_chunk_size": optional_positive_int(raw, "prefill_chunk_size", 256),
    }

PLUGIN = MethodPlugin(
    name="featcal",
    aliases=("feature_calibration",),
    op_code="d09",
    contract_version=3,
    requirements=KernelRequirements("full_rows", "none", 1, False),
    parameter_scope="decoder_layer_linear_weights_forward_order",
    aggregation="feature_drift_closed_form_calibration",
    contract_tags=(
        "paper_closed_form_objective",
        "post_merge_calibration",
        "architecture_preserving",
        "decoder_linear_mapping",
    ),
    user_parameters=(
        "ridge_lambda",
        "anchor_rho",
        "teacher_alpha",
        "covariance_eps",
        "examples",
        "prefill_chunk_size",
    ),
    runtime_parameters=(
        "ridge_lambda",
        "anchor_rho",
        "teacher_alpha",
        "covariance_eps",
        "examples",
        "prefill_chunk_size",
    ),
    normalize=normalize,
    execution="data",
    requires_data=True,
    data_mode="per_expert",
    requires_initial_model=True,
    compute_precision="runtime",
    status="implemented",
)
