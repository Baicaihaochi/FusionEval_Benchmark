from ..base import MethodPlugin
from ..fields import choice, optional_mapping, optional_number, optional_positive_int
from ...protocol import KernelRequirements

def normalize(raw):
    raw = optional_mapping(raw, ("rank", "steps", "learning_rate", "examples", "prefill_chunk_size", "feature_cache"))
    return {
        "rank": optional_positive_int(raw, "rank", 16),
        "steps": optional_positive_int(raw, "steps", 500),
        "learning_rate": optional_number(
            raw, "learning_rate", 1e-3, minimum=0.0, closed_min=False
        ),
        "examples": optional_positive_int(raw, "examples", 256),
        "prefill_chunk_size": optional_positive_int(raw, "prefill_chunk_size", 256),
        "feature_cache": choice(raw, "feature_cache", "none", ("none", "cpu")),
    }

PLUGIN = MethodPlugin(
    name="surgery",
    aliases=("representation_surgery",),
    op_code="d08",
    contract_version=2,
    requirements=KernelRequirements("full_rows", "none", 1, False, True),
    parameter_scope="task_routed_final_hidden_state_adapters",
    aggregation="low_rank_representation_bias_subtraction",
    contract_tags=(
        "paper_training_objective",
        "official_vision_code_reference",
        "decoder_final_hidden_extension",
        "requires_explicit_domain_router",
        "initial_checkpoint_weights_unchanged",
    ),
    user_parameters=("rank", "steps", "learning_rate", "examples", "prefill_chunk_size", "feature_cache"),
    runtime_parameters=("rank", "steps", "learning_rate", "examples", "prefill_chunk_size", "feature_cache"),
    normalize=normalize,
    execution="data",
    requires_data=True,
    data_mode="per_expert",
    requires_initial_model=True,
    status="implemented",
)
