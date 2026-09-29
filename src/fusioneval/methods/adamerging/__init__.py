from ..base import MethodPlugin
from ..fields import choice, optional_mapping
from ...protocol import KernelRequirements

def normalize(raw):
    raw = optional_mapping(raw, ("embed_coefficient",))
    return {
        "steps": 500,
        "learning_rate": 3e-4,
        "initial_coefficient": 0.3,
        "max_prompt_tokens": 4096,
        "embed_coefficient": choice(
            raw, "embed_coefficient", "learned", ("learned", "zero", "mean")
        ),
    }

PLUGIN = MethodPlugin(
    name="adamerging",
    aliases=("ada_merging",),
    op_code="d05",
    contract_version=2,
    requirements=KernelRequirements("flat", "none", 1, False),
    parameter_scope="all_floating_tensorwise_coefficients",
    aggregation="entropy_adapted_task_vectors",
    contract_tags=(
        "paper_layerwise_optimizer",
        "decoder_last_token_entropy_extension",
        "decoder_full_edge_extension",
    ),
    user_parameters=("embed_coefficient",),
    runtime_parameters=(
        "steps",
        "learning_rate",
        "initial_coefficient",
        "max_prompt_tokens",
        "embed_coefficient",
    ),
    normalize=normalize,
    execution="data",
    requires_data=True,
    data_mode="per_expert",
)
