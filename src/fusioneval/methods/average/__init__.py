from ..base import MethodPlugin
from ..fields import no_parameters
from ...protocol import KernelRequirements

PLUGIN = MethodPlugin(
    name="average", aliases=("avg", "weight_averaging", "uniform_average"), op_code="o01",
    contract_version=2, requirements=KernelRequirements("flat", "none", 1, False),
    parameter_scope="all_floating", aggregation="expert_mean",
    contract_tags=("paper_standard", "official_uniform_scale_then_add", "decoder_full_edge_extension"),
    user_parameters=(), runtime_parameters=(),
    normalize=no_parameters, input_mode="experts",
)
