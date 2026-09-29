from ..base import MethodPlugin
from ..fields import parameters, scale
from ...protocol import KernelRequirements

def normalize(raw):
    value = parameters(raw, ("scale",))
    return {"scale": scale(value["scale"])}

PLUGIN = MethodPlugin(
    name="task_arithmetic", aliases=("ta",), op_code="o02", contract_version=1,
    requirements=KernelRequirements("flat", "none", 1, False),
    parameter_scope="all_floating", aggregation="task_vector_sum",
    contract_tags=("paper_standard", "decoder_full_edge_extension"),
    user_parameters=("scale",),
    runtime_parameters=("scale",), normalize=normalize,
)
