from ..base import MethodPlugin
from ..fields import number, parameters, scale
from ...errors import ConfigError
from ...protocol import KernelRequirements

def normalize(raw):
    value = parameters(raw, ("drop_rate", "window", "scale"))
    drop_rate, window = number(value["drop_rate"], "drop_rate"), number(value["window"], "window")
    low, high = 1.0 - drop_rate - window / 2.0, 1.0 - drop_rate + window / 2.0
    if window < 0.0 or not (0.0 < low <= high <= 1.0):
        raise ConfigError("DELLA survival-probability range is invalid")
    return {"drop_rate": drop_rate, "window": window, "scale": scale(value["scale"])}

PLUGIN = MethodPlugin(
    name="della", aliases=(), op_code="o05", contract_version=1,
    requirements=KernelRequirements("full_rows", "none", 1, False, stochastic=True),
    parameter_scope="all_floating", aggregation="disjoint_mean",
    contract_tags=("official_code_compat", "decoder_full_edge_extension"),
    user_parameters=("drop_rate", "window", "scale"),
    runtime_parameters=("drop_rate", "window", "scale"), normalize=normalize,
    compute_precision="float32",
)
