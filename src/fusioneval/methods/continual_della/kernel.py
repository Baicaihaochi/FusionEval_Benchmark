from ...continual_sparse import delta_block
from ...kernels import Kernel
from .._continual_sparse import DELLA_PARAMETERS, encode_by_name
from ..della.kernel import build as della_build

class ContinualDELLA(Kernel):
    name = "continual_della"

    def __init__(self, parameters, shared):
        super().__init__(parameters, shared)
        value = encode_by_name(parameters, DELLA_PARAMETERS)
        self.inner = della_build(
            (value["drop_rate"], value["window"], value["scale"]), shared
        )
        self.initial_seed = int(value["seed"])
        if self.initial_seed != int(shared.seed):
            raise ValueError(
                "continual della seed {} disagrees with runtime.seed {}".format(
                    self.initial_seed, shared.seed
                )
            )
        if shared.rng_state is not None:
            raise ValueError("continual DELLA uses a stage seed; historical RNG state is unsupported")
        self.experts_included = int(value["experts_included"])
        self.participant_count = int(value["participant_count"])

    def prepare(self, source):

        return None

    def combine(self, block, state):
        return self.counted(block, self.inner.combine(delta_block(block), state))

    def diagnostics(self, state):
        value = dict(self.inner.diagnostics(state))
        value.update(
            continual_rule="M_t = B + scale * DELLA_drop_window(h, d)",
            delta_definition="h = previous_merged - B, d = incoming_expert - B",
            both_inputs_resparsified=True,
            participant_count=self.participant_count,
            participant_roles=["previous_accumulated", "incoming"],
            experts_included=self.experts_included,
        )
        value["rng"] = {"policy": "independent_stage_seed", "seed": self.initial_seed,
                        "historical_state_loaded": False}
        return value

def build(parameters, shared):
    return ContinualDELLA(parameters, shared)
