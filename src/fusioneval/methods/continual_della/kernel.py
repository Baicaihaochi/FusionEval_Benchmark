from ...continual_rng import TRAVERSAL, encode_rng_state, restore_generator_state
from ...continual_sparse import TwoInputDeltaView, delta_block
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
        self.input_state = shared.rng_state
        self.seeded_fresh = shared.rng_state is None
        if not self.seeded_fresh:
            restore_generator_state(self.inner.generator, shared.rng_state)
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
        value["rng"] = encode_rng_state(
            self.inner.generator,
            method=self.name,
            device=str(self.inner.generator.device),
            initial_seed=self.initial_seed,
            seeded_fresh=self.seeded_fresh,
            input_state=self.input_state,
            traversal=TRAVERSAL[self.name],
        )
        return value

def build(parameters, shared):
    return ContinualDELLA(parameters, shared)
