from ...continual_sparse import TwoInputDeltaView, delta_block
from ...kernels import Kernel
from .._continual_sparse import TIES_PARAMETERS, encode_by_name
from ..ties.kernel import build as ties_build

class ContinualTIES(Kernel):
    name = "continual_ties"

    def __init__(self, parameters, shared):
        super().__init__(parameters, shared)
        value = encode_by_name(parameters, TIES_PARAMETERS)
        self.inner = ties_build(
            (
                value["density"],
                value["scale"],
                value["merge_func"],
                value["exclude_edge"],
                "base",
            ),
            shared,
        )
        self.experts_included = int(value["experts_included"])
        self.participant_count = int(value["participant_count"])

    def prepare(self, source):

        return self.inner.prepare(TwoInputDeltaView(source))

    def combine(self, block, state):
        return self.counted(block, self.inner.combine(delta_block(block), state))

    def diagnostics(self, state):
        value = dict(self.inner.diagnostics(state))
        value.update(
            continual_rule="M_t = B + scale * TIES_density(h, d)",
            delta_definition="h = previous_merged - B, d = incoming_expert - B",
            accumulated_vector_votes=1,
            participant_count=self.participant_count,
            participant_roles=["previous_accumulated", "incoming"],
            experts_included=self.experts_included,
        )
        return value

def build(parameters, shared):
    return ContinualTIES(parameters, shared)
