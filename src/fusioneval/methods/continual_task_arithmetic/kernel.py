from ...kernels import Kernel

class ContinualTaskArithmetic(Kernel):

    name = "continual_task_arithmetic"

    def combine(self, block, state):
        scale, experts_included = self.parameters
        if experts_included < 2:
            raise ValueError(
                "continual_task_arithmetic fuses only once at least two experts are "
                "included; Stage 0 is the initial expert and runs no kernel"
            )
        if len(block.experts) != 2:
            raise ValueError(
                "continual_task_arithmetic requires the anchor and incoming experts"
            )

        increment = block.experts[1] - block.experts[0]
        return self.counted(block, block.base + scale * increment)

def build(parameters, shared):
    return ContinualTaskArithmetic(parameters, shared)
