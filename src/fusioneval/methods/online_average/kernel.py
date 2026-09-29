from ...kernels import Kernel

class OnlineAverage(Kernel):

    name = "online_average"

    def combine(self, block, state):
        seen_count, = self.parameters
        if len(block.deltas) != 1:
            raise ValueError("online_average requires one incoming expert delta")
        return self.counted(block, block.base + block.deltas[0] / seen_count)

def build(parameters, shared):
    return OnlineAverage(parameters, shared)
