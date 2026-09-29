import torch

from ...kernels import Kernel, sequential_sum

class TaskArithmetic(Kernel):
    name = "task_arithmetic"

    def combine(self, block, state):
        scale, = self.parameters
        delta = sequential_sum(block.deltas)
        return self.counted(block, block.base + scale * delta)

def build(parameters, shared):
    return TaskArithmetic(parameters, shared)
