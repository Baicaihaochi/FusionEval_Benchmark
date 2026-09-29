import torch

from ...kernels import Kernel

class Average(Kernel):
    name = "average"

    def combine(self, block, state):
        experts = tuple(block.experts or (block.base + item for item in block.deltas))

        coefficient = 1.0 / len(experts)
        total = experts[0] * coefficient
        for expert in experts[1:]:
            total = expert * coefficient + total
        self.stats.update(soup_variant="uniform", accumulation="scale_then_add",
                          ingredients=len(experts), coefficient=coefficient)
        return self.counted(block, total)

def build(parameters, shared):
    return Average(parameters, shared)
