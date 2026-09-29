import torch

from ...kernels import Kernel, sequential_sum
from ...method_policies import is_edge, leftover_label, leftover_value

def _dare_settings(parameters):
    drop_rate, scale = parameters[0], parameters[1]
    exclude_edge = parameters[2] if len(parameters) > 2 else False
    leftover_edge = parameters[3] if len(parameters) > 3 else "base"
    return drop_rate, scale, exclude_edge, leftover_edge

class DARE(Kernel):
    name = "dare"

    def __init__(self, parameters, shared):
        super().__init__(parameters, shared)
        self.kept = self.total = 0
        self.generator = torch.Generator(device="cpu").manual_seed(shared.seed)

    def prepare(self, source):
        drop_rate, _, exclude_edge, _ = _dare_settings(self.parameters)
        first = next(
            (block for block in source.scan() if not (exclude_edge and is_edge(block))),
            None,
        )
        if first is None:
            return {}
        states = {}

        for expert in range(len(first.deltas)):
            for block in source.scan():
                if exclude_edge and is_edge(block):
                    continue
                states.setdefault(block.key, [None] * len(first.deltas))[expert] = (
                    self.generator.get_state().clone()
                )
                keep = torch.bernoulli(
                    torch.full((block.base.numel(),), 1.0 - drop_rate),
                    generator=self.generator,
                )
                self.kept += int(keep.sum())
                self.total += keep.numel()
        return {key: tuple(value) for key, value in states.items()}

    def combine(self, block, state):
        drop_rate, scale, exclude_edge, leftover_edge = _dare_settings(self.parameters)
        if exclude_edge and is_edge(block):
            value = leftover_value(block, leftover_edge)
            label = leftover_label(leftover_edge, "edge")
            self.stats[label] = self.stats.get(label, 0) + value.numel()
            return self.counted(block, value)
        sparse = []
        for delta, rng_state in zip(block.deltas, state[block.key]):
            generator = torch.Generator(device="cpu")
            generator.set_state(rng_state)
            keep = torch.bernoulli(
                torch.full((delta.numel(),), 1.0 - drop_rate), generator=generator
            ).reshape(delta.shape)
            mask = keep.to(delta.device, dtype=torch.float32)
            sparse.append(delta * mask / (1.0 - drop_rate))
        output = block.base + scale * sequential_sum(sparse)
        return self.counted(block, output)

    def diagnostics(self, state):
        value = super().diagnostics(state)
        value.update(actual_drop_rate=1.0 - self.kept / max(self.total, 1),
                     rng="torch-bernoulli-fp32-expert-major-replay-v2")
        return value

def build(parameters, shared):
    return DARE(parameters, shared)
