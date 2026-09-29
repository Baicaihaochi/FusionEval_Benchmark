import torch

from ...kernels import Kernel, coordinate_counts, integer_sum, overlap_histogram
from ...source import FilteredSource

class LocalizeStitch(Kernel):
    name = "localize_stitch"

    def __init__(self, parameters, shared):
        super().__init__(parameters, shared)
        self.selected = None
        self.total_per_expert = 0
        self.overlap_histogram = None

    def prepare(self, deltas):
        density, comparison = self.parameters[:2]
        if len(self.parameters) > 2 and self.parameters[2] == "transformer_blocks":
            deltas = FilteredSource(deltas, lambda block: block.key.startswith("model.layers."))
        thresholds = tuple(deltas.global_thresholds(density, "topk"))
        self.stats.update(target_density=density, thresholds=list(thresholds), comparison=comparison)
        return thresholds

    def combine(self, block, state):
        _, comparison = self.parameters[:2]
        if len(self.parameters) > 2 and self.parameters[2] == "transformer_blocks" and not block.key.startswith("model.layers."):
            return self.counted(block, block.base)
        values = torch.stack(block.deltas)
        flat = values.reshape(len(values), -1)
        threshold = torch.as_tensor(state, dtype=values.dtype, device=values.device).unsqueeze(1)
        masks = flat.abs() > threshold if comparison == "strict" else flat.abs() >= threshold
        raw_count = coordinate_counts(masks, torch.int32)
        if self.selected is None:
            self.selected = [0] * len(values)
            self.overlap_histogram = [0] * (len(values) + 1)
        for index in range(len(values)):
            self.selected[index] += integer_sum(masks[index])
        self.total_per_expert += flat.shape[1]
        histogram = overlap_histogram(raw_count, len(values) + 1)
        for overlap, amount in enumerate(histogram):
            self.overlap_histogram[overlap] += int(amount)
        count = raw_count.clamp_(min=1)
        merged = torch.where(masks, flat, 0).sum(dim=0).div(count).reshape(block.shape)
        return self.counted(block, block.base + merged)

    def diagnostics(self, state):
        value = super().diagnostics(state)
        value.update(
            expert_selected=self.selected or [],
            expert_actual_density=[
                count / max(self.total_per_expert, 1) for count in (self.selected or [])
            ],
            overlap_histogram=self.overlap_histogram or [],
        )
        return value

def build(parameters, shared):
    return LocalizeStitch(parameters, shared)
