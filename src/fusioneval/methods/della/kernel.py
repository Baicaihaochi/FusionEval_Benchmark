import torch

from ...kernels import Kernel, coordinate_counts, integer_sum

def survival_probabilities(delta, drop_rate, window):
    rows = delta.reshape((-1, delta.shape[-1] if delta.ndim > 1 else delta.numel()))
    width = rows.shape[1]
    if width == 1:
        rank = torch.full_like(rows, 0.5, dtype=torch.float32)
    else:
        order = torch.argsort(rows.abs().float(), dim=1)
        rank = torch.empty_like(rows, dtype=torch.float32)
        values = torch.arange(width, dtype=torch.float32, device=rows.device).expand_as(rows)
        rank.scatter_(1, order, values)
        rank /= width - 1
    low = 1.0 - drop_rate - window / 2.0
    return (low + window * rank).reshape(delta.shape)

class DELLA(Kernel):
    name = "della"

    def __init__(self, parameters, shared):
        super().__init__(parameters, shared)
        self.kept = self.total = 0
        self.generator = torch.Generator(device=shared.device).manual_seed(shared.seed)

    def combine(self, block, state):
        drop_rate, window, scale = self.parameters
        sparse = []
        for delta in block.deltas:
            value = delta.float()
            probability = survival_probabilities(value, drop_rate, window)
            keep = torch.bernoulli(probability, generator=self.generator).bool()
            self.kept += integer_sum(keep); self.total += keep.numel()
            sparse.append(value * keep / probability)
        values = torch.stack(sparse, dim=0)
        del sparse, probability, keep, value
        elected = values.sum(dim=0) >= 0
        selected = torch.where(elected.unsqueeze(0), values > 0, values < 0)
        del elected
        merged = torch.where(selected, values, 0).sum(dim=0)
        count = coordinate_counts(selected, torch.float32)
        del selected, values
        merged.div_(count.clamp_(min=1))
        return self.counted(block, block.base.float() + scale * merged)

    def diagnostics(self, state):
        drop_rate, window, _ = self.parameters
        value = super().diagnostics(state)
        value.update(actual_survival_rate=self.kept / max(self.total, 1),
                     survival_probability_bounds=[
                         1.0 - drop_rate - window / 2.0,
                         1.0 - drop_rate + window / 2.0,
                     ],
                     width_one_rank_rule="probability_window_midpoint",
                     rng="torch-bernoulli-key-order-v1")
        return value

def build(parameters, shared):
    return DELLA(parameters, shared)
