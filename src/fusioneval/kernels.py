from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .protocol import Block, SharedContext

def sequential_sum(values):

    result = values[0].clone()
    for value in values[1:]:
        result = result + value
    return result

def coordinate_counts(masks, dtype):

    import torch

    count = torch.zeros_like(masks[0], dtype=dtype)
    for mask in masks.unbind(0):
        count.add_(mask)
    return count

def integer_sum(value, chunk_size=1_048_576):

    import torch

    total = torch.zeros((), dtype=torch.int64, device=value.device)
    for part in value.reshape(-1).split(chunk_size):
        total.add_(part.sum(dtype=torch.int64))
    return int(total)

def overlap_histogram(counts, bins, chunk_size=1_048_576):

    import torch

    histogram = torch.zeros(bins, dtype=torch.int64, device=counts.device)
    for part in counts.reshape(-1).split(chunk_size):
        histogram.add_(torch.bincount(part.long(), minlength=bins))
    return histogram.tolist()

class Kernel:
    name = ""

    def __init__(self, parameters: Sequence[Any], shared: SharedContext):
        self.parameters = tuple(parameters)
        self.shared = shared
        self.stats: Dict[str, Any] = {"tensors": 0, "elements": 0}

    def prepare(self, deltas):
        return None

    def diagnostics(self, state) -> Mapping[str, Any]:
        return dict(self.stats)

    def counted(self, block: Block, value):
        import torch

        if tuple(value.shape) != block.shape:
            raise ValueError(f"kernel returned {tuple(value.shape)} for {block.key}, expected {block.shape}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite output from {self.name} at {block.key}")
        self.stats["tensors"] += 1
        self.stats["elements"] += value.numel()
        return value
