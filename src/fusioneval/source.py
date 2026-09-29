from __future__ import annotations

import struct
from typing import Callable, Iterable, Sequence

from .protocol import Block

def _target_rank(total: int, density: float, rule: str) -> int:
    if total <= 0:
        raise ValueError("cannot select a threshold from an empty source")
    if rule == "ties":
        keep = int(total * density)
        return 0 if keep == total else total - keep - 1
    if rule == "topk":
        keep = max(1, int(total * density))
        return total - keep
    raise ValueError("unsupported global threshold rule: {}".format(rule))

def streaming_thresholds(source, density: float, rule: str) -> Sequence[float]:

    import torch

    first = next(iter(source.scan()), None)
    if first is None:
        raise ValueError("cannot select a threshold from an empty source")
    expert_count = len(first.deltas)
    if expert_count == 0:
        raise ValueError("threshold selection requires task-vector deltas")
    totals = [0] * expert_count
    for block in source.scan():
        if len(block.deltas) != expert_count:
            raise ValueError("expert count changed while replaying the tensor source")
        for index, value in enumerate(block.deltas):
            totals[index] += value.numel()
    ranks = [_target_rank(total, density, rule) for total in totals]
    prefixes = [0] * expert_count

    for shift in (24, 16, 8, 0):
        histograms = [torch.zeros(256, dtype=torch.int64) for _ in range(expert_count)]
        for block in source.scan():
            for index, value in enumerate(block.deltas):
                magnitudes = value.detach().abs().float().reshape(-1)
                if not bool(torch.isfinite(magnitudes).all()):
                    raise FloatingPointError("non-finite task vector at {}".format(block.key))
                bits = magnitudes.contiguous().view(torch.int32).to(torch.int64)
                bits.bitwise_and_(0xFFFFFFFF)
                if shift != 24:
                    bits = bits[bits.bitwise_right_shift(shift + 8) == prefixes[index]]
                buckets = bits.bitwise_right_shift(shift).bitwise_and(0xFF)
                histograms[index] += torch.bincount(buckets, minlength=256).cpu()
        for index, counts in enumerate(histograms):
            cumulative = counts.cumsum(0)
            bucket = int(torch.searchsorted(cumulative, torch.tensor(ranks[index] + 1)))
            before = int(cumulative[bucket - 1]) if bucket else 0
            ranks[index] -= before
            prefixes[index] = (prefixes[index] << 8) | bucket

    return [struct.unpack("!f", struct.pack("!I", bits))[0] for bits in prefixes]

class Source:

    def __init__(self, blocks: Sequence[Block] = ()):
        self.blocks = tuple(blocks)

    def scan(self) -> Iterable[Block]:
        return iter(self.blocks)

    def global_thresholds(self, density, rule):
        return streaming_thresholds(self, density, rule)

class FilteredSource:

    def __init__(self, source, predicate: Callable[[Block], bool]):
        self.source = source
        self.predicate = predicate

    def scan(self) -> Iterable[Block]:
        return (block for block in self.source.scan() if self.predicate(block))

    def global_thresholds(self, density, rule):
        return streaming_thresholds(self, density, rule)

    def is_empty(self) -> bool:
        return next(iter(self.scan()), None) is None
