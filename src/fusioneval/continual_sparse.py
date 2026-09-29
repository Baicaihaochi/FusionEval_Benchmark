from __future__ import annotations

from typing import Iterable

from .protocol import Block
from .source import streaming_thresholds

def delta_block(block: Block) -> Block:

    if not block.experts:
        raise ValueError(
            "continual two-input kernels require input_mode=experts (raw previous and "
            "incoming tensors) at {}".format(block.key)
        )
    return Block(
        block.key,
        block.shape,
        block.flat_start,
        block.flat_stop,
        block.row_start,
        block.row_stop,
        block.roles,
        block.base,
        tuple(value - block.base for value in block.experts),
        (),
    )

class TwoInputDeltaView:

    def __init__(self, source):
        self.source = source

    def scan(self) -> Iterable[Block]:
        return (delta_block(block) for block in self.source.scan())

    def global_thresholds(self, density, rule):
        return streaming_thresholds(self, density, rule)
