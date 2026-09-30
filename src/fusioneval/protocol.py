from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence, Tuple

@dataclass(frozen=True)
class KernelRequirements:

    layout: Literal["flat", "full_rows"]
    prepare_scope: Literal["none", "global", "tensor"]
    passes: int
    replayable: bool
    stochastic: bool = False

@dataclass(frozen=True)
class Block:
    key: str
    shape: Tuple[int, ...]
    flat_start: int
    flat_stop: int
    row_start: int
    row_stop: int
    roles: Tuple[str, ...]
    base: Any
    deltas: Sequence[Any]
    experts: Sequence[Any] = ()

@dataclass(frozen=True)
class SharedContext:

    seed: int
    precision: Literal["bfloat16", "float32"] = "bfloat16"
    device: Literal["cpu", "cuda"] = "cpu"
    decoder_layers: int | None = None
    decoder_layer_prefix: str = "model.layers"
    attention_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_modules: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

    rng_state: str | None = None

class DeltaSource(Protocol):

    def scan(self) -> Iterable[Block]:
        ...

    def global_thresholds(
        self, density: float, rule: Literal["ties", "topk"]
    ) -> Sequence[float]:

        ...

class MethodKernel(Protocol):

    name: str

    def prepare(self, deltas: DeltaSource) -> Any:

        ...

    def combine(self, block: Block, state: Any) -> Any:

        ...

    def diagnostics(self, state: Any) -> Mapping[str, Any]:

        ...

class KernelFactory(Protocol):

    def build(
        self, parameters: Sequence[Any], shared: SharedContext
    ) -> MethodKernel:
        ...
