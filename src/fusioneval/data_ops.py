from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .config import RunConfig
from .engine import open_model, output_dtype, read_map, save_tensors, tensor
from .errors import CheckpointError


def precision(config: RunConfig) -> str:
    return "float32" if config.method.compute_precision == "float32" else config.runtime["precision"]

def source_dtype(config: RunConfig, value: Any):
    import torch

    return output_dtype(torch, value, config.runtime["save_dtype"])

def is_edge_key(key: str) -> bool:
    return "embed" in key.lower() or key == "lm_head.weight"

def apply_leftover(config: RunConfig, outputs: Dict[str, Any]) -> Dict[str, Any]:
    leftover_edge = config.method_parameters.get("leftover_edge", "mean")
    leftover_1d = config.method_parameters.get("leftover_1d", "mean")
    if leftover_edge == "mean" and leftover_1d == "mean":
        return outputs
    base_map = read_map(config.base)
    with ExitStack() as stack:
        handle = open_model(stack, config.base, base_map)
        for key, value in list(outputs.items()):
            if not getattr(value, "is_floating_point", lambda: False)():
                continue
            if is_edge_key(key):
                if leftover_edge == "base":
                    outputs[key] = tensor(handle, base_map, key)
            elif leftover_1d == "base" and getattr(value, "ndim", 2) != 2:
                outputs[key] = tensor(handle, base_map, key)
    return outputs

def average_experts(config: RunConfig) -> Dict[str, Any]:
    import torch

    roots = [item.path for item in config.experts]
    maps = [read_map(root) for root in roots]
    outputs = {}
    compute = torch.float32 if precision(config) == "float32" else torch.bfloat16
    with ExitStack() as stack:
        opened = [open_model(stack, root, mapping) for root, mapping in zip(roots, maps)]
        for key in sorted(maps[0]):
            values = [tensor(handle, mapping, key) for handle, mapping in zip(opened, maps)]
            first = values[0]
            if first.is_floating_point():
                merged = first.to(compute).clone()
                for value in values[1:]:
                    merged.add_(value.to(compute))
                merged.div_(len(values))
                outputs[key] = merged.to(source_dtype(config, first)).cpu()
            else:
                if any(not torch.equal(first, value) for value in values[1:]):
                    raise ValueError("non-floating expert values differ at {}".format(key))
                outputs[key] = first
    return outputs

def checkpoint_tensors(config: RunConfig, root: Path) -> Dict[str, Any]:

    mapping = read_map(root)
    outputs = {}
    with ExitStack() as stack:
        opened = open_model(stack, root, mapping)
        for key in sorted(mapping):
            value = tensor(opened, mapping, key)
            outputs[key] = (
                value.to(source_dtype(config, value)).cpu()
                if value.is_floating_point()
                else value.cpu()
            )
    return outputs


def write_stats(values: Mapping[str, Any], root: Path, max_gib: float = 5.0) -> None:
    root.mkdir(parents=True, exist_ok=False)
    save_tensors(dict(values), root, int(max_gib * 2**30))

def stat_readers(stack: ExitStack, roots: Sequence[Path]):
    maps = [read_map(root) for root in roots]
    opened = [open_model(stack, root, mapping) for root, mapping in zip(roots, maps)]
    return maps, opened
