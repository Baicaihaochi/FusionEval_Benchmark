from __future__ import annotations

import torch

TA_EDGE_SCALE = 0.3

def is_edge(block) -> bool:
    return any(role in {"input_embedding", "output_head"} for role in block.roles)

def leftover_value(block, policy: str, scale: float = TA_EDGE_SCALE):
    base = block.base.float()
    if policy == "base" or not block.deltas:
        return base
    stacked = torch.stack([item.float() for item in block.deltas])
    if policy == "ta":
        return base + float(scale) * stacked.sum(dim=0)
    return base + stacked.mean(dim=0)

def leftover_label(policy: str, kind: str) -> str:
    if policy == "ta":
        return "ta_edge" if kind == "edge" else "ta_1d"
    if policy == "mean":
        return "mean_edge" if kind == "edge" else "mean_1d"
    return "base_kept_edge" if kind == "edge" else "base_kept_1d"
