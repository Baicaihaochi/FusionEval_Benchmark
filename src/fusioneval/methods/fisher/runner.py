from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import math

from ...config import RunConfig
from ...data_engine import DataResult
from ...data_ops import checkpoint_tensors, stat_readers
from .estimate import estimate
from ...engine import open_model, read_map, tensor

def fisher_weighted_mean(weights, fishers, epsilon):

    numerator = denominator = None
    for weight, fisher in zip(weights, fishers):
        importance = fisher.float().add(epsilon)
        weighted = importance * weight.float()
        denominator = importance.clone() if denominator is None else denominator.add_(importance)
        numerator = weighted.clone() if numerator is None else numerator.add_(weighted)
    return numerator / denominator

def weighted_mean(weights, fishers, coefficients, norms, floor, favor_target):
    import torch
    numerator = denominator = None
    for index, (weight, fisher, coefficient, norm) in enumerate(zip(weights, fishers, coefficients, norms)):
        importance = fisher.float()
        if not favor_target or index == 0:
            importance = importance.clamp_min(floor)
        importance = importance * (coefficient / norm)
        term = importance * weight.float()
        numerator = term if numerator is None else numerator + term
        denominator = importance if denominator is None else denominator + importance
    if not torch.isfinite(denominator).all() or (denominator <= 0).any():
        raise ValueError("undefined Fisher merge denominator; adjust coefficients/floor policy")
    return numerator / denominator

def _merge(config: RunConfig, stat_roots):
    import torch

    outputs = checkpoint_tensors(config, config.experts[0].path)
    expert_roots = [item.path for item in config.experts]
    expert_maps = [read_map(root) for root in expert_roots]
    options = config.method_parameters
    coefficients = options["coefficients"] or [1.0 / len(expert_roots)] * len(expert_roots)
    if len(coefficients) != len(expert_roots):
        raise ValueError("one coefficient required per expert, in expert order")
    with ExitStack() as stack:
        experts = [
            open_model(stack, root, mapping)
            for root, mapping in zip(expert_roots, expert_maps)
        ]
        stat_maps, stats = stat_readers(stack, stat_roots)
        common = set.intersection(*(set(mapping) for mapping in stat_maps))
        if any(set(mapping) != common for mapping in stat_maps):
            raise ValueError("Fisher parameter coverage differs across experts")
        import json
        tied = json.loads((config.base / "config.json").read_text())["tie_word_embeddings"]
        expected = {key for key, value in outputs.items() if value.is_floating_point()
                    and (tied or key != "lm_head.weight")}

        if tied:
            expected.discard("lm_head.weight")
        if common != expected:
            raise ValueError("Fisher statistics do not cover the complete mergeable body")
        norms = []
        for handle, mapping in zip(stats, stat_maps):
            total = sum(float(tensor(handle, mapping, key).double().square().sum()) for key in sorted(common))
            norm = total ** 0.5 if options["normalize_fishers"] else 1.0
            if not norm > 0 or not math.isfinite(norm):
                raise ValueError("zero/nonfinite global Fisher norm")
            norms.append(norm)
        for key in sorted(common):
            weights, fishers = [], []
            for weight_handle, weight_map, stat_handle, stat_map in zip(
                experts, expert_maps, stats, stat_maps
            ):
                fishers.append(tensor(stat_handle, stat_map, key))
                weights.append(tensor(weight_handle, weight_map, key))
            outputs[key] = weighted_mean(weights, fishers, coefficients, norms,
                options["fisher_floor"], options["favor_target_model"]).to(
                outputs[key].dtype
            )
        if tied and "lm_head.weight" in outputs:
            outputs["lm_head.weight"] = outputs["model.embed_tokens.weight"]
    return outputs, {"global_fisher_norms": norms, "coefficients": coefficients,
                     "mergeable_tensors": len(common), "untied_head": "first expert retained"}

def execute(config: RunConfig, workspace: Path) -> DataResult:
    stat_roots = []
    domains = {}
    for expert in config.experts:
        root = workspace / expert.id
        domains[expert.id] = estimate(config, expert, root)
        stat_roots.append(root)
    outputs, merging = _merge(config, stat_roots)
    return DataResult(
        outputs,
        {
            "estimator": config.method_parameters["estimator"],
            "accumulator_dtype": "float32",
            "options": config.method_parameters,
            "merging": merging,
            "domains": domains,
        },
    )
