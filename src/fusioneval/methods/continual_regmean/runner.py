from __future__ import annotations

import gc
from contextlib import ExitStack
from pathlib import Path

from ...config import RunConfig
from ...data_engine import DataResult
from ...data_ops import checkpoint_tensors, is_edge_key, precision, stat_readers, write_stats
from ...data_runtime import load_model, load_tokenizer
from ...engine import open_model, read_map, tensor
from ...errors import CheckpointError
from ...regression import collect_grams, shrink


def exact_solve(matrix, rhs):
    import torch
    try:
        return torch.linalg.solve(matrix, rhs)
    except torch.linalg.LinAlgError as exc:
        raise CheckpointError("exact RegMean solve failed: {}".format(exc)) from exc


def pairwise_regmean_weight(previous_weight, incoming_weight, previous_gram, incoming_gram, alpha):
    import torch
    previous = shrink(previous_gram.to(torch.float32), alpha)
    incoming = shrink(incoming_gram.to(torch.float32), alpha)
    matrix = previous + incoming
    rhs = previous @ previous_weight.to(torch.float32).transpose(0, 1)
    rhs = rhs + incoming @ incoming_weight.to(torch.float32).transpose(0, 1)
    return exact_solve(matrix, rhs).transpose(0, 1).cpu()


def leftover_update(previous, incoming, key, leftover_edge, leftover_1d):
    import torch
    if not previous.is_floating_point():
        if not torch.equal(previous, incoming):
            raise CheckpointError("non-floating expert values differ at {}".format(key))
        return previous
    if is_edge_key(key) and leftover_edge == "base":
        return previous
    if not is_edge_key(key) and previous.ndim != 2 and leftover_1d == "base":
        return previous
    return (previous.float() + incoming.float()) * 0.5


def collect_pair_stats(config, workspace):
    """Recompute both participants on incoming data; never read persisted statistics."""
    import torch
    incoming = config.experts[0]
    tokenizer = load_tokenizer(config.base)
    summaries, module_maps = {}, []
    for label, root in (("previous", config.base), ("incoming", incoming.path)):
        model = load_model(root, config.runtime["device"], precision(config))
        try:
            grams, summary, module_map = collect_grams(
                model, tokenizer, incoming.data, config.runtime["device"],
                config.method_parameters["examples"],
            )
            write_stats(grams, workspace / label, config.runtime["max_shard_size_gib"])
            summaries[label] = summary
            module_maps.append(module_map)
            del grams
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if module_maps[0] != module_maps[1]:
        raise CheckpointError("previous and incoming RegMean module schemas differ")
    return summaries, module_maps[0]


def execute(config: RunConfig, workspace: Path) -> DataResult:
    parameters = config.method_parameters
    incoming_root = config.experts[0].path
    summaries, module_map = collect_pair_stats(config, workspace)
    previous_weights, incoming_weights = read_map(config.base), read_map(incoming_root)
    if set(previous_weights) != set(incoming_weights):
        raise CheckpointError("previous and incoming checkpoints are not tensor-aligned")
    outputs = checkpoint_tensors(config, config.base)
    covered = set()
    with ExitStack() as stack:
        previous_model = open_model(stack, config.base, previous_weights)
        incoming_model = open_model(stack, incoming_root, incoming_weights)
        maps, handles = stat_readers(stack, [workspace / "previous", workspace / "incoming"])
        if set(maps[0]) != set(maps[1]):
            raise CheckpointError("previous and incoming RegMean Gram schemas differ")
        for module_name, gram_key in sorted(module_map.items()):
            key = module_name + ".weight"
            if key not in outputs or gram_key not in maps[0]:
                raise CheckpointError("missing RegMean weight or Gram for {}".format(key))
            previous = tensor(previous_model, previous_weights, key)
            incoming = tensor(incoming_model, incoming_weights, key)
            grams = [tensor(handle, mapping, gram_key) for handle, mapping in zip(handles, maps)]
            if previous.shape != incoming.shape or any(
                tuple(gram.shape) != (previous.shape[1], previous.shape[1]) for gram in grams
            ):
                raise CheckpointError("RegMean tensor dimensions differ at {}".format(key))
            merged = pairwise_regmean_weight(previous, incoming, *grams, parameters["alpha"])
            outputs[key] = merged.to(outputs[key].dtype).cpu()
            covered.add(key)
        for key in sorted(outputs):
            if key in covered:
                continue
            value = leftover_update(
                tensor(previous_model, previous_weights, key),
                tensor(incoming_model, incoming_weights, key), key,
                parameters["leftover_edge"], parameters["leftover_1d"],
            )
            outputs[key] = value.to(outputs[key].dtype).cpu()
    return DataResult(outputs, {
        "alpha": parameters["alpha"],
        "experts_included": parameters["experts_included"],
        "estimator": "per-token-mean input activation Gram; recomputed for both models",
        "calibration_policy": "incoming_domain_only_for_both_models",
        "calibration": summaries,
        "examples_per_model": parameters["examples"],
        "covered_linear_weights": len(covered),
        "solver_dtype": "float32",
        "solver": "torch.linalg.solve",
        "shrinkage": "alpha*G+(1-alpha)*diag(G)",
        "leftover_edge": parameters["leftover_edge"],
        "leftover_1d": parameters["leftover_1d"],
        "mean_policy": "equal pairwise mean, independent of historical expert count",
        "historical_expert_models_loaded": 0,
        "historical_statistics_loaded": False,
        "historical_calibration_data_loaded": False,
        "cumulative_statistics_saved": False,
        "peak_full_models_on_device": 1,
    })
