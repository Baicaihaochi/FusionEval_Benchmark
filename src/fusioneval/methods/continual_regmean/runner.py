from __future__ import annotations

import gc
from contextlib import ExitStack
from pathlib import Path

from ...config import RunConfig
from ...data_engine import DataResult
from ...data_ops import (
    checkpoint_tensors,
    is_edge_key,
    precision,
    stat_readers,
    write_stats,
)
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

def recursive_regmean_weight(
    previous_weight,
    incoming_weight,
    previous_gram,
    incoming_gram,
    alpha: float,
):

    import torch

    previous = shrink(previous_gram.to(torch.float32), alpha)
    incoming = shrink(incoming_gram.to(torch.float32), alpha)
    matrix = previous.add(incoming)
    rhs = previous @ previous_weight.to(torch.float32).transpose(0, 1)
    rhs = rhs.add(incoming @ incoming_weight.to(torch.float32).transpose(0, 1))
    return exact_solve(matrix, rhs).transpose(0, 1).cpu()

def leftover_update(
    previous, incoming, key: str, experts_included: int, leftover_edge: str, leftover_1d: str
):

    import torch

    if not previous.is_floating_point():
        if not torch.equal(previous, incoming):
            raise CheckpointError("non-floating expert values differ at {}".format(key))
        return previous
    if is_edge_key(key) and leftover_edge == "base":
        return previous
    if not is_edge_key(key) and previous.ndim != 2 and leftover_1d == "base":
        return previous
    return previous.float().add(
        incoming.float() - previous.float(), alpha=1.0 / experts_included
    )

def _accumulate(previous_row, incoming_row, experts_included: int):
    if experts_included == 1 or previous_row is None:
        return incoming_row.float().cpu()
    return previous_row.float().add(incoming_row.float()).cpu()

def _initial_stage(config: RunConfig, incoming_root: Path, incoming_stats: Path):

    outputs = checkpoint_tensors(config, incoming_root)
    cumulative = {}
    with ExitStack() as stack:
        maps, handles = stat_readers(stack, [incoming_stats])
        for key in sorted(maps[0]):
            cumulative[key] = tensor(handles[0], maps[0], key).float().cpu()
    return outputs, cumulative

def execute(config: RunConfig, workspace: Path) -> DataResult:
    import torch

    parameters = config.method_parameters
    experts_included = parameters["experts_included"]
    alpha = float(parameters["alpha"])
    examples = parameters["examples"]
    leftover_edge = parameters.get("leftover_edge", "mean")
    leftover_1d = parameters.get("leftover_1d", "mean")
    device = config.runtime["device"]

    incoming_root = config.experts[0].path
    tokenizer = load_tokenizer(config.base)
    model = load_model(incoming_root, device, precision(config))
    grams, summary, module_map = collect_grams(
        model, tokenizer, config.experts[0].data, device, examples
    )
    incoming_stats = workspace / "incoming"
    write_stats(grams, incoming_stats, config.runtime["max_shard_size_gib"])
    del model, grams
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if experts_included == 1:
        outputs, cumulative = _initial_stage(config, incoming_root, incoming_stats)
        state = workspace / "state"
        write_stats(cumulative, state, config.runtime["max_shard_size_gib"])
        return DataResult(
            outputs,
            {
                "alpha": alpha,
                "experts_included": 1,
                "observation_point": 0,
                "estimator": "per-token-mean input activation gram of decoder linears",
                "initial_stage": "unmodified initial expert weights; raw Gram accumulated",
                "weights_modified": False,
                "solve_performed": False,
                "state_shrunk_on_disk": False,
                "incoming": summary,
                "incoming_domain": config.experts[0].id,
                "examples_per_domain": examples,
                "covered_linear_weights": 0,
                "state_gram_keys": len(cumulative),
                "cumulative_state": "state",
                "peak_full_models_on_device": 1,
                "historical_expert_models_loaded": 0,
            },
            {"state": state},
        )

    previous_weights = read_map(config.base)
    incoming_weights = read_map(incoming_root)
    if set(previous_weights) != set(incoming_weights):
        raise CheckpointError("previous and incoming checkpoints are not tensor-aligned")
    outputs = checkpoint_tensors(config, config.base)

    previous_state = config.base / "state"
    if not previous_state.is_dir():
        raise CheckpointError("previous continual RegMean stage has no cumulative state")

    cumulative = {}
    covered = set()
    with ExitStack() as stack:
        previous_model = open_model(stack, config.base, previous_weights)
        incoming_model = open_model(stack, incoming_root, incoming_weights)
        incoming_stat_maps, incoming_stat_handles = stat_readers(stack, [incoming_stats])
        incoming_gram_map = incoming_stat_maps[0]
        previous_stat_maps, previous_stat_handles = stat_readers(stack, [previous_state])
        if set(previous_stat_maps[0]) != set(incoming_gram_map):
            raise CheckpointError(
                "previous and incoming RegMean state schemas differ"
            )
        previous_gram_map = previous_stat_maps[0]
        previous_gram_handle = previous_stat_handles[0]
        incoming_gram_handle = incoming_stat_handles[0]

        for gram_key in sorted(incoming_gram_map):
            incoming_row = tensor(incoming_gram_handle, incoming_gram_map, gram_key)
            previous_row = tensor(previous_gram_handle, previous_gram_map, gram_key)
            cumulative[gram_key] = _accumulate(
                previous_row, incoming_row, experts_included
            )

        for module_name, gram_key in sorted(module_map.items()):
            weight_key = module_name + ".weight"
            if (
                weight_key not in outputs
                or weight_key not in previous_weights
                or weight_key not in incoming_weights
                or gram_key not in incoming_gram_map
            ):
                continue
            previous_weight = tensor(previous_model, previous_weights, weight_key)
            incoming_weight = tensor(incoming_model, incoming_weights, weight_key)
            if previous_weight.shape != incoming_weight.shape:
                raise CheckpointError(
                    "model tensors are misaligned at {}".format(weight_key)
                )
            incoming_gram = tensor(incoming_gram_handle, incoming_gram_map, gram_key)
            if incoming_gram.ndim != 2 or incoming_gram.shape[0] != incoming_gram.shape[1]:
                raise CheckpointError(
                    "RegMean gram is not square at {}".format(gram_key)
                )
            if incoming_gram.shape[0] != incoming_weight.shape[1]:
                raise CheckpointError(
                    "RegMean gram does not match {} input width".format(weight_key)
                )
            previous_gram = tensor(previous_gram_handle, previous_gram_map, gram_key)
            merged = recursive_regmean_weight(
                previous_weight, incoming_weight, previous_gram, incoming_gram, alpha
            )
            outputs[weight_key] = merged.to(outputs[weight_key].dtype).cpu()
            covered.add(weight_key)

        for key in sorted(outputs):
            if key in covered:
                continue
            if key not in previous_weights or key not in incoming_weights:
                raise CheckpointError("model tensors are not aligned at {}".format(key))
            value = leftover_update(
                tensor(previous_model, previous_weights, key),
                tensor(incoming_model, incoming_weights, key),
                key,
                experts_included,
                leftover_edge,
                leftover_1d,
            )
            if outputs[key].is_floating_point():
                value = value.to(outputs[key].dtype)
            outputs[key] = value.cpu()

    state = workspace / "state"
    write_stats(cumulative, state, config.runtime["max_shard_size_gib"])
    return DataResult(
        outputs,
        {
            "alpha": alpha,
            "experts_included": experts_included,
            "observation_point": experts_included - 1,
            "estimator": "per-token-mean input activation gram of decoder linears",
            "gram_dtype": "model_forward_dtype",
            "solver_dtype": "float32",
            "accumulator": "raw_summed_per_domain_grams",
            "shrunk_at_solve": True,
            "state_shrunk_on_disk": False,
            "shrinkage": "alpha*G+(1-alpha)*diag(G)",
            "solver": "torch.linalg.solve",
            "spectrum_truncation": "none",
            "pseudoinverse": 0,
            "incoming": summary,
            "incoming_domain": config.experts[0].id,
            "examples_per_domain": examples,
            "covered_linear_weights": len(covered),
            "state_gram_keys": len(cumulative),
            "cumulative_state": "state",
            "previous_cumulative_state": "state",
            "leftover_edge": leftover_edge,
            "leftover_1d": leftover_1d,
            "nonlinear_and_edge_policy": (
                "leftover_edge={};leftover_1d={};"
                "mean=online expert mean with weight 1/experts_included".format(
                    leftover_edge, leftover_1d
                )
            ),
            "peak_full_models_on_device": 1,
            "historical_expert_models_loaded": 0,
        },
        {"state": state},
    )
