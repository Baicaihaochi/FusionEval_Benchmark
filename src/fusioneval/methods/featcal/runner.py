from __future__ import annotations

import gc
import copy
from contextlib import ExitStack, closing
from functools import partial
from pathlib import Path

from ...config import RunConfig
from ...data_engine import DataResult
from ...data_ops import checkpoint_tensors, precision
from ...data_runtime import iter_features, linear_modules, load_model, load_tokenizer, model_layers
from ...engine import open_model, read_map, tensor
from ...feature_runtime import hidden_chunks
from .math import solve_weight
from .cache import advance, collect

def _group_name(module_name: str) -> str:
    parent, _, leaf = module_name.rpartition(".")
    if leaf in {"q_proj", "k_proj", "v_proj"}:
        return parent + ".__qkv_input__"
    if leaf in {"gate_proj", "up_proj"}:
        return parent + ".__gate_up_input__"
    return module_name

def _groups(layer):
    result = {}
    modules = linear_modules(layer)
    for name in modules:
        result.setdefault(_group_name(name), []).append(name)
    return modules, result

def _collect_pair_stats(
    student,
    expert,
    student_layer,
    expert_layer,
    tokenizer,
    path,
    device,
    examples,
    teacher_alpha,
    prefill_chunk_size=256,
):
    import torch

    student_modules, groups = _groups(student_layer)
    expert_modules, expert_groups = _groups(expert_layer)
    if groups != expert_groups:
        raise RuntimeError("student and expert decoder-layer layouts differ")
    student_cache = {}
    expert_cache = {}

    def capture(cache, group, _module, inputs):
        cache[group] = inputs[0].detach()

    student_handles = [
        student_modules[names[0]].register_forward_pre_hook(
            partial(capture, student_cache, group)
        )
        for group, names in groups.items()
    ]
    expert_handles = [
        expert_modules[names[0]].register_forward_pre_hook(
            partial(capture, expert_cache, group)
        )
        for group, names in groups.items()
    ]
    gram_sums = {}
    cross_sums = {}
    rows = {}
    seen = tokens = 0
    alpha = float(teacher_alpha)
    student_index = next(i for i, layer in enumerate(model_layers(student)) if layer is student_layer)
    expert_index = next(i for i, layer in enumerate(model_layers(expert)) if layer is expert_layer)
    try:
        with torch.inference_mode():
            for sample in iter_features(tokenizer, path, device, examples):
                with closing(hidden_chunks(student, sample, prefill_chunk_size, student_index)) as sc, closing(hidden_chunks(expert, sample, prefill_chunk_size, expert_index)) as ec:
                    for _ in range(0, sample.tokens, prefill_chunk_size):
                        student_cache.clear()
                        expert_cache.clear()
                        next(sc)
                        next(ec)
                        if set(student_cache) != set(groups) or set(expert_cache) != set(groups):
                            raise RuntimeError("FeatCal did not capture every decoder linear input")
                        for group in groups:
                            xs = student_cache[group].reshape(-1, student_cache[group].shape[-1]).float()
                            xe = expert_cache[group].reshape(-1, expert_cache[group].shape[-1]).float()
                            if xs.shape != xe.shape:
                                raise RuntimeError("FeatCal student/expert feature shapes differ")
                            target = alpha * xe + (1.0 - alpha) * xs
                            gram = xs.T @ xs
                            cross = xs.T @ target
                            gram_sums[group] = gram if group not in gram_sums else gram_sums[group].add_(gram)
                            cross_sums[group] = cross if group not in cross_sums else cross_sums[group].add_(cross)
                            rows[group] = rows.get(group, 0) + xs.shape[0]
                seen += 1
                tokens += sample.tokens
    finally:
        for handle in student_handles + expert_handles:
            handle.remove()
    return (
        {group: gram_sums[group].div(rows[group]).cpu() for group in groups},
        {group: cross_sums[group].div(rows[group]).cpu() for group in groups},
        groups,
        {"examples": seen, "tokens": tokens, "feature_rows": rows,
         "prefill_chunk_size": prefill_chunk_size},
    )

def execute(config: RunConfig, workspace: Path) -> DataResult:
    import torch

    del workspace
    assert config.initial_model is not None
    device = config.runtime["device"]
    tokenizer = load_tokenizer(config.base)
    student = load_model(config.initial_model, device, precision(config))
    layers = model_layers(student)
    tied = bool(student.config.tie_word_embeddings)
    base_map = read_map(config.base)
    outputs = checkpoint_tensors(config, config.initial_model)
    per_layer = []
    domain_totals = {expert.id: {"examples": 0, "tokens": 0} for expert in config.experts}
    ridge = float(config.method_parameters["ridge_lambda"])
    rho = float(config.method_parameters["anchor_rho"])
    alpha = float(config.method_parameters["teacher_alpha"])
    eps = float(config.method_parameters["covariance_eps"])
    examples = int(config.method_parameters["examples"])
    chunk_size = int(config.method_parameters["prefill_chunk_size"])

    with ExitStack() as stack:
        base_handle = open_model(stack, config.base, base_map)
        sources = {}
        student_inputs, expert_inputs = {}, {}
        with torch.inference_mode():
            for source in config.experts:
                mapping = read_map(source.path)
                handle = open_model(stack, source.path, mapping)
                sources[source.id] = (handle, mapping)
                embedding = tensor(handle, mapping, "model.embed_tokens.weight").to(
                    dtype=student.model.embed_tokens.weight.dtype)
                si, ei = [], []
                for sample in iter_features(tokenizer, source.data, "cpu", examples):
                    si.append(student.model.embed_tokens(sample.input_ids.to(device)).cpu())
                    ei.append(torch.nn.functional.embedding(sample.input_ids, embedding).cpu())
                student_inputs[source.id], expert_inputs[source.id] = si, ei
                del embedding, si, ei
        for layer_index, student_layer in enumerate(layers):
            print("FeatCal layer {}/{}".format(layer_index + 1, len(layers)), flush=True)
            modules, groups = _groups(student_layer)
            grams = {group: [] for group in groups}
            crosses = {group: [] for group in groups}
            expert_weights = {name: [] for name in modules}
            summaries = {}
            for source in config.experts:
                handle, mapping = sources[source.id]
                expert_layer = copy.deepcopy(student_layer)
                prefix = "model.layers.{}.".format(layer_index)
                expert_layer.load_state_dict({k: tensor(handle, mapping, prefix + k)
                                              for k in expert_layer.state_dict()}, strict=True)
                domain_grams, domain_crosses, domain_groups, summary, next_inputs = collect(
                    student, student_layer, expert_layer, student_inputs[source.id],
                    expert_inputs[source.id], alpha, chunk_size)
                expert_inputs[source.id] = next_inputs
                if groups != domain_groups:
                    raise RuntimeError("expert decoder-layer group layouts differ")
                for group in groups:
                    grams[group].append(domain_grams[group])
                    crosses[group].append(domain_crosses[group])
                expert_state = expert_layer.state_dict()
                for name in modules:
                    expert_weights[name].append(expert_state[name + ".weight"].detach().cpu())
                summaries[source.id] = summary
                domain_totals[source.id]["examples"] += summary["examples"]
                domain_totals[source.id]["tokens"] += summary["tokens"]
                del expert_layer, expert_state, domain_grams, domain_crosses, next_inputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            layer_updates = {}
            solver_summary = []
            current_state = student_layer.state_dict()
            prefix = "model.layers.{}.".format(layer_index)
            for group, names in groups.items():

                keys = [name + ".weight" for name in names]
                weights = [torch.cat([expert_weights[n][i] for n in names]) for i in range(len(config.experts))]
                merged_weight = torch.cat([current_state[k].detach() for k in keys])
                base_weight = torch.cat([tensor(base_handle, base_map, prefix + k) for k in keys])
                value, solver = solve_weight(
                    weights, grams[group], crosses[group], merged_weight, base_weight,
                    ridge, rho, eps, device,
                )
                offset = 0
                for key in keys:
                    count = current_state[key].shape[0]
                    layer_updates[key] = value[offset:offset + count].to(current_state[key].dtype)
                    offset += count
                solver_summary.append({"modules": names, **solver})
                del weights, merged_weight, base_weight, value
            student_layer.load_state_dict(layer_updates, strict=False)
            if layer_index < len(layers) - 1:
                for source in config.experts:
                    student_inputs[source.id] = advance(
                        student, student_layer, student_inputs[source.id], chunk_size)
            for local_key, value in layer_updates.items():
                global_key = prefix + local_key
                outputs[global_key] = value.to(outputs[global_key].dtype).cpu()
            per_layer.append(
                {
                    "layer": layer_index,
                    "groups": len(groups),
                    "linear_weights": len(layer_updates),
                    "domains": summaries,
                    "solver": solver_summary,
                }
            )
            del grams, crosses, expert_weights, layer_updates
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    del student
    return DataResult(
        outputs,
        {
            "ridge_lambda": ridge,
            "anchor_rho": rho,
            "teacher_alpha": alpha,
            "covariance_eps": eps,
            "forward_precision": precision(config),
            "statistics_and_solver_precision": "float32",
            "prefill_chunk_size": chunk_size,
            "examples_per_domain_per_layer": examples,
            "schedule": "forward-order CPU boundary cache; updated student layer replay only",
            "feature_cache": "cpu",
            "expert_loading": "checkpoint tensors for current decoder layer only",
            "calibrated_scope": "decoder attention and MLP linear weights",
            "preserved_scope": "initial checkpoint embeddings, RMSNorm, final norm, and lm_head",
            "edge_policy": {
                "source": "initial_model",
                "embedding": "preserved subject to explicit save_dtype",
                "lm_head": "preserved subject to explicit save_dtype",
                "tie_word_embeddings": tied,
                "stored_units": 1 if tied else 2,
            },
            "domains": domain_totals,
            "layers": per_layer,
        },
    )
