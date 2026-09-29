from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

from ...config import RunConfig
from ...data_engine import DataResult
from ...data_ops import precision, source_dtype
from ...data_runtime import iter_prompts, load_model, load_tokenizer
from ...engine import open_model, read_map, tensor

_GRADIENT_CHUNK = 8_388_608

def _load_working_parameters(named, base_weights, coefficients, deltas, groups, device):
    import torch

    with torch.no_grad():
        effective = coefficients.clamp(0.0, 1.0)
        for name, parameter in named.items():
            parameter.copy_(base_weights[name])
            group = groups[name]
            for expert_index, expert_deltas in enumerate(deltas):
                parameter.add_(
                    expert_deltas[name].to(device),
                    alpha=float(effective[group, expert_index]),
                )

def _enable_recomputation(model):

    import torch

    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

def _coefficient_vjp(merged, deltas, groups, shape, device, raw_coefficients=None):
    import torch

    result = torch.zeros(shape, dtype=torch.float32, device=device)
    for name, value in merged.items():
        if value.grad is None:
            continue
        gradient = value.grad.detach().reshape(-1)
        group = groups[name]
        for expert_index, expert_deltas in enumerate(deltas):
            delta = expert_deltas[name].reshape(-1)
            for start in range(0, gradient.numel(), _GRADIENT_CHUNK):
                stop = min(start + _GRADIENT_CHUNK, gradient.numel())
                result[group, expert_index].add_(
                    (gradient[start:stop] * delta[start:stop].to(device)).sum(
                        dtype=torch.float32
                    )
                )
    if raw_coefficients is not None:

        result.mul_((raw_coefficients.detach() >= 0) & (raw_coefficients.detach() <= 1))
    return result

def _parameter_groups(named):

    identities, groups = {}, {}
    for name, parameter in named.items():
        identities.setdefault(id(parameter), len(identities))
        groups[name] = identities[id(parameter)]
    return groups

def _next(iterator, tokenizer, path, device):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter_prompts(tokenizer, path, device)
        return next(iterator), iterator

def execute(config: RunConfig, workspace: Path) -> DataResult:
    import json
    import torch
    import torch.nn.functional as functional

    del workspace
    device = config.runtime["device"]
    model = load_model(config.base, "cpu", precision(config), gradients=False)

    base_weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    model.to(device)
    _enable_recomputation(model)
    tokenizer = load_tokenizer(config.base)
    named = dict(model.named_parameters())
    tied = bool(json.loads((config.base / "config.json").read_text())["tie_word_embeddings"])
    groups = _parameter_groups(named)
    embedding_group = groups["model.embed_tokens.weight"]
    group_count = max(groups.values()) + 1
    roots = [config.base] + [item.path for item in config.experts]
    maps = [read_map(root) for root in roots]
    compute_dtype = torch.bfloat16 if precision(config) == "mergebench" else torch.float32
    with ExitStack() as stack:
        opened = [open_model(stack, root, mapping) for root, mapping in zip(roots, maps)]
        deltas = []
        for expert_index in range(len(config.experts)):
            values = {}
            for name in named:
                base = tensor(opened[0], maps[0], name).to(compute_dtype)
                expert = tensor(opened[expert_index + 1], maps[expert_index + 1], name).to(
                    compute_dtype
                )
                values[name] = expert - base
            deltas.append(values)

        expert_count = len(config.experts)
        coefficients = torch.nn.Parameter(
            torch.full(
                (group_count, expert_count),
                float(config.method_parameters["initial_coefficient"]),
                dtype=torch.float32,
                device=device,
            )
        )
        embed_mode = config.method_parameters.get("embed_coefficient", "learned")
        embed_fill = (
            None
            if embed_mode == "learned"
            else (0.0 if embed_mode == "zero" else 1.0 / expert_count)
        )
        if embed_fill is not None:
            with torch.no_grad():
                coefficients[embedding_group].fill_(embed_fill)
        for name, parameter in named.items():
            parameter.requires_grad_(embed_fill is None or groups[name] != embedding_group)
        optimizer = torch.optim.Adam(
            [coefficients], lr=float(config.method_parameters["learning_rate"])
        )
        iterators = [
            iter_prompts(tokenizer, item.data, device) for item in config.experts
        ]
        entropy_sum = 0.0
        steps = int(config.method_parameters["steps"])
        max_prompt_tokens = int(config.method_parameters["max_prompt_tokens"])
        for step in range(steps):
            optimizer.zero_grad()
            model.zero_grad(set_to_none=True)
            _load_working_parameters(
                named, base_weights, coefficients, deltas, groups, device
            )
            for domain, expert in enumerate(config.experts):
                sample, iterators[domain] = _next(
                    iterators[domain], tokenizer, expert.data, device
                )

                output = model(
                    input_ids=sample.input_ids[:, :max_prompt_tokens],
                    attention_mask=sample.attention_mask[:, :max_prompt_tokens],
                    use_cache=False,
                    logits_to_keep=1,
                )
                logits = output.logits[:, -1].float()
                entropy = -(
                    functional.softmax(logits, -1) * functional.log_softmax(logits, -1)
                ).sum(-1).mean()
                entropy.backward()
                entropy_sum += float(entropy.detach())
                del output, logits, entropy, sample
            coefficients.grad = _coefficient_vjp(
                named, deltas, groups, coefficients.shape, device, coefficients
            )
            if embed_fill is not None and coefficients.grad is not None:
                coefficients.grad[embedding_group].zero_()
            optimizer.step()
            if embed_fill is not None:
                with torch.no_grad():
                    coefficients[embedding_group].fill_(embed_fill)
            if device == "cuda" and (step == 0 or (step + 1) % 10 == 0):
                print(config.run_id, "step", step + 1, "/", steps, flush=True)

        model.zero_grad(set_to_none=True)
        outputs = {}
        with torch.no_grad():
            effective = coefficients.clamp(0.0, 1.0)
            for key in sorted(maps[0]):
                base = tensor(opened[0], maps[0], key)
                if not base.is_floating_point():
                    outputs[key] = base
                    continue
                canonical = "model.embed_tokens.weight" if tied and key == "lm_head.weight" else key
                group = groups[canonical]
                base_compute = base.to(device, compute_dtype)
                merged = base_compute.clone()
                for expert_index in range(len(config.experts)):
                    merged.add_(
                        deltas[expert_index][canonical].to(device),
                        alpha=float(effective[group, expert_index]),
                    )
                outputs[key] = merged.to(source_dtype(config, base)).cpu()
    return DataResult(
        outputs,
        {
            "objective": "mean Shannon entropy of the final next-token distribution",
            "objective_scope": "decoder extension; original paper uses classifier logits",
            "input_view": "prefix before the final assistant response with generation prompt",
            "steps": steps,
            "max_prompt_tokens": max_prompt_tokens,
            "mean_training_entropy": entropy_sum / (steps * len(config.experts)),
            "coefficient_groups": group_count,
            "embed_coefficient": embed_mode,
            "final_coefficients": effective.detach().cpu().tolist(),
            "raw_coefficients": coefficients.detach().cpu().tolist(),
            "coefficient_group_map": groups,
            "coefficient_granularity": "one row per unique parameter tensor",
            "coefficient_constraint": "effective clamp[0,1]; raw Adam state not projected",
            "coefficient_gradient": "chunked VJP with clamp derivative; runtime-precision arithmetic",
            "base_storage": "cpu",
            "working_model_copies_on_device": 1,
            "gradient_checkpointing": "non_reentrant_training_zero_dropout",
            "fixed_embedding_gradient": embed_fill is None,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
        },
    )
