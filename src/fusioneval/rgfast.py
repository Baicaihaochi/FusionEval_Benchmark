from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Any, Dict, Mapping

from .config import RunConfig
from .data_engine import DataResult
from .data_ops import apply_leftover, average_experts, precision
from .data_runtime import iter_examples, linear_modules, load_model, load_tokenizer

_CACHE: Dict[Any, Any] = {}

def gram_key(module_name: str) -> str:
    parent, _, leaf = module_name.rpartition(".")
    if leaf in {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}:
        return parent + ".__input__"
    return module_name

def signature(config: RunConfig):
    return (
        str(config.base),
        tuple((str(item.path), str(item.data)) for item in config.experts),
        int(config.method_parameters["examples"]),
        str(config.method_parameters.get("gram_dtype", "float32")),
        bool(config.method_parameters.get("edge_head", "leftover") == "regmean"),
        str(config.runtime["precision"]),
    )

HEAD_KEY = "__head_input__"

def collect_expert(config: RunConfig, expert, tokenizer, device: str, examples: int,
                   gram_dtype: str, head_in_gram: bool = False):

    import torch

    model = load_model(expert.path, device, precision(config))
    modules = linear_modules(model)
    representatives: Dict[str, Any] = {}
    module_to_gram: Dict[str, str] = {}
    module_weight: Dict[str, Any] = {}
    for name, module in modules.items():
        if name == "lm_head":
            continue
        key = gram_key(name)
        module_to_gram[name] = key
        representatives.setdefault(key, module)
        module_weight[name] = module.weight
    if head_in_gram:
        final_norm = getattr(getattr(model, "model", None), "norm", None)
        head = getattr(model, "lm_head", None)
        if final_norm is None or head is None:
            raise RuntimeError("lm_head input Gram requested but the backbone has no final norm")
        representatives[HEAD_KEY] = final_norm
        module_to_gram["lm_head"] = HEAD_KEY
        module_weight["lm_head"] = head.weight
    grams: Dict[str, Any] = {}
    counts: Dict[str, int] = {}

    def capture(key):
        def hook(_module, inputs, _output):
            source = _output if key == HEAD_KEY else inputs[0]
            value = source.detach().reshape(-1, source.shape[-1])
            if gram_dtype == "float32":
                value = value.float()
            current = value.transpose(0, 1) @ value
            rows = value.shape[0]
            if key not in grams:
                grams[key], counts[key] = current / rows, rows
            else:
                total = counts[key] + rows
                grams[key].mul_(counts[key] / total).add_(current, alpha=1.0 / total)
                counts[key] = total
        return hook

    handles = [module.register_forward_hook(capture(key))
               for key, module in representatives.items()]
    seen = tokens = 0
    try:
        with torch.inference_mode():
            for sample in iter_examples(tokenizer, expert.data, device, examples):
                model.model(
                    input_ids=sample.input_ids,
                    attention_mask=sample.attention_mask,
                    use_cache=False,
                )
                seen += 1
                tokens += sample.tokens
    finally:
        for handle in handles:
            handle.remove()
    weights = {name: module_weight[name].detach().float().cpu()
               for name in module_to_gram}
    dropped = {key: value.float().cpu().clone() for key, value in grams.items()}
    summary = {"examples": seen, "tokens": tokens, "gram_rows": dict(counts),
               "gram_dtype": gram_dtype}
    del model, grams, representatives, module_weight
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dropped, weights, module_to_gram, summary

def build_state(config: RunConfig) -> Dict[str, Any]:
    import torch

    device = config.runtime["device"]
    examples = int(config.method_parameters["examples"])
    gram_dtype = str(config.method_parameters.get("gram_dtype", "float32"))
    tokenizer = load_tokenizer(config.base)
    sum_g: Dict[str, Any] = {}
    sum_diag: Dict[str, Any] = {}
    accum_a: Dict[str, Any] = {}
    accum_b: Dict[str, Any] = {}
    expert_diag = []
    reports = []
    module_to_gram = None
    for expert in config.experts:
        head_in_gram = str(config.method_parameters.get("edge_head", "leftover")) == "regmean"
        grams, weights, current_map, summary = collect_expert(
            config, expert, tokenizer, device, examples, gram_dtype, head_in_gram)
        if module_to_gram is None:
            module_to_gram = current_map
        elif module_to_gram != current_map:
            raise RuntimeError("expert linear-module layouts differ")
        diags = {}
        for key, value in grams.items():
            if key in sum_g:
                sum_g[key].add_(value)
            else:
                sum_g[key] = value
            diagonal = value.diagonal().clone()
            diags[key] = diagonal
            if key in sum_diag:
                sum_diag[key].add_(diagonal)
            else:
                sum_diag[key] = diagonal
        expert_diag.append(diags)
        for name, key in module_to_gram.items():
            transposed = weights[name].transpose(0, 1).contiguous()
            contribution_a = grams[key] @ transposed
            contribution_b = diags[key].unsqueeze(1) * transposed
            if name in accum_a:
                accum_a[name].add_(contribution_a)
                accum_b[name].add_(contribution_b)
            else:
                accum_a[name], accum_b[name] = contribution_a, contribution_b
        reports.append({"expert": expert.id, **summary})
        del grams, weights, diags
        gc.collect()
    outputs = apply_leftover(config, average_experts(config))
    modules_by_gram: Dict[str, list] = {}
    for name, key in module_to_gram.items():
        modules_by_gram.setdefault(key, []).append(name)
    return {
        "sum_g": sum_g, "sum_diag": sum_diag, "accum_a": accum_a, "accum_b": accum_b,
        "expert_diag": expert_diag, "outputs": outputs, "module_to_gram": module_to_gram,
        "modules_by_gram": modules_by_gram, "reports": reports, "gram_dtype": gram_dtype,
    }

def solve_state(config: RunConfig, state: Mapping[str, Any], alpha: float, variant: str,
                solver: str):
    import torch

    device = config.runtime.get("device", "cpu")
    outputs = dict(state["outputs"])
    sum_diag = state["sum_diag"]
    accum_a, accum_b = state["accum_a"], state["accum_b"]
    modules_by_gram = state["modules_by_gram"]
    diagnostics: Dict[str, Any] = {
        "alpha": alpha, "variant": variant, "solver": solver,
        "gram_dtype": state["gram_dtype"], "systems": 0, "truncated_directions": 0,
        "head_in_gram": bool("lm_head" in state["modules_by_gram"]),
        "surveyed_systems": 0, "max_condition_number": 0.0, "worst_system": None,
        "min_positive_eigenvalue_ratio": None,
        "solve_seconds": 0.0, "objective": {}, "weight_delta": {},
    }
    objective_merged = 0.0
    objective_average = 0.0
    delta_sq = 0.0
    average_sq = 0.0
    started = time.time()
    for key, gram in state["sum_g"].items():
        diagonal = sum_diag[key]
        matrix = gram * alpha
        matrix.diagonal().copy_(diagonal)
        scale = None
        if variant == "white":
            scale = 1.0 / diagonal.clamp(min=1e-30).sqrt()
            matrix.mul_(scale.unsqueeze(0)).mul_(scale.unsqueeze(1))
        matrix = matrix.to(device)
        if solver == "exact":

            inverse_matrix = None
        else:
            rtol = 1e-12 if solver == "pinv1e12" else float(torch.finfo(torch.float32).eps) * matrix.shape[0]
            inverse_matrix = torch.linalg.pinv(matrix, rtol=rtol)
        diagnostics["systems"] += 1
        for name in modules_by_gram[key]:
            weight_key = name + ".weight"
            if weight_key not in state["outputs"]:
                continue
            rhs = accum_a[name] * alpha + accum_b[name] * (1.0 - alpha)
            if variant == "white":
                rhs = rhs * scale.unsqueeze(1)
            rhs = rhs.to(device)
            if solver == "exact":
                merged_prime = torch.linalg.solve(matrix, rhs)
            else:
                merged_prime = inverse_matrix @ rhs
            reference = state["outputs"][weight_key]
            reference_f = reference.float()

            working = merged_prime.float()
            reference_working = reference_f.transpose(0, 1).to(device)
            if variant == "white":
                reference_working = reference_working / scale.unsqueeze(1).to(device)
            objective_merged += float((working * (matrix @ working)).sum()) \
                - 2.0 * float((working * rhs).sum())
            objective_average += float((reference_working * (matrix @ reference_working)).sum()) \
                - 2.0 * float((reference_working * rhs).sum())
            merged = merged_prime.transpose(0, 1)
            if variant == "white":
                merged = merged * scale.unsqueeze(0).to(device)
            merged_cpu = merged.to(torch.float32).cpu()
            delta_sq += float(((merged_cpu - reference_f) ** 2).sum())
            average_sq += float((reference_f ** 2).sum())
            outputs[weight_key] = merged_cpu.to(reference.dtype)
            del rhs, merged_prime, working, reference_working, merged, merged_cpu
        del matrix, inverse_matrix
        if device != "cpu":
            torch.cuda.empty_cache()
    diagnostics["solve_seconds"] = time.time() - started
    diagnostics["objective"] = {
        "merged": objective_merged, "average": objective_average,
        "merged_minus_average": objective_merged - objective_average,
        "merged_is_better": bool(objective_merged < objective_average),
    }
    diagnostics["weight_delta"] = {
        "relative_l2_vs_expert_mean": math.sqrt(delta_sq / average_sq) if average_sq else None}
    return outputs, diagnostics

def run_regmean(config: RunConfig, workspace: Path) -> DataResult:
    key = signature(config)
    state = _CACHE.get(key)
    if state is None:
        built = time.time()
        state = build_state(config)
        state["build_seconds"] = time.time() - built
        _CACHE[key] = state
    alpha = float(config.method_parameters["alpha"])
    variant = str(config.method_parameters.get("variant", "raw"))
    solver = str(config.method_parameters.get("solver", "exact"))
    outputs, diagnostics = solve_state(config, state, alpha, variant, solver)
    diagnostics["state_build_seconds"] = state["build_seconds"]
    diagnostics["domains"] = [
        {"id": report["expert"], "examples": report["examples"], "tokens": report["tokens"],
         "gram_rows_sample": len(report["gram_rows"]), "gram_dtype": report["gram_dtype"]}
        for report in state["reports"]]
    return DataResult(outputs, {
        "alpha": alpha,
        "variant": variant,
        "solver": solver,
        "gram_dtype": state["gram_dtype"],
        "examples": int(config.method_parameters["examples"]),
        "leftover_edge": config.method_parameters.get("leftover_edge", "mean"),
        "leftover_1d": config.method_parameters.get("leftover_1d", "mean"),
        "nonlinear_and_edge_policy": "leftover_edge={};leftover_1d={}".format(
            config.method_parameters.get("leftover_edge", "mean"),
            config.method_parameters.get("leftover_1d", "mean")),
        "solver_diagnostics": diagnostics,
    })
