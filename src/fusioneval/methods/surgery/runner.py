from __future__ import annotations

import gc
import json
import math
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from ...config import RunConfig
from ...data_engine import DataResult
from ...data_ops import checkpoint_tensors, precision
from ...data_runtime import iter_features, load_model, load_tokenizer
from ...feature_runtime import hidden_chunks
from .runtime import correct_hidden

def _adapter(hidden_size, rank, device):
    import torch

    down = torch.nn.Linear(hidden_size, rank, bias=False, device=device, dtype=torch.float32)
    up = torch.nn.Linear(rank, hidden_size, bias=False, device=device, dtype=torch.float32)
    torch.nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
    torch.nn.init.zeros_(up.weight)
    return down, up

def _next(iterator, tokenizer, path, device, examples):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter_features(tokenizer, path, device, examples)
        return next(iterator), iterator

def _pairs(student, expert, sample, chunk_size):
    with closing(hidden_chunks(student, sample, chunk_size)) as sc, closing(hidden_chunks(expert, sample, chunk_size)) as ec:
        for _ in range(0, sample.tokens, chunk_size):
            yield next(sc), next(ec)

def domain_features(student, expert, tokenizer, path, device, examples, steps, chunk_size, cache_mode):

    if cache_mode == "none":
        iterator = iter_features(tokenizer, path, device, examples)
        try:
            for _ in range(steps):
                sample, iterator = _next(iterator, tokenizer, path, device, examples)
                yield sample.tokens, _pairs(student, expert, sample, chunk_size)
        finally:
            iterator.close()
        return
    samples = list(iter_features(tokenizer, path, "cpu", examples))
    cache = {}
    for step in range(steps):
        index = step % len(samples)
        sample = samples[index]
        if index not in cache:
            moved = replace(sample, input_ids=sample.input_ids.to(device), attention_mask=sample.attention_mask.to(device))
            with closing(_pairs(student, expert, moved, chunk_size)) as pairs:
                cache[index] = [(s.cpu(), e.cpu()) for s, e in pairs]
        yield sample.tokens, ((s.to(device), e.to(device)) for s, e in cache[index])

def execute(config: RunConfig, workspace: Path) -> DataResult:
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import save_file

    assert config.initial_model is not None
    device = config.runtime["device"]
    torch.manual_seed(int(config.runtime["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.runtime["seed"]))
    tokenizer = load_tokenizer(config.base)
    student = load_model(config.initial_model, device, precision(config))
    hidden_size = int(student.config.hidden_size)
    tied = bool(student.config.tie_word_embeddings)
    rank = int(config.method_parameters["rank"])
    steps = int(config.method_parameters["steps"])
    examples = int(config.method_parameters["examples"])
    learning_rate = float(config.method_parameters["learning_rate"])
    chunk_size = int(config.method_parameters["prefill_chunk_size"])
    cache_mode = config.method_parameters["feature_cache"]
    artifact_tensors = {}
    domains = {}

    for domain_index, source in enumerate(config.experts):
        expert = load_model(source.path, device, precision(config))
        torch.manual_seed(int(config.runtime["seed"]) + domain_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.runtime["seed"]) + domain_index)
        down, up = _adapter(hidden_size, rank, device)
        optimizer = torch.optim.Adam(
            [down.weight, up.weight],
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        loss_sum = 0.0
        step_losses = []
        token_exposures = 0
        with closing(domain_features(student, expert, tokenizer, source.data, device, examples, steps, chunk_size, cache_mode)) as features:
            for step, (token_count, pairs) in enumerate(features):
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                with closing(pairs):
                    for student_hidden, expert_hidden in pairs:
                        corrected = correct_hidden(student_hidden.float(), down.weight, up.weight)

                        loss = functional.l1_loss(corrected, expert_hidden.float(), reduction="sum")
                        loss = loss / (token_count * hidden_size)
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError("non-finite Surgery feature loss")
                        loss.backward()
                        step_loss += float(loss.detach())
                optimizer.step()
                loss_sum += step_loss
                token_exposures += token_count
                step_losses.append(step_loss)
                if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
                    print("Surgery {} step {}/{} L1={:.6g}".format(source.id, step + 1, steps, step_loss), flush=True)
        if not all(bool(torch.isfinite(p).all()) for p in (down.weight, up.weight)):
            raise RuntimeError("non-finite Surgery adapter")
        artifact_tensors["experts.{}.down.weight".format(source.id)] = down.weight.detach().cpu()
        artifact_tensors["experts.{}.up.weight".format(source.id)] = up.weight.detach().cpu()
        domains[source.id] = {
            "steps": steps,
            "mean_l1_feature_loss": loss_sum / steps,
            "step_l1_losses": step_losses,
            "token_exposures": token_exposures,
        }
        del expert, down, up, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    artifact_started = time.perf_counter()
    adapter_path = workspace / "surgery_adapters.safetensors"
    save_file(artifact_tensors, str(adapter_path), metadata={"format": "pt"})
    config_path = workspace / "surgery_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": student.config.model_type,
                "compute_dtype": "float32",
                "feature_site": "model.model.norm output",
                "operation": "hidden - up(relu(down(hidden)))",
                "hidden_size": hidden_size,
                "rank": rank,
                "experts": [source.id for source in config.experts],
                "runtime": "fusioneval.methods.surgery.runtime.surgery_context",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_write_seconds = time.perf_counter() - artifact_started
    del student
    return DataResult(
        checkpoint_tensors(config, config.initial_model),
        {
            "rank": rank,
            "steps_per_domain": steps,
            "learning_rate": learning_rate,
            "examples_per_domain_cycle": examples,
            "prefill_chunk_size": chunk_size,
            "forward_precision": precision(config),
            "adapter_precision": "float32",
            "feature_cache": cache_mode,
            "artifact_write_seconds": artifact_write_seconds,
            "step_definition": "one Adam update per full sequence; token-weighted chunk gradients",
            "loss": "mean absolute distance between corrected merged and expert final hidden states",
            "feature_tokens": "all non-padding calibration tokens (batch size one)",
            "routing": "caller must select exactly one expert id",
            "checkpoint_weight_policy": "preserve initial merged checkpoint",
            "edge_policy": {
                "source": "initial_model",
                "embedding": "preserved subject to explicit save_dtype",
                "lm_head": "preserved; no expert-head routing",
                "tie_word_embeddings": tied,
                "stored_units": 1 if tied else 2,
                "feature_site": "after final norm, before initial_model lm_head",
            },
            "domains": domains,
        },
        {
            "surgery/surgery_adapters.safetensors": adapter_path,
            "surgery/surgery_config.json": config_path,
        },
    )
