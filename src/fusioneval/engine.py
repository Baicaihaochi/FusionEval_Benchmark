from __future__ import annotations

import importlib
import json
import gc
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Mapping

from safetensors import safe_open

from .checkpoint import inspect_checkpoints
from .config import RunConfig
from .errors import CheckpointError
from .protocol import Block, SharedContext
from .source import Source, streaming_thresholds

def read_map(root: Path) -> Dict[str, str]:
    index = root / "model.safetensors.index.json"
    single = root / "model.safetensors"
    if index.is_file():
        return json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    if single.is_file():
        with safe_open(str(single), framework="numpy") as handle:
            return {key: single.name for key in handle.keys()}
    raise CheckpointError(f"missing safetensors weights under {root}")

def member(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if path.parent != root.resolve():
        raise CheckpointError("checkpoint index references a non-local shard")
    return path

def open_model(stack: ExitStack, root: Path, mapping: Mapping[str, str]):
    return {
        shard: stack.enter_context(safe_open(str(member(root, shard)), framework="pt", device="cpu"))
        for shard in sorted(set(mapping.values()))
    }

def tensor(opened, mapping, key):
    return opened[mapping[key]].get_tensor(key)

class CheckpointSource:

    def __init__(self, opened, mappings, keys, compute_dtype, tied, keep_experts=False):
        self.opened = opened
        self.mappings = mappings
        self.keys = tuple(keys)
        self.compute_dtype = compute_dtype
        self.tied = tied
        self.keep_experts = keep_experts

    def load(self, key):
        import torch

        values = [
            tensor(handle, mapping, key)
            for handle, mapping in zip(self.opened, self.mappings)
        ]
        base, experts = values[0], values[1:]
        if not base.is_floating_point():
            if any(not torch.equal(base, expert) for expert in experts):
                raise CheckpointError("non-floating tensor differs at {}".format(key))
            return None, base
        base_compute = base.to(self.compute_dtype)
        expert_compute = tuple(value.to(self.compute_dtype) for value in experts)
        deltas = (
            ()
            if self.keep_experts
            else tuple(value - base_compute for value in expert_compute)
        )
        block = Block(
            key,
            tuple(base.shape),
            0,
            base.numel(),
            0,
            base.shape[0] if base.ndim else 1,
            roles(key, self.tied),
            base_compute,
            () if self.keep_experts else deltas,
            expert_compute if self.keep_experts else (),
        )
        return block, base

    def scan(self):
        for key in self.keys:
            block, _ = self.load(key)
            if block is not None:
                yield block

    def global_thresholds(self, density, rule):
        return streaming_thresholds(self, density, rule)

def copy_metadata(base: Path, staging: Path) -> None:
    for source in base.iterdir():
        is_weight = (
            source.name == "model.safetensors"
            or source.name == "model.safetensors.index.json"
            or (source.name.startswith("model-") and source.name.endswith(".safetensors"))
        )
        if source.is_file() and not is_weight:
            shutil.copy2(source, staging / source.name)

class ShardedWriter:

    def __init__(self, staging: Path, max_bytes: int):
        self.staging = staging
        self.max_bytes = max_bytes
        self.current = {}
        self.current_bytes = 0
        self.total_bytes = 0
        self.parts = []

    def add(self, key, value) -> None:
        value = value.detach().contiguous().cpu()
        value_bytes = value.numel() * value.element_size()
        if self.current and self.current_bytes + value_bytes > self.max_bytes:
            self._flush()
        self.current[key] = value
        self.current_bytes += value_bytes
        self.total_bytes += value_bytes

    def _flush(self) -> None:
        from safetensors.torch import save_file

        if not self.current:
            return
        part = self.staging / ".part-{:05d}.safetensors".format(len(self.parts) + 1)
        keys = tuple(self.current)
        save_file(self.current, str(part), metadata={"format": "pt"})
        self.parts.append((part, keys))
        self.current = {}
        self.current_bytes = 0

    def close(self) -> Dict[str, str]:
        self._flush()
        if not self.parts:
            raise CheckpointError("cannot write an empty checkpoint")
        count = len(self.parts)
        weight_map = {}
        for index, (part, keys) in enumerate(self.parts, 1):
            name = (
                "model.safetensors"
                if count == 1
                else "model-{:05d}-of-{:05d}.safetensors".format(index, count)
            )
            os.rename(part, self.staging / name)
            weight_map.update({key: name for key in keys})
        if count > 1:
            index = {
                "metadata": {"total_size": self.total_bytes},
                "weight_map": weight_map,
            }
            (self.staging / "model.safetensors.index.json").write_text(
                json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return weight_map

def save_tensors(tensors, staging: Path, max_bytes: int) -> Dict[str, str]:

    writer = ShardedWriter(staging, max_bytes)
    for key, value in tensors.items():
        writer.add(key, value)
    return writer.close()

def validate_saved(staging: Path, expected, tie_word_embeddings: bool) -> Dict[str, object]:
    import gc
    import torch

    mapping = read_map(staging)
    if set(mapping) != set(expected):
        raise CheckpointError("saved tensor keys differ from the source schema")
    count = 0
    for shard in sorted(set(mapping.values())):
        with safe_open(str(staging / shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                value = handle.get_tensor(key)
                item = expected[key]
                expected_shape, expected_dtype = (
                    (tuple(item.shape), item.dtype)
                    if hasattr(item, "shape")
                    else item
                )
                if tuple(value.shape) != tuple(expected_shape) or value.dtype != expected_dtype:
                    raise CheckpointError(f"saved schema mismatch at {key}")
                if value.is_floating_point() and not bool(value.isfinite().all()):
                    raise CheckpointError(f"non-finite saved tensor at {key}")
                count += 1
    if count != len(expected):
        raise CheckpointError("saved tensor count mismatch")
    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(staging), torch_dtype="auto", low_cpu_mem_usage=True
        )
        tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
        if tied != tie_word_embeddings:
            raise CheckpointError("saved model does not preserve the configured embedding/head tying")
        vocab = int(model.config.vocab_size)
        ids = torch.tensor([[min(1, vocab - 1), min(2, vocab - 1)]])
        with torch.inference_mode():
            logits = model(input_ids=ids).logits
        if not bool(torch.isfinite(logits).all()):
            raise CheckpointError("saved model produced non-finite fixed-prefix logits")
        result = {"reload": True, "weight_tying": tied, "finite_prefix_forward": True}
        del model, logits
        gc.collect()
        return result
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"Transformers reload validation failed: {exc}") from exc

def roles(key: str, tied: bool):
    if key == "model.embed_tokens.weight":
        return ("input_embedding", "output_head") if tied else ("input_embedding",)
    if key == "lm_head.weight":
        return ("output_head",)
    if "norm" in key.lower():
        return ("normalization",)
    return ("transformer",)

def move(block: Block, device: str) -> Block:
    if device == "cpu":
        return block
    return Block(
        block.key, block.shape, block.flat_start, block.flat_stop,
        block.row_start, block.row_stop, block.roles, block.base.to(device),
        tuple(value.to(device) for value in block.deltas),
        tuple(value.to(device) for value in block.experts),
    )

def output_dtype(torch, source, setting: str):
    return {
        "source": source.dtype,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[setting]

def validate_output_tensors(config: RunConfig, tensors: Mapping[str, object]) -> None:

    import torch

    mapping = read_map(config.base)
    expected_keys = set(mapping)
    actual_keys = set(tensors)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise CheckpointError(
            "runner returned an incomplete tensor inventory; missing={} extra={}".format(
                missing[:8], extra[:8]
            )
        )
    with ExitStack() as stack:
        opened = open_model(stack, config.base, mapping)
        for key in sorted(mapping):
            reference = tensor(opened, mapping, key)
            value = tensors[key]
            if not isinstance(value, torch.Tensor):
                raise CheckpointError("runner returned a non-tensor value at {}".format(key))
            if tuple(value.shape) != tuple(reference.shape):
                raise CheckpointError("runner returned the wrong shape at {}".format(key))
            expected_dtype = (
                output_dtype(torch, reference, config.runtime["save_dtype"])
                if reference.is_floating_point()
                else reference.dtype
            )
            if value.dtype != expected_dtype:
                raise CheckpointError(
                    "runner returned {} at {}, expected {}".format(
                        value.dtype, key, expected_dtype
                    )
                )
            if value.is_floating_point() and not bool(value.isfinite().all()):
                raise CheckpointError("runner returned non-finite values at {}".format(key))

def run(config: RunConfig, *, provenance=None):
    if config.method.execution == "data":
        from .data_engine import run as run_data_method

        return run_data_method(config, provenance=provenance)

    import torch

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    inspection = inspect_checkpoints(config)
    preparation_done = time.perf_counter()
    if config.output.exists():
        raise CheckpointError(f"output already exists: {config.output}")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    sources = [config.base] + [expert.path for expert in config.experts]
    maps = [read_map(path) for path in sources]
    keys = sorted(maps[0])
    if config.runtime["device"] == "cuda" and not torch.cuda.is_available():
        raise CheckpointError("runtime.device=cuda but CUDA is unavailable")
    effective_precision = (
        "float32"
        if config.method.compute_precision == "float32"
        else config.runtime["precision"]
    )
    compute_dtype = torch.bfloat16 if effective_precision == "mergebench" else torch.float32
    tied = bool(json.loads((config.base / "config.json").read_text(encoding="utf-8"))["tie_word_embeddings"])
    module = importlib.import_module(config.method.kernel_module)
    kernel = module.build(
        config.method.encode(config.method_parameters),
        SharedContext(
            config.runtime["seed"],
            effective_precision,
            config.runtime["device"],
            config.model_profile.num_hidden_layers if config.model_profile else None,
            config.model_profile.layer_prefix if config.model_profile else "model.layers",
            config.model_profile.attention_modules
            if config.model_profile
            else ("q_proj", "k_proj", "v_proj", "o_proj"),
            config.model_profile.mlp_modules
            if config.model_profile
            else ("gate_proj", "up_proj", "down_proj"),

            rng_state=config.runtime.get("rng_state"),
        ),
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{config.run_id}.", dir=str(config.output.parent)))
    try:
        method_started = time.perf_counter()
        coverage, expected = {}, {}
        with ExitStack() as stack:
            opened = [
                open_model(stack, path, mapping)
                for path, mapping in zip(sources, maps)
            ]
            source = CheckpointSource(
                opened,
                maps,
                keys,
                compute_dtype,
                tied,
                keep_experts=config.method.input_mode == "experts",
            )
            scope = config.method.requirements.prepare_scope
            state = kernel.prepare(source) if scope == "global" else None
            writer = ShardedWriter(
                staging, int(config.runtime["max_shard_size_gib"] * 2**30)
            )
            for key in keys:
                block, original = source.load(key)
                if block is None:
                    output = original
                else:
                    active = move(block, config.runtime["device"])
                    local_state = (
                        kernel.prepare(Source((active,)))
                        if scope == "tensor"
                        else state
                    )
                    merged = kernel.combine(active, local_state).cpu()
                    del active, local_state
                    output = merged.to(
                        output_dtype(
                            torch, original, config.runtime["save_dtype"]
                        )
                    )
                    for group in block.roles:
                        coverage[group] = coverage.get(group, 0) + merged.numel()
                writer.add(key, output)
                expected[key] = (tuple(output.shape), output.dtype)
            diagnostics = kernel.diagnostics(state)
            mapping = writer.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        method_done = time.perf_counter()
        save_started = time.perf_counter()
        copy_metadata(config.base, staging)
        validation = validate_saved(staging, expected, tied)
        files = {name: {"bytes": (staging / name).stat().st_size}
                 for name in sorted(set(mapping.values()))}
        index_path = staging / "model.safetensors.index.json"
        if index_path.is_file():
            files[index_path.name] = {
                "bytes": index_path.stat().st_size,
            }
        finished = time.perf_counter()
        finished_at = datetime.now(timezone.utc)
        timing = {
            "protocol": "mergebench-4.3-wall-clock-v1",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "preparation_seconds": preparation_done - started,
            "method_seconds": method_done - method_started,
            "save_validation_seconds": finished - save_started,
            "algorithm_runtime_seconds": finished - started,
            "single_gpu_algorithm_seconds": method_done - method_started,
            "algorithm_device_count": 1,
            "algorithm_runtime_scope": "single_gpu_sequential",
            "visible_cuda_device_count": (
                int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            ),
            "full_models_loaded_on_device": 0,
        }
        contract = config.method.effective_contract(config.method_parameters)
        manifest = {
            "status": "PASS", "method": config.method.local_name,
            "contract_id": contract["id"],
            "contract_variant": contract["variant"],
            "contract_tags": contract["tags"],
            "requested_precision": config.runtime["precision"],
            "effective_precision": effective_precision,
            "model_profile": config.model_profile.name if config.model_profile else "auto",
            "parameters": config.method_parameters, "source_inspection": inspection,
            "coverage": coverage, "diagnostics": diagnostics,
            "validation": validation, "timing": timing,
            "elapsed_seconds": timing["algorithm_runtime_seconds"],
            "weight_files": files,
        }
        if provenance is not None:

            manifest["orchestration"] = dict(provenance)
        (staging / "fusioneval_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging, config.output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
