from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from safetensors import SafetensorError, safe_open

from .config import RunConfig
from .errors import CheckpointError

_CONFIG_IGNORED = {"_name_or_path", "transformers_version", "torch_dtype", "dtype"}
_TOKENIZER_CONFIG_IGNORED = {"name_or_path", "tokenizer_file"}
_GENERATION_CONFIG_IGNORED = {"_from_model_config", "transformers_version"}
_OPTIONAL_METADATA_FILES = (
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
)
_REQUIRED_ARCHITECTURE_KEYS = (
    "model_type", "vocab_size", "hidden_size", "num_hidden_layers",
    "tie_word_embeddings",
)

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CheckpointError("missing {}".format(path)) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError("cannot read JSON {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise CheckpointError("expected JSON object: {}".format(path))
    return value

def _normalized(value: Mapping[str, Any], ignored: set) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if key not in ignored}

def _architecture(root: Path, profile=None) -> Dict[str, Any]:
    value = _normalized(_read_json(root / "config.json"), _CONFIG_IGNORED)
    missing = [key for key in _REQUIRED_ARCHITECTURE_KEYS if key not in value]
    if missing:
        raise CheckpointError("config.json missing architecture keys: {}".format(", ".join(missing)))
    if value["model_type"] not in ("qwen2", "qwen3", "llama"):
        raise CheckpointError(
            "the checkpoint inspector currently supports Qwen2/Qwen3 and Llama checkpoints only"
        )
    if not isinstance(value["tie_word_embeddings"], bool):
        raise CheckpointError("tie_word_embeddings must be boolean")
    if profile is not None:
        mismatches = {
            key: {"expected": expected, "actual": value.get(key)}
            for key, expected in profile.architecture.items()
            if value.get(key) != expected
        }
        if mismatches:
            raise CheckpointError(
                "checkpoint does not match model profile {}: {}".format(
                    profile.name, mismatches
                )
            )
    return value

def _metadata_identity(root: Path) -> Dict[str, Any]:
    tokenizer_config = _normalized(
        _read_json(root / "tokenizer_config.json"), _TOKENIZER_CONFIG_IGNORED
    )
    value = {
        "tokenizer_json": _read_json(root / "tokenizer.json"),
        "tokenizer_config": tokenizer_config,
        "optional_files": {
            name: (
                _read_json(root / name)
                if (root / name).is_file() and name.endswith(".json")
                else (root / name).read_text(encoding="utf-8")
                if (root / name).is_file()
                else None
            )
            for name in _OPTIONAL_METADATA_FILES
        },
    }
    generation_path = root / "generation_config.json"
    value["generation_config"] = (
        _normalized(_read_json(generation_path), _GENERATION_CONFIG_IGNORED)
        if generation_path.is_file() else None
    )
    return value

def _checkpoint_member(root: Path, name: str) -> Path:
    result = (root / name).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as exc:
        raise CheckpointError("checkpoint index references a path outside its directory") from exc
    return result

def _weight_map(root: Path) -> Dict[str, str]:
    index = root / "model.safetensors.index.json"
    single = root / "model.safetensors"
    if index.is_file():
        value = _read_json(index).get("weight_map")
        if not isinstance(value, dict) or not value:
            raise CheckpointError("invalid weight_map in {}".format(index))
        result = {str(key): str(shard) for key, shard in value.items()}
        for shard in result.values():
            member = _checkpoint_member(root, shard)
            if member.name != shard:
                raise CheckpointError("checkpoint shards must be direct files in the checkpoint directory")
        return result
    if single.is_file():
        try:
            with safe_open(str(single), framework="numpy") as handle:
                return {key: single.name for key in handle.keys()}
        except (OSError, SafetensorError) as exc:
            raise CheckpointError("cannot read safetensors header {}: {}".format(single, exc)) from exc
    raise CheckpointError("missing safetensors weights under {}".format(root))

def _schema(root: Path, mapping: Mapping[str, str]) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    result = {}
    for shard in sorted(set(mapping.values())):
        shard_path = _checkpoint_member(root, shard)
        if not shard_path.is_file():
            raise CheckpointError("missing shard {}".format(shard_path))
        try:
            with safe_open(str(shard_path), framework="numpy") as handle:
                shard_keys = set(handle.keys())
                expected = {key for key, filename in mapping.items() if filename == shard}
                if shard_keys != expected:
                    raise CheckpointError("index/header key mismatch in {}".format(shard_path))
                for key in sorted(shard_keys):
                    view = handle.get_slice(key)
                    result[key] = (tuple(view.get_shape()), str(view.get_dtype()))
        except CheckpointError:
            raise
        except (OSError, SafetensorError) as exc:
            raise CheckpointError("cannot read safetensors header {}: {}".format(shard_path, exc)) from exc
    return result

def _weight_files(root: Path) -> set:
    return {
        path.name
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name == "model.safetensors"
            or (path.name.startswith("model-") and path.name.endswith(".safetensors"))
        )
    }

def inspect_checkpoints(config: RunConfig) -> Dict[str, Any]:
    sources = [("base", config.base)] + [(item.id, item.path) for item in config.experts]
    if config.initial_model is not None:
        sources.append(("initial_model", config.initial_model))
    inspected = []
    reference_architecture = reference_schema = reference_metadata = None
    for source_id, root in sources:
        if not root.is_dir():
            raise CheckpointError("checkpoint directory does not exist: {}".format(root))
        architecture = _architecture(root, config.model_profile)
        mapping = _weight_map(root)
        expected_files = set(mapping.values())
        if _weight_files(root) != expected_files:
            raise CheckpointError("checkpoint contains unindexed or duplicate model weight files")
        schema = _schema(root, mapping)
        metadata = _metadata_identity(root)
        if reference_architecture is None:
            reference_architecture = architecture
            reference_schema = schema
            reference_metadata = metadata
        else:
            if architecture != reference_architecture:
                raise CheckpointError("architecture config mismatch at source {}".format(source_id))
            if schema != reference_schema:
                raise CheckpointError("tensor schema mismatch at source {}".format(source_id))
            if metadata != reference_metadata:
                raise CheckpointError("tokenizer/template metadata mismatch at source {}".format(source_id))
        inspected.append({
            "id": source_id,
            "path": str(root),
            "tensor_count": len(schema),
            "weight_file_count": len(expected_files),
            "metadata_files": [
                name
                for name in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "generation_config.json",
                    *_OPTIONAL_METADATA_FILES,
                )
                if (root / name).is_file()
            ],
        })

    assert reference_architecture is not None and reference_schema is not None
    keys = set(reference_schema)
    input_key = "model.embed_tokens.weight"
    output_key = "lm_head.weight"
    tied = reference_architecture["tie_word_embeddings"]
    if input_key not in keys:
        raise CheckpointError("decoder checkpoint is missing model.embed_tokens.weight")
    if tied and output_key in keys:
        raise CheckpointError(
            "tied checkpoint stores both edge keys; alias equality must be proven before deduplication"
        )
    if not tied and output_key not in keys:
        raise CheckpointError("untied checkpoint is missing lm_head.weight")
    edge = {
        "tie_word_embeddings": tied,
        "canonical_units": (
            [{"key": input_key, "roles": ["input_embedding", "output_head"]}]
            if tied else
            [
                {"key": input_key, "roles": ["input_embedding"]},
                {"key": output_key, "roles": ["output_head"]},
            ]
        ),
    }
    return {
        "status": "HEADER_PASS",
        "scope": "headers_and_runtime_metadata",
        "sources": inspected,
        "architecture": {
            key: reference_architecture[key]
            for key in _REQUIRED_ARCHITECTURE_KEYS
        },
        "tensor_count": len(reference_schema),
        "edge": edge,
        **(
            {"model_profile": config.model_profile.name}
            if config.model_profile is not None
            else {"model_profile": "auto"}
        ),
    }
