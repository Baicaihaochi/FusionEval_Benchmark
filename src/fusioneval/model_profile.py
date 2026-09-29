from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import yaml

from .errors import ConfigError

@dataclass(frozen=True)
class ModelProfile:
    name: str
    path: Path
    architecture: Dict[str, Any]
    layer_prefix: str
    attention_modules: Tuple[str, ...]
    mlp_modules: Tuple[str, ...]

    @property
    def num_hidden_layers(self) -> int:
        return int(self.architecture["num_hidden_layers"])

def load_model_profile(path: Path) -> ModelProfile:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError("cannot read model profile {}: {}".format(path, exc)) from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ConfigError("model profile schema_version must be 1")
    allowed = {"schema_version", "name", "architecture", "decoder"}
    extra = set(value) - allowed
    if extra:
        raise ConfigError("unsupported model profile key(s): {}".format(", ".join(sorted(extra))))
    name = value.get("name")
    architecture = value.get("architecture")
    decoder = value.get("decoder")
    if not isinstance(name, str) or not name:
        raise ConfigError("model profile name must be a non-empty string")
    if not isinstance(architecture, Mapping) or not architecture:
        raise ConfigError("model profile architecture must be a non-empty mapping")
    if not isinstance(decoder, Mapping):
        raise ConfigError("model profile decoder must be a mapping")
    required = {
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "tie_word_embeddings",
    }
    missing = required - set(architecture)
    if missing:
        raise ConfigError(
            "model profile architecture is missing: {}".format(", ".join(sorted(missing)))
        )
    if architecture["model_type"] not in {"qwen3", "llama"}:
        raise ConfigError("model profile supports model_type qwen3 or llama")
    for key in required - {"model_type", "tie_word_embeddings"}:
        item = architecture[key]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ConfigError("model profile architecture.{} must be a positive integer".format(key))
    if not isinstance(architecture["tie_word_embeddings"], bool):
        raise ConfigError("model profile tie_word_embeddings must be boolean")
    prefix = decoder.get("layer_prefix")
    attention = decoder.get("attention_modules")
    mlp = decoder.get("mlp_modules")
    if not isinstance(prefix, str) or not prefix:
        raise ConfigError("model profile decoder.layer_prefix must be a string")
    if (
        not isinstance(attention, list)
        or not attention
        or not all(isinstance(item, str) and item for item in attention)
        or len(set(attention)) != len(attention)
    ):
        raise ConfigError("model profile decoder.attention_modules must be a unique string list")
    if (
        not isinstance(mlp, list)
        or not mlp
        or not all(isinstance(item, str) and item for item in mlp)
        or len(set(mlp)) != len(mlp)
    ):
        raise ConfigError("model profile decoder.mlp_modules must be a unique string list")
    return ModelProfile(
        name,
        path,
        dict(architecture),
        prefix,
        tuple(attention),
        tuple(mlp),
    )
