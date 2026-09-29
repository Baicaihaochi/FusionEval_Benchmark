from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional

from .errors import ExecutionError

RNG_SCHEMA = "torch-generator-state-v1"
_HEX_STATE = re.compile(r"^[0-9a-f]+$")

RNG_DEVICE = {"continual_della": "device"}

TRAVERSAL = {
    "continual_della": (
        "two float32 Bernoulli draws per tensor in sorted tensor-key order, one for "
        "(previous - anchor) and one for (incoming - anchor), each with the kernel's "
        "row-rank survival probability"
    ),
}

def state_to_hex(state) -> str:

    return bytes(int(item) for item in state.reshape(-1).tolist()).hex()

def expected_generator_device(method: str, runtime_device: str) -> str:

    rule = RNG_DEVICE.get(method)
    if rule is None:
        raise KeyError("method {} has no continual RNG contract".format(method))
    return "cpu" if rule == "cpu" else str(runtime_device)

def hex_to_state(value: str):

    import torch

    return torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8)

def sha256_of_state(value: str) -> str:
    return hashlib.sha256(bytes.fromhex(value)).hexdigest()

def restore_generator_state(generator, value: str) -> None:
    generator.set_state(hex_to_state(value))

def encode_rng_state(
    generator,
    *,
    method: str,
    device: str,
    initial_seed: int,
    seeded_fresh: bool,
    input_state: Optional[str],
    traversal: str,
) -> Mapping[str, Any]:

    import torch

    state = state_to_hex(generator.get_state())
    return {
        "schema": RNG_SCHEMA,
        "method": method,
        "generator_device": str(device),
        "state_hex": state,
        "state_bytes": len(state) // 2,
        "state_sha256": sha256_of_state(state),
        "initial_seed": int(initial_seed),
        "seeded_fresh": bool(seeded_fresh),
        "input_state_sha256": (
            sha256_of_state(input_state) if input_state is not None else None
        ),
        "traversal": traversal,
        "torch_version": str(torch.__version__),
        "cuda_version": (
            str(torch.version.cuda) if torch.version.cuda is not None else None
        ),
    }

def read_stage_rng(
    manifest_path: Path,
    *,
    method: str,
    expected_device: str,
    stage: int,
) -> Mapping[str, Any]:

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionError(
            "cannot read stage manifest {} for RNG continuation: {}".format(
                manifest_path, exc
            )
        ) from exc
    diagnostics = manifest.get("diagnostics") if isinstance(manifest, Mapping) else None
    payload = diagnostics.get("rng") if isinstance(diagnostics, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ExecutionError(
            "stage {} has no persisted RNG state; the stream cannot be continued: "
            "{}".format(stage, manifest_path)
        )
    if payload.get("schema") != RNG_SCHEMA:
        raise ExecutionError(
            "stage {} RNG state uses an unknown schema: {}".format(
                stage, payload.get("schema")
            )
        )
    if payload.get("method") != method:
        raise ExecutionError(
            "stage {} RNG state belongs to {}, not {}".format(
                stage, payload.get("method"), method
            )
        )
    if payload.get("generator_device") != expected_device:
        raise ExecutionError(
            "stage {} RNG state was produced on device {}, but this run requires {}; a "
            "different device would change the stream".format(
                stage, payload.get("generator_device"), expected_device
            )
        )
    state = payload.get("state_hex")
    if not isinstance(state, str) or not state or not _HEX_STATE.fullmatch(state):
        raise ExecutionError("stage {} RNG state is not a hex blob".format(stage))
    if len(state) % 2:
        raise ExecutionError("stage {} RNG state has an odd byte length".format(stage))
    if payload.get("state_sha256") != sha256_of_state(state):
        raise ExecutionError("stage {} RNG state hash mismatch".format(stage))
    if payload.get("state_bytes") != len(state) // 2:
        raise ExecutionError("stage {} RNG state length mismatch".format(stage))
    return payload
