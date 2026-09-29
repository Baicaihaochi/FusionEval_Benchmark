from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

def correct_hidden(hidden, down_weight, up_weight):

    import torch
    import torch.nn.functional as functional

    with torch.autocast(device_type=hidden.device.type, enabled=False):
        value = hidden.float()
        correction = functional.linear(
            functional.relu(functional.linear(value, down_weight.float())), up_weight.float()
        )
        return (value - correction).to(hidden.dtype)

@contextmanager
def surgery_context(model, checkpoint: str | Path, expert_id: str):

    import torch
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    config = json.loads(
        (checkpoint / "surgery" / "surgery_config.json").read_text(encoding="utf-8")
    )
    if config.get("schema_version") != 1 or config.get("operation") != "hidden - up(relu(down(hidden)))":
        raise ValueError("unsupported Surgery adapter schema or operation")
    if config.get("hidden_size") != model.config.hidden_size:
        raise ValueError("Surgery hidden size does not match the loaded model")
    if config.get("model_type", model.config.model_type) != model.config.model_type:
        raise ValueError("Surgery model family does not match the loaded model")
    norm = getattr(getattr(model, "model", None), "norm", None)
    if norm is None:
        raise TypeError("Surgery runtime requires model.model.norm")
    if getattr(norm, "_fusioneval_surgery_active", False):
        raise RuntimeError("Surgery routes must not overlap on the same model")
    if expert_id not in config["experts"]:
        raise KeyError("unknown Surgery expert id: {}".format(expert_id))
    path = checkpoint / "surgery" / "surgery_adapters.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        down_weight = handle.get_tensor("experts.{}.down.weight".format(expert_id))
        up_weight = handle.get_tensor("experts.{}.up.weight".format(expert_id))
    if down_weight.shape != (config["rank"], config["hidden_size"]) or up_weight.shape != (config["hidden_size"], config["rank"]):
        raise ValueError("invalid Surgery adapter tensor shapes")
    if not bool(torch.isfinite(down_weight).all() and torch.isfinite(up_weight).all()):
        raise ValueError("non-finite Surgery adapter tensors")
    parameter = next(norm.parameters())
    down_weight = down_weight.to(device=parameter.device, dtype=torch.float32)
    up_weight = up_weight.to(device=parameter.device, dtype=torch.float32)

    def correct(_module, _inputs, output):
        return correct_hidden(output, down_weight, up_weight)

    handle = norm.register_forward_hook(correct)
    norm._fusioneval_surgery_active = True
    try:
        yield model
    finally:
        handle.remove()
        del norm._fusioneval_surgery_active
