from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .data_runtime import iter_examples, linear_modules

def gram_key(module_name: str) -> str:
    parent, _, leaf = module_name.rpartition(".")
    if leaf in {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}:
        return parent + ".__input__"
    return module_name

def collect_grams(model: Any, tokenizer: Any, data: Path, device: str, examples: int):
    import torch

    modules = linear_modules(model)
    representatives: Dict[str, Any] = {}
    module_to_gram = {}
    for name, module in modules.items():
        if name == "lm_head":
            continue
        key = gram_key(name)
        module_to_gram[name] = key
        representatives.setdefault(key, module)
    grams: Dict[str, Any] = {}
    counts: Dict[str, int] = {}

    def capture(key):
        def hook(_module, inputs, _output):
            value = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            current = value.transpose(0, 1) @ value
            rows = value.shape[0]
            if key not in grams:
                grams[key], counts[key] = current / rows, rows
            else:
                total = counts[key] + rows
                grams[key].mul_(counts[key] / total).add_(current, alpha=1.0 / total)
                counts[key] = total
        return hook

    handles = [module.register_forward_hook(capture(key)) for key, module in representatives.items()]
    seen = tokens = 0
    try:
        with torch.inference_mode():
            for sample in iter_examples(tokenizer, data, device, examples):
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
    return (
        {key: value.cpu() for key, value in grams.items()},
        {"examples": seen, "tokens": tokens, "gram_rows": counts},
        module_to_gram,
    )

def shrink(gram, alpha: float):
    diagonal = gram.diagonal().clone()
    gram = gram.mul(alpha)
    gram.diagonal().copy_(diagonal)
    return gram
