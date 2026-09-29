from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence

from .errors import CheckpointError

MAX_SEQUENCE_TOKENS = 4_096

FEATURE_TOKEN_LIMIT = 32_768

@dataclass(frozen=True)
class Example:
    input_ids: Any
    attention_mask: Any
    labels: Any
    tokens: int
    loss_tokens: int

@dataclass(frozen=True)
class Prompt:
    input_ids: Any
    attention_mask: Any
    tokens: int

def _role(value: Any) -> str:
    role = str(value).lower()
    return {
        "human": "user",
        "gpt": "assistant",
        "bot": "assistant",
        "function": "tool",
    }.get(role, role)

def _messages(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    source = row.get("messages", row.get("conversations", row.get("conversation")))
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        result = []
        for item in source:
            if not isinstance(item, Mapping):
                raise ValueError("conversation entries must be mappings")
            role = _role(item.get("role", item.get("from", "")))
            content = item.get("content", item.get("value", ""))
            message = {"role": role, "content": content}
            for key in ("tool_calls", "tool_call_id", "name", "reasoning_content"):
                if key in item:
                    message[key] = item[key]
            result.append(message)
        return result

    prompt = row.get("prompt", row.get("instruction", row.get("input")))
    answer = row.get("response", row.get("output", row.get("answer")))
    if prompt is not None and answer is not None:
        return [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(answer)},
        ]
    raise ValueError("row has neither a supported conversation nor prompt/answer fields")

def _template_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    add_generation_prompt: bool = False,
    tools: Any = None,
) -> List[int]:
    arguments = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        arguments["tools"] = tools
    value = tokenizer.apply_chat_template(list(messages), **arguments)
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)

def _assistant_mask(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    ids: List[int],
    tools: Any = None,
) -> List[bool]:
    try:
        arguments = {
            "tokenize": True,
            "add_generation_prompt": False,
            "return_dict": True,
            "return_assistant_tokens_mask": True,
        }
        if tools is not None:
            arguments["tools"] = tools
        encoded = tokenizer.apply_chat_template(list(messages), **arguments)
        mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
        if mask and isinstance(mask[0], list):
            mask = mask[0]
        if mask is not None and len(mask) == len(ids) and any(mask):
            return [bool(item) for item in mask]
    except (TypeError, ValueError, KeyError):
        pass

    mask = [False] * len(ids)
    previous = 0
    for index, message in enumerate(messages):
        current = min(
            len(ids),
            len(_template_ids(tokenizer, messages[: index + 1], tools=tools)),
        )
        if message.get("role") == "assistant":
            for position in range(previous, current):
                mask[position] = True
        previous = current
    return mask

def encode_row(tokenizer: Any, row: Mapping[str, Any], device: str) -> Example:
    import torch

    if isinstance(row.get("input_ids"), list):
        ids = [int(item) for item in row["input_ids"]]
        raw_labels = row.get("labels")
        labels = [int(item) for item in raw_labels] if isinstance(raw_labels, list) else list(ids)
    else:
        messages = _messages(row)
        tools = row.get("tools")
        ids = _template_ids(tokenizer, messages, tools=tools)
        assistant = _assistant_mask(tokenizer, messages, ids, tools=tools)
        labels = [token if keep else -100 for token, keep in zip(ids, assistant)]
        if not any(item != -100 for item in labels):
            labels = list(ids)
    if not ids:
        raise ValueError("sequence must not be empty")
    if len(labels) != len(ids):
        raise ValueError("labels and input_ids differ in length")
    if len(ids) > MAX_SEQUENCE_TOKENS:
        ids = ids[:MAX_SEQUENCE_TOKENS]
        labels = labels[:MAX_SEQUENCE_TOKENS]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)

    loss_tokens = sum(item != -100 for item in labels[1:])
    return Example(
        input_ids,
        attention_mask,
        label_tensor,
        len(ids),
        loss_tokens,
    )

def iter_examples(
    tokenizer: Any, path: Path, device: str, limit: int
) -> Iterator[Example]:
    accepted = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if accepted >= limit:
                break
            try:
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError("row must be a JSON object")
                example = encode_row(tokenizer, row, device)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise CheckpointError(
                    "invalid calibration row {}:{}: {}".format(path, line_number, exc)
                ) from exc
            accepted += 1
            yield example
    if accepted == 0:
        raise CheckpointError("calibration data is empty: {}".format(path))

def encode_prompt_row(tokenizer: Any, row: Mapping[str, Any], device: str) -> Prompt:
    import torch

    if isinstance(row.get("input_ids"), list):
        ids = [int(item) for item in row["input_ids"]]
        raw_labels = row.get("labels")
        if isinstance(raw_labels, list):
            if len(raw_labels) != len(ids):
                raise ValueError("labels and input_ids differ in length")
            supervised = [
                index for index, value in enumerate(raw_labels) if int(value) != -100
            ]
            if supervised:

                ids = ids[: supervised[0]]
    else:
        messages = _messages(row)
        tools = row.get("tools")
        assistant_indices = [
            index for index, message in enumerate(messages) if message.get("role") == "assistant"
        ]
        prefix = messages[: assistant_indices[-1]] if assistant_indices else messages
        ids = _template_ids(
            tokenizer, prefix, add_generation_prompt=True, tools=tools
        )
    if not ids:
        raise ValueError("prompt must not be empty")
    if len(ids) > MAX_SEQUENCE_TOKENS:
        ids = ids[:MAX_SEQUENCE_TOKENS]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    return Prompt(input_ids, torch.ones_like(input_ids), len(ids))

def iter_prompts(tokenizer: Any, path: Path, device: str) -> Iterator[Prompt]:
    accepted = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError("row must be a JSON object")
                prompt = encode_prompt_row(tokenizer, row, device)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise CheckpointError(
                    "invalid calibration row {}:{}: {}".format(path, line_number, exc)
                ) from exc
            accepted += 1
            yield prompt
    if accepted == 0:
        raise CheckpointError("calibration data is empty: {}".format(path))

def iter_features(tokenizer: Any, path: Path, device: str, limit: int) -> Iterator[Prompt]:

    import torch

    seen = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if seen >= limit:
                break
            try:
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError("row must be a JSON object")
                if isinstance(row.get("input_ids"), list):
                    ids = [int(x) for x in row["input_ids"]]
                    mask = row.get("attention_mask", [1] * len(ids))
                    if len(mask) != len(ids) or any(x not in (0, 1) for x in mask):
                        raise ValueError("invalid attention_mask")
                    ids = [x for x, keep in zip(ids, mask) if keep]
                elif isinstance(row.get("text"), str):
                    ids = list(tokenizer.encode(row["text"]))
                else:
                    ids = _template_ids(tokenizer, _messages(row), tools=row.get("tools"))
                if not ids or len(ids) > FEATURE_TOKEN_LIMIT:
                    raise ValueError("feature sequence must contain 1..32768 tokens; no truncation")
                if min(ids) < 0:
                    raise ValueError("input token IDs must be nonnegative")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise CheckpointError(
                    "invalid calibration row {}:{}: {}".format(path, line_number, exc)
                ) from exc
            seen += 1
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            yield Prompt(input_ids, torch.ones_like(input_ids), len(ids))
    if not seen:
        raise CheckpointError("calibration data is empty: {}".format(path))


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=False)

def load_model(path: Path, device: str, precision: str, gradients: bool = False):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if precision == "mergebench" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=False
    )
    model.config.use_cache = False
    model.to(device)
    if gradients:
        model.train()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    else:
        model.eval()
        model.requires_grad_(False)
    return model


def model_layers(model: Any):
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise CheckpointError("data methods require a Qwen/Llama-style model.layers stack")
    return layers

def linear_modules(module: Any):
    import torch

    return {
        name: item
        for name, item in module.named_modules()
        if name and isinstance(item, torch.nn.Linear)
    }

def data_inventory(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    result = []
    for path in paths:
        if not path.is_file():
            raise CheckpointError("calibration data does not exist: {}".format(path))
        rows = 0
        with path.open("rb") as stream:
            for _ in stream:
                rows += 1
        result.append({
            "path": str(path),
            "rows": rows,
            "bytes": path.stat().st_size,
        })
    return result
