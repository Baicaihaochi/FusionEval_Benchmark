from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

ATTN_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_LEAVES = ("gate_proj", "up_proj", "down_proj")

def selector_from_context(shared):
    return {
        "layer_prefix": shared.decoder_layer_prefix,
        "attention_modules": shared.attention_modules,
        "mlp_modules": shared.mlp_modules,
    }

def decoder_linear_kind(
    key: str,
    layer_prefix: str = "model.layers",
    attention_modules: Sequence[str] = ATTN_LEAVES,
    mlp_modules: Sequence[str] = MLP_LEAVES,
) -> Optional[str]:
    parts = key.split(".")
    prefix = layer_prefix.split(".")
    offset = len(prefix)
    if (
        len(parts) != offset + 4
        or parts[:offset] != prefix
        or not parts[offset].isdigit()
        or parts[offset + 3] != "weight"
    ):
        return None
    if parts[offset + 1] == "self_attn" and parts[offset + 2] in attention_modules:
        return "attn"
    if parts[offset + 1] == "mlp" and parts[offset + 2] in mlp_modules:
        return "mlp"
    return None

def is_decoder_linear(key: str, **selector) -> bool:
    return decoder_linear_kind(key, **selector) is not None

def count_decoder_linears(keys: Iterable[str], **selector) -> Tuple[int, int]:
    attn = mlp = 0
    for key in keys:
        kind = decoder_linear_kind(key, **selector)
        if kind == "attn":
            attn += 1
        elif kind == "mlp":
            mlp += 1
    return attn, mlp

def assert_decoder_linears(
    keys: Sequence[str],
    layer_count: int,
    layer_prefix: str = "model.layers",
    attention_modules: Sequence[str] = ATTN_LEAVES,
    mlp_modules: Sequence[str] = MLP_LEAVES,
) -> Tuple[int, int]:
    selector = {
        "layer_prefix": layer_prefix,
        "attention_modules": attention_modules,
        "mlp_modules": mlp_modules,
    }
    expected_leaves = {("self_attn", leaf) for leaf in attention_modules} | {
        ("mlp", leaf) for leaf in mlp_modules
    }
    prefix_size = len(layer_prefix.split("."))
    by_layer = {}
    unexpected = []
    for key in keys:
        if decoder_linear_kind(key, **selector) is None:
            unexpected.append(key)
            continue
        parts = key.split(".")
        by_layer.setdefault(int(parts[prefix_size]), set()).add(
            (parts[prefix_size + 1], parts[prefix_size + 2])
        )
    problems = []
    if unexpected:
        problems.append("non-linear keys selected: {}".format(unexpected[:8]))
    if set(by_layer) != set(range(layer_count)):
        problems.append(
            "layer indices {} != 0..{}".format(sorted(by_layer), layer_count - 1)
        )
    for index in range(layer_count):
        have = by_layer.get(index, set())
        if have != expected_leaves:
            problems.append(
                "layer {} missing {} extra {}".format(
                    index,
                    sorted(expected_leaves - have),
                    sorted(have - expected_leaves),
                )
            )
    attn, mlp = count_decoder_linears(keys, **selector)
    expected = (
        layer_count * len(attention_modules),
        layer_count * len(mlp_modules),
    )
    if (attn, mlp) != expected:
        problems.append(
            "counts attn={} mlp={} expected {} {}".format(attn, mlp, *expected)
        )
    if problems:
        raise RuntimeError(
            "decoder linear selection failed on scanned keys: "
            + "; ".join(problems[:8])
        )
    return attn, mlp

def optimize_matrix(
    key: str,
    ndim: int,
    edge: bool,
    exclude_edge: bool,
    decoder_linear_only: bool,
    **selector,
) -> bool:
    if ndim != 2:
        return False
    if decoder_linear_only:
        return is_decoder_linear(key, **selector)
    return not (exclude_edge and edge)
