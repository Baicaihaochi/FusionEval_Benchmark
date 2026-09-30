"""MCore -> HuggingFace weight conversion for GaviaV1-130B-A8B (RL weight sync, raw mode).

Inverse of slime_plugins/mbridge/gavia_v1.py's HF->mcore mapping. Emits the exact HF
names sglang's GaviaV1ForCausalLM.load_weights expects:
  - attention: q/k/v_proj (split from mcore linear_qkv; heterogeneous full-attn hd128/qg4
    vs SWA hd192/qg8, inferred from tensor shape), o_proj, q/k_norm, softmax_offset
  - dense layer 0: mlp.{gate,up,down}_proj
  - MoE layers 1..29: moe.gate.weight (router), moe.router_bias, share_expert.{gate,up,down}_proj,
    and GROUPED routed experts moe.{gate,up,down}_proj.weight = [E, ffn, hidden].

sglang loads routed experts GROUPED (loaded_weight[expert_id]); mcore streams them per-expert
(linear_fc{1,2}.weight{N}) under grouped-gemm. We accumulate per-expert slices in a module cache
and emit the stacked [E, ...] tensor once all args.num_experts slices for a (layer, proj) arrive.
The EP broadcast happens before convert_to_hf (see UpdateWeightFromTensor docstring), so every
converting rank sees the full expert set.
"""
import re

import torch

# (layer_idx, "gate"|"up"|"down") -> {expert_id: tensor}; flushed to grouped when full.
_expert_cache: dict = {}


def _strip(name: str) -> str:
    for p in ("module.module.", "module."):
        if name.startswith(p):
            return name[len(p) :]
    return name


def _qkv_head_cfg(args, out_rows: int):
    """Return (kv_groups, head_dim, heads_per_group) for this layer, inferred from the
    mcore linear_qkv row count: out_rows == head_dim * (num_attention_heads + 2*kv_groups)."""
    H = args.num_attention_heads
    candidates = [
        (args.num_query_groups, args.kv_channels),
        (getattr(args, "swa_num_query_groups", None), getattr(args, "swa_kv_channels", None)),
    ]
    for G, hd in candidates:
        if G and hd and out_rows == hd * (H + 2 * G):
            return G, hd, H // G
    raise ValueError(f"cannot infer gavia qkv head config from out_rows={out_rows}, H={H}")


def convert_gavia_to_hf(args, name, param):
    name = _strip(name)

    if name.endswith("embedding.word_embeddings.weight"):
        return [("model.embed_tokens.weight", param)]
    if name.endswith("output_layer.weight"):
        return [("lm_head.weight", param)]
    if name.endswith("decoder.final_layernorm.weight"):
        return [("model.norm.weight", param)]

    m = re.search(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not m:
        raise ValueError(f"Unknown gavia parameter name: {name}")
    layer_idx, rest = int(m.group(1)), m.group(2)
    L = f"model.layers.{layer_idx}"

    # ---- routed experts: per-expert -> grouped [E, ...] ----
    em = re.match(r"mlp\.experts\.(linear_fc1|linear_fc2)\.weight(\d+)$", rest)
    if em:
        proj, e = em.group(1), int(em.group(2))
        E = args.num_experts

        def _accum(key, tensor, hf_name):
            # Copy each expert into a pre-allocated grouped [E, ...] buffer (copy-in-place)
            # instead of holding all E per-expert tensors + torch.stack -- that doubled the
            # transient (~2x per proj) and OOMd during colocate update (sglang resident + megatron).
            entry = _expert_cache.get((layer_idx, key))
            if entry is None:
                grouped = torch.empty((E, *tensor.shape), dtype=tensor.dtype, device=tensor.device)
                entry = [grouped, 0]
                _expert_cache[(layer_idx, key)] = entry
            grouped, cnt = entry
            grouped[e].copy_(tensor)
            entry[1] = cnt + 1
            if entry[1] == E:
                del _expert_cache[(layer_idx, key)]
                return [(hf_name, grouped)]
            return []

        if proj == "linear_fc1":
            gate, up = param.chunk(2, dim=0)  # each [ffn, hidden]
            out = _accum("gate", gate, f"{L}.moe.gate_proj.weight")
            out += _accum("up", up, f"{L}.moe.up_proj.weight")
            return out
        return _accum("down", param, f"{L}.moe.down_proj.weight")  # [hidden, ffn]

    # ---- shared expert ----
    if rest == "mlp.shared_experts.linear_fc1.weight":
        gate, up = param.chunk(2, dim=0)
        return [
            (f"{L}.share_expert.gate_proj.weight", gate),
            (f"{L}.share_expert.up_proj.weight", up),
        ]
    if rest == "mlp.shared_experts.linear_fc2.weight":
        return [(f"{L}.share_expert.down_proj.weight", param)]

    # ---- attention ----
    if rest == "self_attention.linear_proj.weight":
        return [(f"{L}.self_attn.o_proj.weight", param)]
    if rest == "self_attention.linear_qkv.layer_norm_weight":
        return [(f"{L}.input_layernorm.weight", param)]
    if rest == "self_attention.q_layernorm.weight":
        return [(f"{L}.self_attn.q_norm.weight", param)]
    if rest == "self_attention.k_layernorm.weight":
        return [(f"{L}.self_attn.k_norm.weight", param)]
    if rest == "self_attention.core_attention.softmax_offset":
        return [(f"{L}.self_attn.softmax_offset", param)]
    if rest == "self_attention.linear_qkv.weight":
        hidden = param.shape[1]
        G, hd, hpg = _qkv_head_cfg(args, param.shape[0])
        qkv = param.view(G, hpg + 2, hd, hidden)
        q, k, v = torch.split(qkv, [hpg, 1, 1], dim=1)
        return [
            (f"{L}.self_attn.q_proj.weight", q.reshape(-1, hidden).contiguous()),
            (f"{L}.self_attn.k_proj.weight", k.reshape(-1, hidden).contiguous()),
            (f"{L}.self_attn.v_proj.weight", v.reshape(-1, hidden).contiguous()),
        ]

    # ---- MoE router / layernorm ----
    if rest == "mlp.router.weight":
        return [(f"{L}.moe.gate.weight", param)]
    if rest == "mlp.router.expert_bias":
        return [(f"{L}.moe.router_bias", param)]
    if rest == "pre_mlp_layernorm.weight":
        return [(f"{L}.post_attention_layernorm.weight", param)]

    # ---- dense layer 0 MLP ----
    if rest == "mlp.linear_fc1.layer_norm_weight":
        return [(f"{L}.post_attention_layernorm.weight", param)]
    if rest == "mlp.linear_fc1.weight":
        gate, up = param.chunk(2, dim=0)
        return [
            (f"{L}.mlp.gate_proj.weight", gate),
            (f"{L}.mlp.up_proj.weight", up),
        ]
    if rest == "mlp.linear_fc2.weight":
        return [(f"{L}.mlp.down_proj.weight", param)]

    raise ValueError(f"Unknown gavia parameter name: {name}")
