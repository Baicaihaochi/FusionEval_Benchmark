"""mbridge adapter for GaviaV1 (infix_v1 / Moonlight-SWA) MoE models.

GaviaV1-130B-A8B = 30-layer GQA+SWA MoE with:
  - heterogeneous attention: full-attn layers (num_query_groups / kv_channels) and
    SWA layers (swa_num_query_groups / swa_kv_channels) with different KV heads + head_dim
  - learnable-softmax "sink" -> per-Q-head ``core_attention.softmax_offset`` parameter
  - QK RMSNorm (q_layernorm / k_layernorm, length = per-layer head_dim)
  - MoE: sigmoid router with per-expert bias buffer + one shared expert; layer 0 dense
  - HF stores routed experts GROUPED: moe.{gate,up,down}_proj.weight = [E, ffn, hidden]

The megatron model is built by slime's own model_provider from MODEL_ARGS; this bridge
is used ONLY for HF<->mcore weight mapping (convert_hf_to_torch_dist + RL weight sync).

HF weight names (per config.json / modeling_gavia_v1.py):
  model.embed_tokens.weight / model.norm.weight / lm_head.weight
  model.layers.{i}.input_layernorm.weight
  model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
  model.layers.{i}.self_attn.{q,k}_norm.weight
  model.layers.{i}.self_attn.softmax_offset            (learnable softmax)
  model.layers.{i}.post_attention_layernorm.weight
  dense (layer 0):  model.layers.0.mlp.{gate,up,down}_proj.weight
  moe   (1..29):    model.layers.{i}.moe.gate.weight        (router)
                    model.layers.{i}.moe.router_bias        (expert bias)
                    model.layers.{i}.moe.{gate,up,down}_proj.weight   [E, ffn, hidden] grouped
                    model.layers.{i}.share_expert.{gate,up,down}_proj.weight
"""
from functools import lru_cache

import torch

from mbridge.core import LLMBridge, register_model
from mbridge.core.safetensor_io import SafeTensorIO


class _CachingSafeTensorIO(SafeTensorIO):
    """Cache whole-tensor reads so 256 per-expert loads of the same grouped
    HF tensor (moe.{gate,up,down}_proj.weight, ~GBs) hit memory, not disk."""

    @lru_cache(maxsize=6)
    def _cached_one(self, name):  # noqa: D401
        return super().load_one_hf_weight(name)

    def load_one_hf_weight(self, hf_weight_name):
        return self._cached_one(hf_weight_name)


@register_model("gavia_v1")
class GaviaV1Bridge(LLMBridge):
    # GaviaV1Config omits a few knobs the base config-builder reads; give them defaults.
    _CONFIG_MAPPING = {
        **LLMBridge._CONFIG_MAPPING,
        "attention_dropout": ("attention_dropout", 0.0),
        "hidden_dropout": ("hidden_dropout", 0.0),
    }

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": ["model.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.linear_qkv.layer_norm_weight": ["model.layers.{layer_number}.input_layernorm.weight"],
        "self_attention.q_layernorm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
        "self_attention.k_layernorm.weight": ["model.layers.{layer_number}.self_attn.k_norm.weight"],
        "self_attention.core_attention.softmax_offset": ["model.layers.{layer_number}.self_attn.softmax_offset"],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
    }

    _MLP_MAPPING = {
        # dense layer (layer 0)
        "mlp.linear_fc1.layer_norm_weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
        # moe layers (1..29)
        "pre_mlp_layernorm.weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp.router.weight": ["model.layers.{layer_number}.moe.gate.weight"],
        "mlp.router.expert_bias": ["model.layers.{layer_number}.moe.router_bias"],
        "mlp.shared_experts.linear_fc1.weight": [
            "model.layers.{layer_number}.share_expert.gate_proj.weight",
            "model.layers.{layer_number}.share_expert.up_proj.weight",
        ],
        "mlp.shared_experts.linear_fc2.weight": ["model.layers.{layer_number}.share_expert.down_proj.weight"],
        "mlp.experts.linear_fc1": [
            "model.layers.{layer_number}.moe.gate_proj.weight",
            "model.layers.{layer_number}.moe.up_proj.weight",
        ],
        "mlp.experts.linear_fc2": ["model.layers.{layer_number}.moe.down_proj.weight"],
    }

    # ---- weight mapping ----
    def _get_safetensor_io(self, weights_path):
        return _CachingSafeTensorIO(self._get_actual_hf_path(weights_path))

    def _weight_name_mapping_mcore_local_to_global(self, model, consider_ep=True):
        ret = super()._weight_name_mapping_mcore_local_to_global(model, consider_ep=consider_ep)
        # drop non-persistent / runtime router stat buffers that have no HF counterpart
        return {k: v for k, v in ret.items() if "local_tokens_per_expert" not in k}

    @staticmethod
    def _expert_id(name):
        return int(name.rsplit("weight", 1)[1])

    def _merge_qkv_per_layer(self, hf_weights):
        """Fuse HF q/k/v_proj into mcore linear_qkv, per-group interleave [Q,K,V].
        Head config (kv heads, head_dim) is inferred from tensor shapes, so it is
        correct for both full-attn and SWA layers and independent of PP layer index.
        Assumes num_attention_heads == swa_num_attention_heads (true for gavia_v1)."""
        H = self.hf_config.num_attention_heads
        hidden = self.hf_config.hidden_size
        q, k, v = [w.to(self.dtype) for w in hf_weights]
        hd = q.shape[0] // H          # head_dim (128 full / 192 SWA)
        G = k.shape[0] // hd          # kv groups (4 full / 8 SWA)
        hpg = H // G
        q = q.view(G, hpg * hd, hidden)
        k = k.view(G, hd, hidden)
        v = v.view(G, hd, hidden)
        return torch.cat([q, k, v], dim=1).reshape(-1, hidden).contiguous()

    def _weight_to_mcore_format(self, mcore_weights_name, hf_weights):
        name = mcore_weights_name
        # routed experts: grouped HF [E, ffn, hidden] -> per-expert mcore
        if ".mlp.experts.linear_fc1" in name and "layer_norm" not in name:
            e = self._expert_id(name)
            gate, up = [w.to(self.dtype) for w in hf_weights]
            return torch.cat([gate[e], up[e]], dim=0).contiguous()
        if ".mlp.experts.linear_fc2" in name:
            e = self._expert_id(name)
            return hf_weights[0][e].to(self.dtype).contiguous()
        # heterogeneous qkv fusion
        if "self_attention.linear_qkv." in name and "layer_norm" not in name:
            return self._merge_qkv_per_layer(hf_weights)
        # keep fp32-sensitive scalars in source dtype (no bf16 downcast)
        if name.endswith("softmax_offset") or name.endswith("expert_bias"):
            return hf_weights[0]
        return super()._weight_to_mcore_format(name, hf_weights)

    # ---- config (only used if slime ever builds via bridge; harmless otherwise) ----
    def _build_config(self):
        hf = self.hf_config
        return self._build_base_config(
            num_moe_experts=hf.moe_num_experts,
            moe_ffn_hidden_size=hf.moe_intermediate_size,
            moe_shared_expert_intermediate_size=hf.share_expert_dim,
            moe_router_topk=hf.moe_top_k,
            moe_router_score_function=hf.moe_router_activation,
            moe_router_enable_expert_bias=hf.use_moe_router_bias,
            moe_router_topk_scaling_factor=hf.moe_router_scaling_factor,
            moe_grouped_gemm=True,
            qk_layernorm=hf.use_qk_norm,
            softmax_type=hf.softmax_type,
            kv_channels=hf.head_dim,
            moe_router_load_balancing_type="none",
        )
