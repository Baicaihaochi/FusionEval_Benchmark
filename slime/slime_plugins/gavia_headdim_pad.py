"""Workaround for a cuDNN 9.21.1 bug: the THD (packed / `arbitrary_seqlen`) fused
attention BACKWARD kernel raises err700 (CUDA_ERROR_ILLEGAL_ADDRESS) for NON-power-of-2
head_dim (160/192/224). GaviaV1 SWA layers use swa_kv_channels=192 -> they crash in
SFT/RL (THD). Pretrain never hit it (BSHD path, which is fine at hd192).

Fix: pad q/k/v head_dim 192->256 (a working pow2) right around the fused core-attention.
Numerically EXACT: the zero-padded head dims contribute 0 to Q·K^T and 0 to the output;
TE's stored `softmax_scale = 1/sqrt(192)` (computed at __init__, NOT from the padded input)
is unchanged, so no rescale is needed. Verified: BSHD pad256 == BSHD-192 bit-exact;
THD pad256 == BSHD-192 to bf16 precision (~0.5%, the normal THD-vs-BSHD kernel diff).

Only the TE head_dim assertion needs relaxing (set hidden_size_per_attention_head_{k,v}).
Auto-noops for any layer whose head_dim is not a broken non-pow2 value.
"""
import torch.nn.functional as F

_BROKEN_HEAD_DIMS = {160, 192, 224}  # non-pow2 dims cuDNN 9.21 THD-bwd mishandles
_PAD_TO = 256


def _patch_core_attention(ca):
    d = getattr(ca, "hidden_size_per_attention_head_k", None)
    if d not in _BROKEN_HEAD_DIMS or getattr(ca, "_gavia_headdim_padded", False):
        return False
    orig_forward = ca.forward
    pad = _PAD_TO - d
    ca.hidden_size_per_attention_head_k = _PAD_TO  # relax TE's head_dim assert
    ca.hidden_size_per_attention_head_v = _PAD_TO

    def padded_forward(q, k, v, *args, **kwargs):
        qp = F.pad(q, (0, pad)); kp = F.pad(k, (0, pad)); vp = F.pad(v, (0, pad))
        out = orig_forward(qp, kp, vp, *args, **kwargs)   # [..., heads * _PAD_TO]
        heads = q.shape[-2]
        lead = out.shape[:-1]
        return out.view(*lead, heads, _PAD_TO)[..., :d].reshape(*lead, heads * d)

    ca.forward = padded_forward
    ca._gavia_headdim_padded = True
    return True


def pad_swa_head_dim(model):
    """Patch every core_attention with a broken non-pow2 head_dim in `model`.
    Safe no-op for models without such heads. Returns #layers patched."""
    decoder = getattr(model, "decoder", model)
    n = 0
    for layer in getattr(decoder, "layers", []) or []:
        self_attn = getattr(layer, "self_attention", None)
        ca = getattr(self_attn, "core_attention", None)
        if ca is not None and _patch_core_attention(ca):
            n += 1
    if n:
        try:
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[gavia] head_dim pad workaround: patched {n} non-pow2 core_attention(s) -> {_PAD_TO}", flush=True)
        except Exception:
            pass
    return n
