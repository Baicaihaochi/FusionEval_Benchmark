from contextlib import closing
from functools import partial

def layer_chunks(model, layer, hidden, chunk_size):
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    device = next(layer.parameters()).device
    cache = DynamicCache()

    attention = layer.self_attn
    old_index = attention.layer_idx
    attention.layer_idx = 0
    try:
        for start in range(0, hidden.shape[1], chunk_size):
            x = hidden[:, start:start + chunk_size].to(device)
            positions = torch.arange(start, start + x.shape[1], device=device)
            ids = positions.unsqueeze(0)
            make_mask = (create_sliding_window_causal_mask
                         if getattr(layer, "attention_type", None) == "sliding_attention"
                         else create_causal_mask)

            mask = make_mask(model.config, x, None, positions,
                             past_key_values=cache, position_ids=ids)
            out = layer(x, attention_mask=mask, position_ids=ids,
                        past_key_values=cache, use_cache=True, cache_position=positions,
                        position_embeddings=model.model.rotary_emb(x, ids))
            yield (out[0] if isinstance(out, tuple) else out).detach().cpu()
    finally:
        attention.layer_idx = old_index
        del cache

def advance(model, layer, inputs, chunk_size):
    import torch
    with torch.inference_mode():
        return [torch.cat(list(layer_chunks(model, layer, x, chunk_size)), dim=1) for x in inputs]

def collect(model, student_layer, expert_layer, student_inputs, expert_inputs, alpha, chunk_size):
    import torch
    from .runner import _groups

    sm, groups = _groups(student_layer)
    em, eg = _groups(expert_layer)
    if groups != eg or len(student_inputs) != len(expert_inputs):
        raise ValueError("cached student/expert layouts or sample counts differ")
    sc, ec, grams, crosses, rows = {}, {}, {}, {}, {}
    handles = []
    def capture(cache, group, module, inputs):
        cache[group] = inputs[0].detach()
    for group, names in groups.items():
        handles.extend([sm[names[0]].register_forward_pre_hook(partial(capture, sc, group)),
                        em[names[0]].register_forward_pre_hook(partial(capture, ec, group))])
    next_expert = []
    try:
        with torch.inference_mode():
            for xs0, xe0 in zip(student_inputs, expert_inputs):
                if xs0.shape != xe0.shape:
                    raise ValueError("cached feature shapes differ")
                outputs = []
                with closing(layer_chunks(model, student_layer, xs0, chunk_size)) as sg, closing(layer_chunks(model, expert_layer, xe0, chunk_size)) as eg:
                    for _ in range(0, xs0.shape[1], chunk_size):
                        sc.clear(); ec.clear()
                        next(sg); outputs.append(next(eg))
                        for group in groups:
                            xs = sc[group].reshape(-1, sc[group].shape[-1]).float()
                            xe = ec[group].reshape_as(xs).float()
                            g, c = xs.T @ xs, xs.T @ (alpha * xe + (1 - alpha) * xs)
                            if group not in grams:
                                grams[group], crosses[group], rows[group] = g, c, 0
                            else:
                                grams[group].add_(g); crosses[group].add_(c)
                            rows[group] += xs.shape[0]
                next_expert.append(torch.cat(outputs, dim=1))
    finally:
        for handle in handles:
            handle.remove()
    return ({g: v.div(rows[g]).cpu() for g, v in grams.items()},
            {g: v.div(rows[g]).cpu() for g, v in crosses.items()}, groups,
            {"examples": len(student_inputs), "tokens": sum(x.shape[1] for x in student_inputs),
             "feature_rows": rows, "prefill_chunk_size": chunk_size}, next_expert)
