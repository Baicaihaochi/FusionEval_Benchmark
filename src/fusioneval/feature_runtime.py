from __future__ import annotations

from .data_runtime import model_layers

class _LayerComplete(Exception):
    def __init__(self, hidden):
        self.hidden = hidden

def hidden_chunks(model, sample, chunk_size=256, stop_layer=None):

    import torch
    from transformers.cache_utils import DynamicCache

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if model.training:
        raise ValueError("feature collection requires model.eval()")
    cache = DynamicCache()
    handle = None
    if stop_layer is not None:
        layer = model_layers(model)[stop_layer]

        def stop(_module, _inputs, output):
            raise _LayerComplete(output[0] if isinstance(output, tuple) else output)

        handle = layer.register_forward_hook(stop)
    try:
        for start in range(0, sample.tokens, chunk_size):
            end = min(start + chunk_size, sample.tokens)
            with torch.no_grad():
                try:
                    out = model.model(
                        input_ids=sample.input_ids[:, start:end],
                        attention_mask=sample.attention_mask[:, :end],
                        past_key_values=cache,
                        use_cache=True,
                    )
                    hidden = out.last_hidden_state
                    if stop_layer is not None:
                        raise RuntimeError("requested decoder layer was not executed")
                except _LayerComplete as reached:
                    hidden = reached.hidden
            yield hidden.detach()
    finally:
        if handle is not None:
            handle.remove()
        del cache
