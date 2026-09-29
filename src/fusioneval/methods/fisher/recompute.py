from contextlib import contextmanager
from functools import partial

@contextmanager
def checkpoint_decoder(model):

    from torch.utils.checkpoint import checkpoint

    if model.training:
        raise ValueError("Predictive Fisher recomputation requires eval mode")
    originals = []
    try:
        for layer in model.model.layers:
            originals.append((layer, layer.__dict__.get("forward")))
            layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        yield
    finally:
        for layer, original in reversed(originals):
            if original is None:
                del layer.forward
            else:
                layer.forward = original
