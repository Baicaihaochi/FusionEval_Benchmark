from itertools import islice
import gc

from ...data_runtime import iter_prompts, load_model, load_tokenizer
from ...data_ops import precision, write_stats
from .recompute import checkpoint_decoder

def accumulate_predictive(logits, parameters, accum, mode="exact", samples=1):
    import torch

    log_probs = logits.float().log_softmax(-1)
    probs = log_probs.detach().exp()
    if mode == "exact":
        classes = range(probs.numel())
    else:
        classes = torch.multinomial(probs, samples, replacement=True).tolist()
    for offset, category in enumerate(classes):
        gradients = torch.autograd.grad(
            log_probs[category], parameters, retain_graph=offset + 1 < len(classes),
            allow_unused=True,
        )
        factor = float(probs[category]) if mode == "exact" else 1.0 / samples
        for value, gradient in zip(accum, gradients):
            if gradient is not None:
                gradient = gradient.detach().to(device="cpu", dtype=torch.float32)
                value.addcmul_(gradient, gradient, value=factor)

def estimate(config, expert, root):
    import torch

    torch.manual_seed(int(config.runtime["seed"]))
    model = load_model(expert.path, config.runtime["device"], precision(config), gradients=True)
    model.eval()
    tied = bool(model.config.tie_word_embeddings)
    parameters = {
        name: value for name, value in model.named_parameters()
        if value.requires_grad and (tied or name != "lm_head.weight")
    }
    accum = {name: torch.zeros_like(value, device="cpu", dtype=torch.float32)
             for name, value in parameters.items()}
    tokenizer = load_tokenizer(config.base)
    examples = tokens = 0
    for prompt in islice(iter_prompts(tokenizer, expert.data, config.runtime["device"]),
                         config.method_parameters["examples"]):
        with checkpoint_decoder(model):
            hidden = model.model(input_ids=prompt.input_ids,
                                 attention_mask=prompt.attention_mask, use_cache=False).last_hidden_state
            logits = model.lm_head(hidden[0, -1])
            accumulate_predictive(logits, list(parameters.values()), list(accum.values()),
                                  config.method_parameters["estimator"], config.method_parameters["samples"])
        examples += 1
        tokens += prompt.tokens
        del hidden, logits
    if not examples:
        raise ValueError("Fisher consumed no prompts")
    for value in accum.values():
        value.div_(examples)
    write_stats(accum, root, config.runtime["max_shard_size_gib"])
    del accum, parameters, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"examples": examples, "tokens": tokens, "estimator": config.method_parameters["estimator"],
            "context": "last assistant generation prompt; pretokenized inputs without labels are prompt-only",
            "model_mode": "eval", "tied_embeddings": tied}
