import aiohttp
import torch

from slime.rollout.balmopd import _compute_balmopd_weights
from slime.utils.metric_utils import has_repetition
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

NON_THINKING = "non_thinking"


def _response_mask_reason(args, sample: Sample) -> str | None:
    """Return the configured whole-response mask reason, if any.

    ``truncated`` preserves the historical fail-closed policy. ``repetition``
    is an ablation that ignores the finish reason and masks only responses
    selected by slime's existing compression-based repetition detector.
    """
    mode = getattr(args, "opd_response_mask_mode", "truncated")
    if mode == "truncated":
        return "truncated" if sample.status == Sample.Status.TRUNCATED else None
    if mode == "repetition":
        return "repetition" if has_repetition(sample.response) else None
    raise ValueError(f"Unsupported OPD response mask mode: {mode}")


def _mask_response(sample: Sample, reason: str) -> None:
    sample.loss_mask = [0] * sample.response_length
    sample.teacher_log_probs = torch.zeros(sample.response_length, dtype=torch.float32)
    sample.remove_sample = True
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["opd_response_mask_reason"] = reason
    if reason == "truncated":
        sample.metadata["opd_skipped_truncated"] = True
    elif reason == "repetition":
        sample.metadata["opd_skipped_repetition"] = True


def _canonical_teacher_input_ids(sample: Sample) -> tuple[list[int], bool]:
    """Replace the student's non-thinking prompt with the Instruct prompt."""
    metadata = sample.metadata or {}
    expected_student_prompt = list(metadata.get("mopd_student_prompt_ids") or [])
    teacher_prompt = list(metadata.get("mopd_teacher_prompt_ids") or [])
    if not expected_student_prompt or not teacher_prompt:
        raise ValueError("Sample is missing canonical single-teacher prompt token metadata.")
    if sample.response_length < 0 or sample.response_length > len(sample.tokens):
        raise ValueError("Invalid response_length for canonical single-teacher OPD sample.")
    prompt_len = len(sample.tokens) - sample.response_length
    actual_student_prompt = list(sample.tokens[:prompt_len])
    if actual_student_prompt != expected_student_prompt:
        raise ValueError("Student prompt tokens changed after canonical non-thinking rendering.")
    if actual_student_prompt[: len(teacher_prompt)] != teacher_prompt:
        raise ValueError("Instruct teacher prompt is not a prefix of the canonical student prompt.")
    response_ids = list(sample.tokens[prompt_len:])
    forbidden = set(metadata.get("mopd_forbidden_response_token_ids") or [])
    return teacher_prompt + response_ids, bool(forbidden.intersection(response_ids))


async def reward_func(args, sample, **kwargs):
    mask_reason = _response_mask_reason(args, sample)
    if mask_reason == "truncated":
        return {"meta_info": {"opd_skipped_truncated": True}}
    if mask_reason == "repetition":
        return {
            "meta_info": {
                "opd_skipped_repetition": True,
                "opd_response_mask_reason": "repetition",
            }
        }

    input_ids = sample.tokens
    metadata = sample.metadata or {}
    if metadata.get("mopd_mode") == NON_THINKING:
        input_ids, has_forbidden_think_token = _canonical_teacher_input_ids(sample)
        if has_forbidden_think_token:
            return {"meta_info": {"opd_skipped_unaligned": True, "opd_alignment_reason": "think_token"}}

    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            result = await resp.json()
            if sample.status == Sample.Status.TRUNCATED:
                result.setdefault("meta_info", {})["opd_retained_truncated"] = True
            return result


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    for sample in samples:
        mask_reason = _response_mask_reason(args, sample)
        if mask_reason is not None:
            # Enforce masking even when the teacher response is missing or malformed.
            _mask_response(sample, mask_reason)
            continue

        reward = sample.get_reward_value(args)
        reward_meta = reward.get("meta_info", {}) if isinstance(reward, dict) else {}
        mask_reason = reward_meta.get("opd_response_mask_reason")
        if reward_meta.get("opd_skipped_truncated"):
            mask_reason = "truncated"
        elif reward_meta.get("opd_skipped_repetition"):
            mask_reason = "repetition"
        if mask_reason is not None:
            _mask_response(sample, mask_reason)
            continue
        if reward_meta.get("opd_skipped_unaligned"):
            sample.loss_mask = [0] * sample.response_length
            sample.teacher_log_probs = torch.zeros(sample.response_length, dtype=torch.float32)
            sample.remove_sample = True
            sample.metadata = dict(sample.metadata or {})
            sample.metadata["opd_skipped_unaligned"] = True
            sample.metadata["opd_alignment_reason"] = reward_meta.get("opd_alignment_reason", "unknown")
            continue
        entries = reward["meta_info"]["input_token_logprobs"][1:]
        teacher_log_probs = torch.tensor([item[0] for item in entries], dtype=torch.float32)
        sample.teacher_log_probs = teacher_log_probs[-sample.response_length :]

        # BalMOPD needs the mask before the single-teacher trainer creates it.
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        if sample.status == Sample.Status.TRUNCATED:
            sample.metadata = dict(sample.metadata or {})
            sample.metadata["opd_retained_truncated"] = True

    _compute_balmopd_weights(args, samples, rollout_id=getattr(args, "_current_rollout_id", None))

    # Pure distillation learns from the OPD KL penalty, with zero task reward.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
