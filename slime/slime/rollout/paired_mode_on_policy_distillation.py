"""Paired think/non-think on-policy distillation for Qwen-style models.

Each source prompt produces exactly two student rollouts.  The student and
teacher prompts are rendered independently with their canonical chat
templates; only the shared response-token span is distilled.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import aiohttp
import torch

from slime.rollout.balmopd import _compute_balmopd_weights, _k3
from slime.rollout.data_source import RolloutDataSource
from slime.rollout.multi_teacher_on_policy_distillation import parse_teacher_urls, select_teacher
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

NON_THINKING = "non_thinking"
THINKING = "thinking"
MODE_ORDER = (NON_THINKING, THINKING)
logger = logging.getLogger(__name__)


class ThinkingPrefixAlignmentError(ValueError):
    """A thinking rollout cannot be aligned to the canonical teacher prompt."""


@dataclass(frozen=True)
class PromptVariant:
    mode: str
    teacher: str
    student_prompt: str
    student_prompt_ids: tuple[int, ...]
    teacher_prompt_ids: tuple[int, ...]


@dataclass(frozen=True)
class TeacherRequest:
    input_ids: tuple[int, ...]
    student_response_length: int
    teacher_response_length: int
    masked_response_prefix_length: int


def _as_messages(prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if not isinstance(prompt, list) or not prompt:
        raise TypeError("Paired MOPD prompt must be a non-empty string or message list.")
    return copy.deepcopy(prompt)


def _render(tokenizer, messages: list[dict[str, Any]], **kwargs) -> tuple[str, tuple[int, ...]]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    ids = tuple(tokenizer.encode(text, add_special_tokens=False))
    return text, ids


def build_prompt_variant(
    *,
    prompt: str | list[dict[str, Any]],
    mode: str,
    teacher: str,
    student_tokenizer,
    teacher_tokenizer,
) -> PromptVariant:
    """Render canonical student and teacher prompts for one mode."""
    if mode not in MODE_ORDER:
        raise ValueError(f"Unknown paired MOPD mode {mode!r}; expected one of {MODE_ORDER}.")

    messages = _as_messages(prompt)
    student_prompt, student_ids = _render(
        student_tokenizer,
        messages,
        enable_thinking=(mode == THINKING),
    )
    _, teacher_ids = _render(teacher_tokenizer, messages)
    return PromptVariant(
        mode=mode,
        teacher=teacher,
        student_prompt=student_prompt,
        student_prompt_ids=student_ids,
        teacher_prompt_ids=teacher_ids,
    )


def build_teacher_request(sample: Sample) -> TeacherRequest:
    """Build a canonical teacher sequence aligned to the student response.

    Qwen3 Thinking-2507 inserts ``<think>\n`` into its prompt, whereas the
    hybrid Qwen3 student emits those tokens as the beginning of its thinking
    response.  Those generated prefix tokens are moved into the teacher prompt
    and masked from training.  In non-thinking mode, the empty think block is
    part of the student prompt only and the response needs no offset.
    """
    metadata = sample.metadata if isinstance(sample.metadata, Mapping) else {}
    mode = metadata.get("mopd_mode")
    if mode not in MODE_ORDER:
        raise ValueError(f"Sample has invalid or missing mopd_mode: {mode!r}.")

    teacher_prompt_ids = tuple(metadata.get("mopd_teacher_prompt_ids") or ())
    expected_student_prompt_ids = tuple(metadata.get("mopd_student_prompt_ids") or ())
    if not teacher_prompt_ids or not expected_student_prompt_ids:
        raise ValueError("Sample is missing canonical paired-MOPD prompt token metadata.")
    if sample.response_length < 0 or sample.response_length > len(sample.tokens):
        raise ValueError("Invalid response_length for paired-MOPD sample.")

    prompt_len = len(sample.tokens) - sample.response_length
    actual_student_prompt_ids = tuple(sample.tokens[:prompt_len])
    if actual_student_prompt_ids != expected_student_prompt_ids:
        raise ValueError("Student prompt tokens changed after canonical paired-MOPD rendering.")

    response_ids = tuple(sample.tokens[prompt_len:])
    masked_prefix_len = 0
    teacher_response_ids = response_ids

    if mode == THINKING:
        if teacher_prompt_ids[: len(actual_student_prompt_ids)] != actual_student_prompt_ids:
            raise ValueError("Thinking teacher prompt is not an extension of the student prompt.")
        teacher_only_prefix = teacher_prompt_ids[len(actual_student_prompt_ids) :]
        if not teacher_only_prefix:
            raise ValueError("Thinking teacher template did not add a thinking prefix.")
        if response_ids[: len(teacher_only_prefix)] != teacher_only_prefix:
            raise ThinkingPrefixAlignmentError(
                "Thinking rollout does not begin with the exact prefix inserted by the teacher template; "
                "refusing misaligned distillation."
            )
        masked_prefix_len = len(teacher_only_prefix)
        teacher_response_ids = response_ids[masked_prefix_len:]
    else:
        if actual_student_prompt_ids[: len(teacher_prompt_ids)] != teacher_prompt_ids:
            raise ValueError("Non-thinking student prompt is not an extension of the Instruct teacher prompt.")

    return TeacherRequest(
        input_ids=teacher_prompt_ids + teacher_response_ids,
        student_response_length=len(response_ids),
        teacher_response_length=len(teacher_response_ids),
        masked_response_prefix_length=masked_prefix_len,
    )


class PairedModeRolloutDataSource(RolloutDataSource):
    """Duplicate every source prompt into canonical non-think and think rows."""

    def __init__(self, args):
        if args.n_samples_per_prompt != 2:
            raise ValueError("PairedModeRolloutDataSource requires --n-samples-per-prompt 2.")
        if args.apply_chat_template:
            raise ValueError("PairedModeRolloutDataSource requires --no-apply-chat-template.")
        super().__init__(args)

        self.student_tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        teacher_tokenizers = parse_teacher_urls(args.opd_teacher_tokenizers)
        self.teacher_tokenizers = {
            name: load_tokenizer(path, trust_remote_code=True) for name, path in teacher_tokenizers.items()
        }
        self.mode_teachers = {
            NON_THINKING: args.paired_mopd_non_thinking_teacher,
            THINKING: args.paired_mopd_thinking_teacher,
        }
        missing = sorted(set(self.mode_teachers.values()) - set(self.teacher_tokenizers))
        if missing:
            raise ValueError(f"Missing --opd-teacher-tokenizer entries for: {', '.join(missing)}")

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        groups = super().get_samples(num_samples)
        for group in groups:
            if len(group) != 2:
                raise AssertionError("Paired MOPD group cardinality changed.")
            raw_prompt = group[0].prompt
            for sample, mode in zip(group, MODE_ORDER, strict=True):
                teacher = self.mode_teachers[mode]
                variant = build_prompt_variant(
                    prompt=raw_prompt,
                    mode=mode,
                    teacher=teacher,
                    student_tokenizer=self.student_tokenizer,
                    teacher_tokenizer=self.teacher_tokenizers[teacher],
                )
                if len(variant.student_prompt_ids) > self.args.rollout_max_prompt_len:
                    raise ValueError(
                        f"Preprocessed prompt exceeds {self.args.rollout_max_prompt_len} tokens in {mode} mode."
                    )
                sample.prompt = variant.student_prompt
                sample.metadata = dict(sample.metadata or {})
                sample.metadata.update(
                    {
                        "teacher": teacher,
                        "mopd_mode": mode,
                        "mopd_student_prompt_ids": list(variant.student_prompt_ids),
                        "mopd_teacher_prompt_ids": list(variant.teacher_prompt_ids),
                    }
                )
        return groups


async def reward_func(args, sample: Sample, **kwargs):
    """Score the response with the mode-specific teacher canonical prompt."""
    teacher_urls = parse_teacher_urls(args.opd_teacher_urls)
    teacher, url = select_teacher(sample, teacher_urls, args.opd_teacher_metadata_key)

    if sample.status == Sample.Status.TRUNCATED:
        return {"meta_info": {"opd_teacher": teacher, "opd_skipped_truncated": True}}

    try:
        request = build_teacher_request(sample)
    except ThinkingPrefixAlignmentError:
        # Without the thinking prefix, token alignment to Thinking-2507 is undefined.
        # Skip this sample without rewriting tokens or removing its paired row.
        sample.metadata["mopd_alignment_status"] = "skipped_thinking_prefix_mismatch"
        logger.warning(
            "Skipping unalignable thinking rollout: response_length=%d response_prefix_ids=%s",
            sample.response_length,
            list(sample.tokens[-sample.response_length :][:8]) if sample.response_length else [],
        )
        return {
            "meta_info": {
                "opd_teacher": teacher,
                "opd_skipped_unaligned": True,
                "opd_alignment_reason": "thinking_prefix_mismatch",
            }
        }
    sample.metadata["mopd_teacher_response_length"] = request.teacher_response_length
    sample.metadata["mopd_masked_response_prefix_length"] = request.masked_response_prefix_length
    sample.metadata["mopd_teacher_input_ids"] = list(request.input_ids)
    payload = {
        "input_ids": list(request.input_ids),
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            response = await resp.json()
    response.setdefault("meta_info", {})["opd_teacher"] = teacher
    return response


def _teacher_input_logprobs(reward: dict, expected_ids: Sequence[int]) -> torch.Tensor:
    entries = reward.get("meta_info", {}).get("input_token_logprobs")
    if not isinstance(entries, list) or len(entries) != len(expected_ids):
        raise ValueError("Teacher returned an unexpected number of input token log-probabilities.")
    returned_ids = [entry[1] for entry in entries]
    if returned_ids != list(expected_ids):
        raise ValueError("Teacher-scored token IDs do not exactly match the canonical request.")
    values = [0.0 if entry[0] is None else float(entry[0]) for entry in entries]
    return torch.tensor(values, dtype=torch.float32)


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Attach response-aligned log-probs and enforce all masking invariants."""
    for sample in samples:
        if sample.status == Sample.Status.TRUNCATED:
            sample.loss_mask = [0] * sample.response_length
            sample.teacher_log_probs = torch.zeros(sample.response_length, dtype=torch.float32)
            sample.remove_sample = True
            continue

        reward_meta = sample.reward.get("meta_info", {}) if isinstance(sample.reward, Mapping) else {}
        if reward_meta.get("opd_skipped_unaligned"):
            if reward_meta.get("opd_alignment_reason") != "thinking_prefix_mismatch":
                raise ValueError("Unknown paired-MOPD alignment skip reason.")
            sample.loss_mask = [0] * sample.response_length
            sample.teacher_log_probs = torch.zeros(sample.response_length, dtype=torch.float32)
            sample.remove_sample = True
            continue

        metadata = sample.metadata
        expected_ids = metadata.get("mopd_teacher_input_ids")
        teacher_response_len = int(metadata.get("mopd_teacher_response_length", -1))
        prefix_len = int(metadata.get("mopd_masked_response_prefix_length", -1))
        if expected_ids is None or teacher_response_len < 0 or prefix_len < 0:
            raise ValueError("Paired-MOPD alignment metadata is incomplete after teacher scoring.")

        all_log_probs = _teacher_input_logprobs(sample.reward, expected_ids)
        if teacher_response_len:
            response_log_probs = all_log_probs[-teacher_response_len:]
        else:
            response_log_probs = torch.empty(0, dtype=torch.float32)
        sample.teacher_log_probs = torch.cat(
            [torch.zeros(prefix_len, dtype=torch.float32), response_log_probs]
        )
        sample.loss_mask = [0] * prefix_len + [1] * teacher_response_len
        if len(sample.teacher_log_probs) != sample.response_length:
            raise ValueError("Teacher/student response alignment produced an invalid log-probability length.")
        if len(sample.loss_mask) != sample.response_length:
            raise ValueError("Teacher/student response alignment produced an invalid loss-mask length.")

    _compute_balmopd_weights(args, samples, rollout_id=getattr(args, "_current_rollout_id", None))

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
