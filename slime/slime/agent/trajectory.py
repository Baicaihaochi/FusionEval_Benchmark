"""Token-level trajectory helpers for agent rollouts."""

from __future__ import annotations

import os

import copy
import dataclasses
import logging
from typing import Any

from slime.utils.types import Sample


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """Exact token snapshot for one assistant generation.

    ``prompt_ids`` is the full tokenized prompt sent to the generator for that
    turn. ``output_ids`` is the raw generated output, and
    ``output_log_probs`` is aligned with it when the rollout engine returns
    per-token log probabilities.
    """

    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class TokenSegment:
    """One training segment assembled from an agent trajectory."""

    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]
    rollout_log_probs: list[float] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class TurnSegment:
    """A frozen group of turns before token-level merge."""

    turns: list[TurnRecord]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


def make_turn_segment(
    turns: list[TurnRecord],
    *,
    kind: str = "",
    metadata: dict[str, Any] | None = None,
) -> TurnSegment:
    """Freeze turns and attach conventional segment metadata."""
    frozen_turns = list(turns)
    segment_metadata = dict(metadata or {})
    if kind:
        segment_metadata.setdefault("segment_kind", kind)
    segment_metadata.setdefault("finish_reason", frozen_turns[-1].finish_reason if frozen_turns else "")
    return TurnSegment(turns=frozen_turns, metadata=segment_metadata)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _output_log_probs(turn: TurnRecord) -> list[float]:
    if len(turn.output_log_probs) == len(turn.output_ids):
        return list(turn.output_log_probs)
    logger.warning(
        "[trajectory] turn logprob length mismatch; zeroing output logprobs (%d ids, %d logprobs)",
        len(turn.output_ids),
        len(turn.output_log_probs),
    )
    return [0.0] * len(turn.output_ids)


def merge_turns(turns: list[TurnRecord], *, metadata: dict[str, Any] | None = None) -> TokenSegment | None:
    """Replay turn records into one linear training segment.

    The first turn's prompt becomes the segment prompt. Later turn prompts are
    stitched against ``prompt + response_so_far``. Any new prompt suffix is
    non-model context and receives loss mask 0. If a later prompt diverges
    inside a previous model output, the retained prefix of that whole output
    turn is also masked out, because partial token matches are not a faithful
    training target for that turn.
    """
    if not turns:
        return None

    prompt_ids = list(turns[0].prompt_ids)
    response_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    output_spans: list[tuple[int, int]] = []

    for i, turn in enumerate(turns):
        if i > 0:
            if turn.prompt_ids[: len(prompt_ids)] != prompt_ids:
                logger.warning("[trajectory] merge prompt base changed; starting segment from drifted prompt")
                prompt_ids = list(turn.prompt_ids)
                response_ids = []
                loss_mask = []
                rollout_log_probs = []
                output_spans = []
            else:
                prompt_suffix = turn.prompt_ids[len(prompt_ids) :]
                matched_len = _common_prefix_len(response_ids, prompt_suffix)
                if matched_len < len(response_ids):
                    logger.warning(
                        "[trajectory] merge prefix drift; truncating %d unstitched response tokens",
                        len(response_ids) - matched_len,
                    )
                    for start, end in output_spans:
                        if start < matched_len < end:
                            loss_mask[start:matched_len] = [0] * (matched_len - start)
                            rollout_log_probs[start:matched_len] = [0.0] * (matched_len - start)
                    response_ids = response_ids[:matched_len]
                    loss_mask = loss_mask[:matched_len]
                    rollout_log_probs = rollout_log_probs[:matched_len]
                    output_spans = [
                        (start, min(end, matched_len)) for start, end in output_spans if start < matched_len
                    ]

                context_tail = prompt_suffix[matched_len:]
                response_ids.extend(context_tail)
                loss_mask.extend([0] * len(context_tail))
                rollout_log_probs.extend([0.0] * len(context_tail))

        output_start = len(response_ids)
        response_ids.extend(turn.output_ids)
        loss_mask.extend([1] * len(turn.output_ids))
        rollout_log_probs.extend(_output_log_probs(turn))
        output_spans.append((output_start, len(response_ids)))

    rollout_log_probs = [logprob if mask else 0.0 for logprob, mask in zip(rollout_log_probs, loss_mask, strict=True)]

    return TokenSegment(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        metadata=dict(metadata or {}),
    )


def _per_turn_segments_from_turn_segment(turn_segment: TurnSegment) -> list[TokenSegment]:
    out: list[TokenSegment] = []
    base_meta = dict(turn_segment.metadata or {})
    num_turns = len(turn_segment.turns)

    for turn_idx, turn in enumerate(turn_segment.turns):
        meta = dict(base_meta)
        meta["segment_mode"] = "per_turn"
        meta["turn_index"] = turn_idx
        meta["num_turns"] = num_turns

        if not turn.output_ids:
            continue

        out.append(
            TokenSegment(
                prompt_ids=list(turn.prompt_ids),
                response_ids=list(turn.output_ids),
                loss_mask=[1] * len(turn.output_ids),
                rollout_log_probs=_output_log_probs(turn),
                metadata=meta,
            )
        )

    return out


def _try_merge_turn_segment(turn_segment: TurnSegment) -> TokenSegment | None:
    """Stitch turns when next prompt exactly replays previous raw output.

    This is intended to work with CODE_AGENT_RAW_REPLAY=1 in OpenAIAdapter.
    If OpenHands summarizes/crops/rewrites history, alignment fails and the
    caller should fall back to per-turn segments.
    """
    turns = list(turn_segment.turns)
    if not turns:
        return None

    first = turns[0]
    merged_ids = list(first.prompt_ids)
    response_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []

    for turn_idx, turn in enumerate(turns):
        expected_prefix = merged_ids

        if list(turn.prompt_ids[: len(expected_prefix)]) != expected_prefix:
            logger.warning(
                "[trajectory] raw replay merge failed at turn %d: prompt_len=%d expected_prefix_len=%d",
                turn_idx,
                len(turn.prompt_ids),
                len(expected_prefix),
            )
            return None

        extra_prompt = list(turn.prompt_ids[len(expected_prefix):])
        merged_ids.extend(extra_prompt)
        response_ids.extend(extra_prompt)
        loss_mask.extend([0] * len(extra_prompt))
        rollout_log_probs.extend([0.0] * len(extra_prompt))

        out_ids = list(turn.output_ids)
        if not out_ids:
            continue

        merged_ids.extend(out_ids)
        response_ids.extend(out_ids)

        loss_mask.extend([1] * len(out_ids))
        rollout_log_probs.extend(_output_log_probs(turn))

    if not response_ids:
        return None

    meta = dict(turn_segment.metadata or {})
    meta["segment_mode"] = "merged_raw_replay"
    meta["num_turns"] = len(turns)

    prompt_len = len(first.prompt_ids)
    return TokenSegment(
        prompt_ids=list(first.prompt_ids),
        response_ids=list(merged_ids[prompt_len:]),
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        metadata=meta,
    )



def _merge_turn_segment_chunks(turn_segment: TurnSegment) -> list[TokenSegment]:
    turns = list(turn_segment.turns)
    if not turns:
        return []

    base_meta = dict(turn_segment.metadata or {})
    out: list[TokenSegment] = []

    chunk_prompt_ids: list[int] | None = None
    chunk_merged_ids: list[int] = []
    chunk_response_ids: list[int] = []
    chunk_loss_mask: list[int] = []
    chunk_rollout_log_probs: list[float] = []
    chunk_start_turn = 0

    def flush(chunk_end_turn: int) -> None:
        nonlocal chunk_prompt_ids
        nonlocal chunk_response_ids
        nonlocal chunk_loss_mask
        nonlocal chunk_rollout_log_probs
        nonlocal chunk_start_turn

        if chunk_prompt_ids is None or not chunk_response_ids:
            return

        meta = dict(base_meta)
        meta["segment_mode"] = "chunked_raw_replay"
        meta["chunk_start_turn"] = chunk_start_turn
        meta["chunk_end_turn"] = chunk_end_turn
        meta["num_turns"] = chunk_end_turn - chunk_start_turn + 1

        out.append(TokenSegment(
            prompt_ids=list(chunk_prompt_ids),
            response_ids=list(chunk_response_ids),
            loss_mask=list(chunk_loss_mask),
            rollout_log_probs=list(chunk_rollout_log_probs),
            metadata=meta,
        ))

    def start_new_chunk(turn_idx: int, turn) -> None:
        nonlocal chunk_prompt_ids
        nonlocal chunk_merged_ids
        nonlocal chunk_response_ids
        nonlocal chunk_loss_mask
        nonlocal chunk_rollout_log_probs
        nonlocal chunk_start_turn

        chunk_start_turn = turn_idx
        chunk_prompt_ids = list(turn.prompt_ids)
        chunk_merged_ids = list(turn.prompt_ids)
        chunk_response_ids = []
        chunk_loss_mask = []
        chunk_rollout_log_probs = []

        out_ids = list(turn.output_ids)
        if out_ids:
            chunk_merged_ids.extend(out_ids)
            chunk_response_ids.extend(out_ids)
            chunk_loss_mask.extend([1] * len(out_ids))
            chunk_rollout_log_probs.extend(_output_log_probs(turn))

    for turn_idx, turn in enumerate(turns):
        if chunk_prompt_ids is None:
            start_new_chunk(turn_idx, turn)
            continue

        expected_prefix = chunk_merged_ids

        if list(turn.prompt_ids[: len(expected_prefix)]) != expected_prefix:
            logger.warning(
                "[trajectory] chunk boundary at turn %d: prompt_len=%d expected_prefix_len=%d",
                turn_idx,
                len(turn.prompt_ids),
                len(expected_prefix),
            )
            flush(turn_idx - 1)
            start_new_chunk(turn_idx, turn)
            continue

        extra_prompt = list(turn.prompt_ids[len(expected_prefix):])
        if extra_prompt:
            chunk_merged_ids.extend(extra_prompt)
            chunk_response_ids.extend(extra_prompt)
            chunk_loss_mask.extend([0] * len(extra_prompt))
            chunk_rollout_log_probs.extend([0.0] * len(extra_prompt))

        out_ids = list(turn.output_ids)
        if out_ids:
            chunk_merged_ids.extend(out_ids)
            chunk_response_ids.extend(out_ids)
            chunk_loss_mask.extend([1] * len(out_ids))
            chunk_rollout_log_probs.extend(_output_log_probs(turn))

    flush(len(turns) - 1)
    return out


def merge_turn_segments(segments: list[TurnSegment]) -> list[TokenSegment]:
    """
    Merge turn segments according to CODE_AGENT_SEGMENT_MODE.

    CODE_AGENT_SEGMENT_MODE:
      - per_turn: one TokenSegment per LLM turn
      - merge: strict full merge; drop if prefix alignment fails
      - hybrid: try full merge; if alignment fails, use chunked merge
    """
    mode = os.environ.get("CODE_AGENT_SEGMENT_MODE", "per_turn").strip().lower()
    if mode not in {"per_turn", "merge", "hybrid"}:
        logger.warning("[trajectory] unknown CODE_AGENT_SEGMENT_MODE=%r; use per_turn", mode)
        mode = "per_turn"

    out: list[TokenSegment] = []
    for turn_segment in segments:
        if mode == "per_turn":
            out.extend(_per_turn_segments_from_turn_segment(turn_segment))
            continue

        merged = _try_merge_turn_segment(turn_segment)
        if merged is not None:
            out.append(merged)
            continue

        if mode == "merge":
            logger.warning("[trajectory] merge failed and fallback disabled")
            continue

        chunks = _merge_turn_segment_chunks(turn_segment)
        if chunks:
            logger.warning("[trajectory] full merge failed; using chunked merge chunks=%d", len(chunks))
            out.extend(chunks)
        else:
            logger.warning("[trajectory] chunked merge failed; fallback to per_turn")
            out.extend(_per_turn_segments_from_turn_segment(turn_segment))

    return out

def write_segment_to_sample(sample: Sample, segment: TokenSegment, reward: float, tokenizer) -> None:
    """Populate token, mask, response, reward, and status fields from a segment."""
    sample.tokens = list(segment.prompt_ids) + list(segment.response_ids)
    sample.response_length = len(segment.response_ids)
    sample.loss_mask = list(segment.loss_mask)
    sample.rollout_log_probs = list(segment.rollout_log_probs)
    sample.response = tokenizer.decode(segment.response_ids, skip_special_tokens=False)
    sample.reward = float(reward)
    sample.status = Sample.Status.COMPLETED


def fan_out_sample_segments(
    sample: Sample,
    segments: list[TokenSegment],
    reward: float,
    tokenizer,
    *,
    metadata: dict[str, Any] | None = None,
    rollout_id: int | None = None,
) -> list[Sample]:
    """Emit one Sample per segment; every segment shares the full trajectory reward.

    Sibling samples share ``rollout_id`` so reducers that average by rollout do
    not over-count trajectories split by compaction or sub-agent dispatch.
    An explicit ``rollout_id`` takes precedence; otherwise the sample's existing
    rollout identifier is preserved, with ``sample.index`` as a compatibility fallback.
    """
    k = len(segments)
    shared_reward = float(reward)

    shared_rollout_id = rollout_id
    if shared_rollout_id is None:
        shared_rollout_id = getattr(sample, "rollout_id", None)
    if shared_rollout_id is None:
        shared_rollout_id = getattr(sample, "index", None)
    base_metadata = {**(sample.metadata or {}), **(metadata or {})}

    out: list[Sample] = []
    for i, segment in enumerate(segments):
        sub = sample if i == 0 else copy.copy(sample)
        write_segment_to_sample(sub, segment, shared_reward, tokenizer)
        sub.rollout_id = shared_rollout_id
        sub.metadata = {
            **base_metadata,
            **(segment.metadata or {}),
            "segment_idx": i,
            "num_segments": k,
        }
        out.append(sub)
    return out
