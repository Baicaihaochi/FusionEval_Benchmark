"""Prompt-routed multi-teacher on-policy distillation with SGLang teachers."""

from collections.abc import Mapping, Sequence
from functools import lru_cache

import aiohttp

from slime.rollout.on_policy_distillation import post_process_rewards as _post_process_rewards
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


def parse_teacher_urls(entries: Sequence[str] | Mapping[str, str] | None) -> dict[str, str]:
    """Parse repeatable ``NAME=URL`` teacher endpoint configuration."""
    if entries is None:
        raise ValueError("Multi-teacher OPD requires at least one --opd-teacher-url NAME=URL entry.")
    if isinstance(entries, str):
        entries = [entries]
    if isinstance(entries, Mapping):
        items = entries.items()
    else:
        return dict(_parse_teacher_url_entries(tuple(entries)))

    return _normalize_teacher_urls(items)


@lru_cache(maxsize=16)
def _parse_teacher_url_entries(entries: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Cache an immutable normalized representation for repeated CLI configs."""
    items = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --opd-teacher-url {entry!r}; expected NAME=URL.")
        items.append(entry.split("=", 1))
    return tuple(_normalize_teacher_urls(items).items())


def _normalize_teacher_urls(items) -> dict[str, str]:
    teacher_urls = {}
    for raw_name, raw_url in items:
        name = str(raw_name).strip()
        url = str(raw_url).strip()
        if not name or not url:
            raise ValueError(f"Invalid OPD teacher endpoint {raw_name!r}={raw_url!r}; name and URL must be non-empty.")
        if name in teacher_urls:
            raise ValueError(f"Duplicate OPD teacher name: {name!r}.")
        teacher_urls[name] = url

    if not teacher_urls:
        raise ValueError("Multi-teacher OPD requires at least one --opd-teacher-url NAME=URL entry.")
    return teacher_urls


def select_teacher(sample: Sample, teacher_urls: Mapping[str, str], metadata_key: str = "teacher") -> tuple[str, str]:
    """Select exactly one teacher using the configured sample metadata field."""
    metadata = sample.metadata if isinstance(sample.metadata, Mapping) else {}
    teacher = metadata.get(metadata_key)
    if teacher is None or not str(teacher).strip():
        available = ", ".join(sorted(teacher_urls))
        raise ValueError(
            f"Sample metadata is missing the multi-teacher OPD routing key {metadata_key!r}; "
            f"available teachers: {available}."
        )

    teacher = str(teacher).strip()
    if teacher not in teacher_urls:
        available = ", ".join(sorted(teacher_urls))
        raise KeyError(f"Unknown OPD teacher {teacher!r}; available teachers: {available}.")
    return teacher, teacher_urls[teacher]


async def reward_func(args, sample: Sample, **kwargs):
    """Request token log-probabilities from the teacher routed to this prompt."""
    teacher_urls = parse_teacher_urls(args.opd_teacher_urls)
    teacher, url = select_teacher(sample, teacher_urls, args.opd_teacher_metadata_key)
    if sample.status == Sample.Status.TRUNCATED:
        return {
            "meta_info": {
                "opd_teacher": teacher,
                "opd_skipped_truncated": True,
            }
        }
    payload = _build_logprob_payload(sample)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            response = await resp.json()

    response.setdefault("meta_info", {})["opd_teacher"] = teacher
    return response


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Reuse the standard OPD response alignment and train-data conversion."""
    return _post_process_rewards(args, samples, **kwargs)


def _build_logprob_payload(sample: Sample) -> dict:
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        payload["image_data"] = [
            encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]
        ]
    return payload
