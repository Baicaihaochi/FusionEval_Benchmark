"""BalMOPD per-group balancing weights (shared by single/multi/paired OPD).

Kept in its own module so ``on_policy_distillation`` (single-teacher) and
``paired_mode_on_policy_distillation`` (two-teacher) can both import it without
a circular import.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_INITIAL_DM_STATE_VERSION = 1
_RUNTIME_STATE_VERSION = 2


def _parse_group_dims(group_by: str) -> list[str]:
    """Split a comma-separated ``--balmopd-group-by`` into metadata field names."""
    return [d.strip() for d in str(group_by or "teacher").split(",") if d.strip()]


def _group_key(sample: Sample, dims: list[str]):
    """Return the hashable per-group key (a tuple of one metadata value per dim).

    Returns ``None`` when any grouping field is missing.  The caller decides
    whether this is allowed in monitor-only mode or is a training-contract error.
    """
    metadata = sample.metadata or {}
    vals = [metadata.get(d) for d in dims]
    if any(v is None for v in vals):
        return None
    return tuple(vals)


def _format_key(key) -> str:
    """Render a group key as a readable string (e.g. ``instruct|math``)."""
    if isinstance(key, tuple):
        return "|".join(str(k) for k in key)
    return str(key)


def _load_initial_dm_state(state_path: Path, group_by: str) -> dict[tuple, float]:
    """Parse and validate a persisted initial-D_m state file."""
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid BalMOPD initial D_m state: {state_path}") from exc
    if (
        not isinstance(state, dict)
        or state.get("version") != _INITIAL_DM_STATE_VERSION
        or state.get("group_by") != group_by
        or not isinstance(state.get("groups"), list)
    ):
        raise ValueError(f"incompatible BalMOPD initial D_m state: {state_path}")

    baseline: dict[tuple, float] = {}
    try:
        for entry in state["groups"]:
            key = tuple(entry["key"])
            if key in baseline:
                raise ValueError(f"duplicate group {_format_key(key)}")
            baseline[key] = float(entry["d_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid BalMOPD initial D_m state: {state_path}") from exc
    return baseline


def _initial_dm_baseline(args, group_kl: dict[tuple, float], group_by: str) -> dict[tuple, float]:
    """Load or atomically initialize the fixed per-group D_m baseline."""
    state_value = getattr(args, "balmopd_initial_dm_state_path", None)
    if not state_value:
        raise ValueError("--balmopd-initial-dm-state-path is required for initial D_m normalization")
    state_path = Path(state_value)

    if state_path.exists():
        baseline = _load_initial_dm_state(state_path, group_by)
    else:
        if not state_path.parent.is_dir():
            raise ValueError(f"BalMOPD initial D_m state parent must be pre-created: {state_path.parent}")
        baseline = dict(group_kl)
        state = {
            "version": _INITIAL_DM_STATE_VERSION,
            "group_by": group_by,
            "groups": [
                {"key": list(key), "d_m": float(value)}
                for key, value in sorted(baseline.items(), key=lambda item: _format_key(item[0]))
            ],
        }
        descriptor, temporary_value = tempfile.mkstemp(
            prefix=f".{state_path.name}.tmp.", dir=state_path.parent
        )
        temporary = Path(temporary_value)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(state, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Link the fsynced file atomically: first writer wins; readers see complete JSON.
                os.link(temporary, state_path)
            except FileExistsError:
                baseline = _load_initial_dm_state(state_path, group_by)
        finally:
            temporary.unlink(missing_ok=True)

    if set(baseline) != set(group_kl):
        raise ValueError(
            "BalMOPD initial D_m state groups do not match current batch: "
            f"state={sorted(map(_format_key, baseline))}, current={sorted(map(_format_key, group_kl))}"
        )
    for group, value in baseline.items():
        if not math.isfinite(value) or value <= 1e-8:
            raise ValueError(f"invalid BalMOPD initial D_m for {_format_key(group)}: {value}")
    return baseline


def _runtime_state_path(state_dir: Path, step: int) -> Path:
    return state_dir / f"step_{step:07d}.json"


def _runtime_config(args, group_by: str) -> dict[str, object]:
    decay = float(getattr(args, "balmopd_dm_ema_decay", 0.0) or 0.0)
    warmup_steps = int(getattr(args, "balmopd_warmup_steps", 0) or 0)
    normalize = bool(getattr(args, "balmopd_normalize_dm_by_initial", False))
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"--balmopd-dm-ema-decay must be in [0, 1), got {decay}")
    if warmup_steps < 0:
        raise ValueError(f"--balmopd-warmup-steps must be non-negative, got {warmup_steps}")
    if warmup_steps and not normalize:
        raise ValueError("--balmopd-warmup-steps requires --balmopd-normalize-dm-by-initial")
    return {
        "group_by": group_by,
        "beta": float(getattr(args, "balmopd_beta", 0.0) or 0.0),
        "normalize": normalize,
        "ema_decay": decay,
        "warmup_steps": warmup_steps,
    }


def _stateful_weighting_enabled(config: dict[str, object]) -> bool:
    return bool(float(config["ema_decay"]) > 0.0 or int(config["warmup_steps"]) > 0)


def _load_runtime_state(
    state_dir: Path,
    step: int,
    config: dict[str, object],
    expected_groups: set[tuple],
) -> dict:
    state_path = _runtime_state_path(state_dir, step)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid or missing BalMOPD runtime state: {state_path}") from exc

    if not isinstance(state, dict) or state.get("version") != _RUNTIME_STATE_VERSION or state.get("step") != step:
        raise ValueError(f"incompatible BalMOPD runtime state: {state_path}")
    for key, expected in config.items():
        if state.get(key) != expected:
            raise ValueError(
                f"BalMOPD runtime state config mismatch for {key}: "
                f"state={state.get(key)!r}, current={expected!r}"
            )

    groups = state.get("groups")
    if not isinstance(groups, list):
        raise ValueError(f"invalid BalMOPD runtime groups: {state_path}")
    parsed: dict[tuple, dict[str, float | None]] = {}
    try:
        for entry in groups:
            group = tuple(entry["key"])
            if group in parsed:
                raise ValueError(f"duplicate group {_format_key(group)}")
            parsed[group] = {
                "warmup_sum": float(entry["warmup_sum"]),
                "warmup_count": float(entry["warmup_count"]),
                "baseline": None if entry["baseline"] is None else float(entry["baseline"]),
                "ema": None if entry["ema"] is None else float(entry["ema"]),
            }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid BalMOPD runtime groups: {state_path}") from exc
    if set(parsed) != expected_groups:
        raise ValueError(
            "BalMOPD runtime state groups do not match current batch: "
            f"state={sorted(map(_format_key, parsed))}, current={sorted(map(_format_key, expected_groups))}"
        )
    state["groups_by_key"] = parsed
    return state


def _write_runtime_state(
    state_dir: Path,
    step: int,
    config: dict[str, object],
    groups: dict[tuple, dict[str, float | None]],
) -> None:
    if not state_dir.is_dir():
        raise ValueError(f"BalMOPD runtime state directory must be pre-created: {state_dir}")
    state = {
        "version": _RUNTIME_STATE_VERSION,
        "step": step,
        **config,
        "groups": [
            {"key": list(group), **values}
            for group, values in sorted(groups.items(), key=lambda item: _format_key(item[0]))
        ],
    }
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".step_{step:07d}.json.tmp.", dir=state_dir
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # On resume, replace stale step state using the loaded model checkpoint.
        os.replace(temporary, _runtime_state_path(state_dir, step))
    finally:
        temporary.unlink(missing_ok=True)


def _advance_runtime_state(
    args,
    *,
    step: int,
    group_kl: dict[tuple, float],
    group_stats: dict[tuple, list[float]],
    group_by: str,
) -> tuple[bool, dict[tuple, float], dict[tuple, float] | None, dict[tuple, float]]:
    """Advance persistent EMA/warm-up state for one optimizer step.

    Returns ``(active, weighting_dm, baseline, ema_dm)``.  During warm-up,
    ``active`` is false and callers must leave OPD sample weights untouched.
    State is journaled by absolute optimizer step, so resume from checkpoint k
    loads exactly step k instead of a potentially-ahead mutable latest state.
    """
    config = _runtime_config(args, group_by)
    if not _stateful_weighting_enabled(config):
        raise ValueError("BalMOPD runtime state requested while EMA and warm-up are disabled")
    state_value = getattr(args, "balmopd_state_dir", None)
    if not state_value:
        raise ValueError("--balmopd-state-dir is required when BalMOPD EMA or warm-up is enabled")
    state_dir = Path(state_value)
    names = set(group_kl)

    if step == 0:
        groups = {
            name: {"warmup_sum": 0.0, "warmup_count": 0.0, "baseline": None, "ema": None}
            for name in names
        }
    else:
        previous = _load_runtime_state(state_dir, step - 1, config, names)
        groups = previous["groups_by_key"]

    decay = float(config["ema_decay"])
    warmup_steps = int(config["warmup_steps"])
    normalize = bool(config["normalize"])

    if step < warmup_steps:
        for name in names:
            groups[name]["warmup_sum"] += float(group_stats[name][0])
            groups[name]["warmup_count"] += float(group_stats[name][1])
        if step == warmup_steps - 1:
            for name in names:
                count = float(groups[name]["warmup_count"])
                baseline = float(groups[name]["warmup_sum"]) / count if count > 0.0 else math.nan
                if not math.isfinite(baseline) or baseline <= 1e-8:
                    raise ValueError(f"invalid BalMOPD warm-up baseline for {_format_key(name)}: {baseline}")
                groups[name]["baseline"] = baseline
                groups[name]["ema"] = baseline
        _write_runtime_state(state_dir, step, config, groups)
        baseline = {
            name: float(groups[name]["baseline"])
            for name in names
            if groups[name]["baseline"] is not None
        }
        ema = {
            name: float(groups[name]["ema"])
            for name in names
            if groups[name]["ema"] is not None
        }
        return False, group_kl, baseline or None, ema

    baseline = None
    if normalize:
        baseline = {name: float(groups[name]["baseline"]) for name in names}
        for name, value in baseline.items():
            if not math.isfinite(value) or value <= 1e-8:
                raise ValueError(f"invalid BalMOPD warm-up baseline for {_format_key(name)}: {value}")

    ema_dm: dict[tuple, float] = {}
    for name in names:
        previous_ema = groups[name]["ema"]
        current = float(group_kl[name])
        ema = current if previous_ema is None else decay * float(previous_ema) + (1.0 - decay) * current
        if not math.isfinite(ema) or ema < 0.0:
            raise ValueError(f"invalid BalMOPD EMA D_m for {_format_key(name)}: {ema}")
        groups[name]["ema"] = ema
        ema_dm[name] = ema

    weighting_dm = ema_dm
    if baseline is not None:
        weighting_dm = {name: ema_dm[name] / baseline[name] for name in names}
    _write_runtime_state(state_dir, step, config, groups)
    return True, weighting_dm, baseline, ema_dm


def _k3(delta: float, *, log_ratio_cap: float | None = 20.0) -> tuple[float, bool]:
    """Non-negative low-variance KL estimator ``exp(-delta) - 1 + delta``.

    ``delta = log p_student - log p_teacher`` (the per-token log-ratio).  Let
    ``z = -delta``; then ``k3 = exp(z) - 1 - z = expm1(z) - z``.

    * ``expm1`` avoids the ``exp(z) - 1`` cancellation for small ``|z|``.
    * For very negative ``delta`` (student exponentially less likely than the
      teacher), ``exp(z)`` overflows float64, so ``z`` is clipped to
      ``log_ratio_cap``.  After clipping the value is computed with the *clipped*
      ``z`` (``expm1(z) - z``), never with the original ``delta``.
    * Returns ``(value, clipped)`` where ``clipped`` flags whether the cap was
      hit, for cap-fraction telemetry.
    * Fail-closed: a non-finite ``delta`` or a non-finite/negative result raises,
      so a corrupt log-ratio never silently poisons the balancing weights.
    """
    if not math.isfinite(delta):
        raise ValueError(f"non-finite BalMOPD log-ratio: {delta}")

    z = -delta  # log p_teacher - log p_student
    clipped = log_ratio_cap is not None and z > log_ratio_cap
    if clipped:
        z = log_ratio_cap

    if abs(z) < 1e-4:
        # Taylor series of expm1(z) - z = z^2/2 + z^3/6 + z^4/24 + z^5/120 + ...
        value = z * z * (0.5 + z * (1.0 / 6.0 + z * (1.0 / 24.0 + z / 120.0)))
    else:
        value = math.expm1(z) - z

    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid BalMOPD k3: delta={delta}, z={z}, value={value}")

    return value, clipped


def _mark_sample_nonfinite(sample: Sample, group, token_pos: int) -> None:
    """Fail-closed: exclude a sample whose student/teacher log-probs are non-finite.

    Zeroes its loss mask and marks it removed so it contributes to neither D_m,
    q_m, nor the training loss.  Records enough context for post-hoc debugging.
    """
    sample.loss_mask = [0] * len(sample.loss_mask)
    sample.remove_sample = True
    sample.opd_bal_weight = None
    sample.metadata["opd_skipped_nonfinite"] = True
    sample.metadata["opd_skipped_nonfinite_group"] = _format_key(group)
    sample.metadata["opd_skipped_nonfinite_token"] = token_pos
    logger.warning(
        "BalMOPD skipping non-finite logprob sample: index=%s group=%s token_pos=%d",
        getattr(sample, "index", None),
        _format_key(group),
        token_pos,
    )


def _global_training_batches(args, samples: list[Sample]) -> list[tuple[list[Sample], int]]:
    """Partition samples exactly at slime's global optimizer-step boundary.

    ``global_batch_size`` counts rollout IDs in the DP scheduler.  Default
    rollouts have no explicit ID and therefore count once per sample; compact
    rollouts may emit several samples with the same explicit ID and must stay in
    the same optimizer step.
    """
    configured_gbs = getattr(args, "global_batch_size", None)
    if configured_gbs is None:
        return [(samples, len(samples))] if samples else []

    global_batch_size = int(configured_gbs)
    if global_batch_size <= 0 or global_batch_size != configured_gbs:
        raise ValueError(f"global_batch_size must be a positive integer, got {configured_gbs!r}")

    rollout_to_samples: dict[tuple[str, object], list[Sample]] = {}
    for position, sample in enumerate(samples):
        key = ("sample", position) if sample.rollout_id is None else ("rollout", sample.rollout_id)
        rollout_to_samples.setdefault(key, []).append(sample)

    rollout_keys = list(rollout_to_samples)
    if len(rollout_keys) % global_batch_size:
        raise ValueError(
            "BalMOPD requires complete global training batches: "
            f"got {len(rollout_keys)} rollout IDs for global_batch_size={global_batch_size}"
        )

    batches = []
    for start in range(0, len(rollout_keys), global_batch_size):
        batch = []
        for key in rollout_keys[start : start + global_batch_size]:
            batch.extend(rollout_to_samples[key])
        batches.append((batch, global_batch_size))
    return batches


def _compute_balmopd_weights(args, samples: list[Sample], *, rollout_id: int | None = None) -> None:
    """Attach BalMOPD weights independently for every global optimizer step."""
    batches = _global_training_batches(args, samples)
    group_by = str(getattr(args, "balmopd_group_by", "teacher") or "teacher")
    runtime_config = _runtime_config(args, group_by)
    stateful = _stateful_weighting_enabled(runtime_config)
    if stateful and rollout_id is None:
        raise ValueError("rollout_id is required when BalMOPD EMA or warm-up is enabled")
    combined_metrics: dict[str, float] = {}
    for step_id, (batch, global_batch_size) in enumerate(batches):
        optimizer_step = None if rollout_id is None else rollout_id * len(batches) + step_id
        step_metrics = _compute_balmopd_weights_for_batch(
            args,
            batch,
            global_batch_size=global_batch_size,
            optimizer_step=optimizer_step,
        )
        if len(batches) == 1:
            combined_metrics.update(step_metrics)
        else:
            combined_metrics.update({f"step_{step_id}/{name}": value for name, value in step_metrics.items()})

    if combined_metrics:
        _attach_group_metrics(samples, combined_metrics)


def _compute_balmopd_weights_for_batch(
    args,
    samples: list[Sample],
    *,
    global_batch_size: int,
    optimizer_step: int | None = None,
) -> dict[str, float]:
    """Attach per-sample BalMOPD balancing weights in place.

    Samples are grouped by the metadata fields ``args.balmopd_group_by``
    (default ``"teacher"``; a comma-separated list groups by several fields, e.g.
    ``"teacher,domain"``).  Estimates each group's reverse KL ``D_m`` over the
    current global training batch with the non-negative k3 estimator, then sets
    ``sample.opd_bal_weight`` to ``alpha_m = omega_m / q_m`` where ``omega_m`` is
    ``softmax(beta * D)_m`` and ``q_m = |B_m^valid| / GBS`` is the empirical
    sampling proportion of the group.  ``alpha_m`` exactly cancels the implicit
    ``q_m`` inside the global response-mean, so the update equals
    ``sum_m omega_m g_m`` for any sampling.  Under uniform sampling this reduces
    to ``alpha_m = M * omega_m``, and ``beta -> 0`` yields exactly ``1.0``
    (vanilla OPD).  The weight is a stop-gradient batch statistic stored as a
    plain float and is never back-propagated through.

    Samples with a non-finite student/teacher log-prob are fail-closed: they are
    masked out and removed, never contributing to ``D_m``/``q_m``/training.

    The per-group ``D_m``/``omega``/``alpha`` (and ``q_m``) are also written as a
    flat dict into ``sample.metadata["opd_bal_group_metrics"]`` so the rollout
    manager can forward them as plottable metrics.  Returns that flat metric dict
    to the rollout-level wrapper, which preserves metrics from every optimizer
    step in a multi-step rollout.
    """
    beta = float(getattr(args, "balmopd_beta", 0.0) or 0.0)
    group_by = str(getattr(args, "balmopd_group_by", "teacher") or "teacher")
    group_dims = _parse_group_dims(group_by)
    cap = getattr(args, "balmopd_k3_log_ratio_cap", 20.0)
    log_ratio_cap = None if cap is None else float(cap)

    # group key -> [sum of per-sample token-means, sample count]
    group_stats: dict[tuple, list[float]] = {}
    cap_tokens = 0
    total_tokens = 0
    nonfinite_samples = 0

    for sample in samples:
        if sample.remove_sample:
            continue
        group = _group_key(sample, group_dims)
        if group is None:
            # Check trainable samples before log-probs to prevent silent fallback to vanilla OPD.
            has_effective_tokens = sample.loss_mask is None or any(sample.loss_mask)
            if beta > 0.0 and has_effective_tokens:
                metadata = sample.metadata or {}
                missing = [dim for dim in group_dims if metadata.get(dim) is None]
                raise ValueError(
                    "sample is missing BalMOPD grouping metadata: "
                    f"{','.join(missing)} (sample_index={getattr(sample, 'index', None)!r})"
                )
            continue
        if sample.loss_mask is None:
            continue
        rollout_log_probs = sample.rollout_log_probs
        teacher_log_probs = sample.teacher_log_probs
        if rollout_log_probs is None or teacher_log_probs is None:
            continue

        n = min(len(sample.loss_mask), len(rollout_log_probs), len(teacher_log_probs))
        sum_k3 = 0.0
        n_tokens = 0
        sample_clipped = False
        nonfinite = False
        nonfinite_token = -1
        for t in range(n):
            rlp = float(rollout_log_probs[t])
            tlp = float(teacher_log_probs[t])
            if not (math.isfinite(rlp) and math.isfinite(tlp)):
                # Reject non-finite log-probs before applying the loss mask.
                nonfinite = True
                nonfinite_token = t
                break
            if not sample.loss_mask[t]:
                continue
            value, clipped = _k3(rlp - tlp, log_ratio_cap=log_ratio_cap)
            sum_k3 += value
            n_tokens += 1
            total_tokens += 1
            if clipped:
                cap_tokens += 1
                sample_clipped = True

        if nonfinite:
            _mark_sample_nonfinite(sample, group, nonfinite_token)
            nonfinite_samples += 1
            continue

        if n_tokens == 0:
            continue

        if sample_clipped:
            sample.metadata["opd_k3_clipped"] = True
        stats = group_stats.setdefault(group, [0.0, 0.0])
        stats[0] += sum_k3 / n_tokens  # per-sample token mean
        stats[1] += 1.0

    # Per-group discrepancy: mean of per-sample token means (length-invariant
    # per response), matching Eq. (18) in the BalMOPD paper.
    group_kl = {name: v[0] / v[1] for name, v in group_stats.items() if v[1] > 0.0}
    if not group_kl:
        if nonfinite_samples:
            logger.warning("BalMOPD: all samples non-finite (%d); no D_m computed", nonfinite_samples)
        return {}

    names = list(group_kl.keys())
    gbs = float(global_batch_size)
    # Empirical sampling proportion q_m = |B_m^valid| / GBS.
    q_m = {name: group_stats[name][1] / gbs for name in names}
    cap_fraction = cap_tokens / total_tokens if total_tokens else 0.0

    if nonfinite_samples:
        logger.warning(
            "BalMOPD skipped %d non-finite samples (k3_cap_tokens=%d k3_cap_frac=%.6f)",
            nonfinite_samples,
            cap_tokens,
            cap_fraction,
        )

    flat_metrics: dict[str, float] = {}
    for name in names:
        ks = _format_key(name)
        flat_metrics[f"d_m/{ks}"] = float(group_kl[name])
        flat_metrics[f"q_m/{ks}"] = float(q_m[name])

    weighting_dm = group_kl
    baseline = None
    ema_dm = None
    runtime_config = _runtime_config(args, group_by)
    stateful = _stateful_weighting_enabled(runtime_config)
    if stateful:
        if optimizer_step is None:
            raise ValueError("optimizer_step is required when BalMOPD EMA or warm-up is enabled")
        active, weighting_dm, baseline, ema_dm = _advance_runtime_state(
            args,
            step=optimizer_step,
            group_kl=group_kl,
            group_stats=group_stats,
            group_by=group_by,
        )
        flat_metrics["balmopd/active"] = float(active)
        for name in names:
            ks = _format_key(name)
            if ema_dm is not None and name in ema_dm:
                flat_metrics[f"d_m_ema/{ks}"] = float(ema_dm[name])
            if baseline is not None and name in baseline:
                flat_metrics[f"d_m_warmup_baseline/{ks}"] = float(baseline[name])
                flat_metrics[f"d_m_normalized_raw/{ks}"] = float(group_kl[name] / baseline[name])
                if ema_dm is not None and name in ema_dm:
                    flat_metrics[f"d_m_normalized_ema/{ks}"] = float(ema_dm[name] / baseline[name])
            if optimizer_step < int(runtime_config["warmup_steps"]):
                flat_metrics[f"warmup_valid_samples/{ks}"] = float(group_stats[name][1])
        if not active:
            logger.info(
                "BalMOPD warm-up step=%d/%d group_by=%s D_m=%s q_m=%s",
                optimizer_step + 1,
                int(runtime_config["warmup_steps"]),
                group_by,
                {_format_key(k): round(v, 5) for k, v in group_kl.items()},
                {_format_key(k): round(v, 5) for k, v in q_m.items()},
            )
            return flat_metrics
        if baseline is not None:
            for name in names:
                ks = _format_key(name)
                flat_metrics[f"d_m_initial/{ks}"] = float(baseline[name])
                flat_metrics[f"d_m_normalized/{ks}"] = float(weighting_dm[name])
    elif bool(getattr(args, "balmopd_normalize_dm_by_initial", False)):
        baseline = _initial_dm_baseline(args, group_kl, group_by)
        weighting_dm = {name: group_kl[name] / baseline[name] for name in names}
        for name in names:
            ks = _format_key(name)
            flat_metrics[f"d_m_initial/{ks}"] = float(baseline[name])
            flat_metrics[f"d_m_normalized/{ks}"] = float(weighting_dm[name])

    if beta <= 0.0:
        # beta=0: monitor D_m and q_m without changing vanilla OPD/MOPD weights.
        logger.info(
            "BalMOPD monitor beta=0.000 group_by=%s D_m=%s q_m=%s k3_cap_tokens=%d k3_cap_frac=%.6f nonfinite_samples=%d",
            group_by,
            {_format_key(k): round(v, 5) for k, v in group_kl.items()},
            {_format_key(k): round(v, 5) for k, v in q_m.items()},
            cap_tokens,
            cap_fraction,
            nonfinite_samples,
        )
        return flat_metrics

    logits = [beta * weighting_dm[n] for n in names]
    zmax = max(logits)
    expz = [math.exp(z - zmax) for z in logits]
    zsum = sum(expz)
    # Theoretical group mixture weight (softmax over discrepancies, sums to 1).
    omega = {name: e / zsum for name, e in zip(names, expz, strict=True)}

    # alpha_m = omega_m / q_m corrects unequal sampling and truncation in the global mean.
    # For uniform sampling, alpha_m = M * omega_m, tending to 1 as beta -> 0.
    alpha = {name: omega[name] * gbs / group_stats[name][1] for name in names}
    for name in names:
        ks = _format_key(name)
        flat_metrics[f"omega/{ks}"] = float(omega[name])
        flat_metrics[f"alpha/{ks}"] = float(alpha[name])

    logger.info(
        "BalMOPD beta=%.3f group_by=%s D_m=%s weighting_D_m=%s omega=%s q_m=%s sample_multiplier=%s k3_cap_tokens=%d k3_cap_frac=%.6f nonfinite_samples=%d",
        beta,
        group_by,
        {_format_key(k): round(v, 5) for k, v in group_kl.items()},
        {_format_key(k): round(v, 5) for k, v in weighting_dm.items()},
        {_format_key(k): round(v, 5) for k, v in omega.items()},
        {_format_key(k): round(v, 5) for k, v in q_m.items()},
        {_format_key(k): round(v, 5) for k, v in alpha.items()},
        cap_tokens,
        cap_fraction,
        nonfinite_samples,
    )

    for sample in samples:
        if sample.remove_sample:
            continue
        if not sample.loss_mask or sum(sample.loss_mask) == 0:
            continue
        group = _group_key(sample, group_dims)
        if group is not None and group in alpha:
            sample.opd_bal_weight = float(alpha[group])
            sample.metadata["opd_bal_weight"] = float(alpha[group])
            sample.metadata["opd_bal_d_m"] = float(group_kl[group])
            if ema_dm is not None:
                sample.metadata["opd_bal_d_m_ema"] = float(ema_dm[group])
            if baseline is not None:
                sample.metadata["opd_bal_d_m_initial"] = float(baseline[group])
                sample.metadata["opd_bal_d_m_normalized"] = float(weighting_dm[group])
    return flat_metrics


def _attach_group_metrics(samples: list[Sample], flat_metrics: dict[str, float]) -> None:
    """Attach the batch-level per-group metrics to every (non-removed) sample."""
    for sample in samples:
        if sample.remove_sample:
            continue
        sample.metadata = dict(sample.metadata or {})
        sample.metadata["opd_bal_group_metrics"] = dict(flat_metrics)
