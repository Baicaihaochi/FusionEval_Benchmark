"""Shared adapter primitives for token-capturing agent rollouts."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import uuid
from collections.abc import Callable
from typing import Any

import aiohttp
from aiohttp import web

from slime.agent.trajectory import TokenSegment, TurnRecord


_sampling_debug_logger = logging.getLogger(__name__)


ADAPTER_KEY = web.AppKey("adapter", object)
TOKENIZER_KEY = web.AppKey("tokenizer", object)
SGLANG_URL_KEY = web.AppKey("sglang_url", object)
TOOL_PARSER_KEY = web.AppKey("tool_parser", object)
REASONING_PARSER_KEY = web.AppKey("reasoning_parser", object)


@dataclasses.dataclass
class AdapterChain:
    """Protocol-neutral chat chain state used by HTTP adapters."""

    system_hash: str = ""
    chat_messages: list[dict] = dataclasses.field(default_factory=list)
    tools_schema: list[dict] | None = None
    seen_msgs: int = 0
    msg_hashes: list[str] = dataclasses.field(default_factory=list)
    turns: list[TurnRecord] = dataclasses.field(default_factory=list)


class BaseAdapter:
    """Base HTTP adapter with per-instance session lifecycle state."""

    session_cls: type

    def __init__(self, *, tokenizer, sglang_url, tool_parser=None, reasoning_parser=None) -> None:
        self.store: dict[str, Any] = {}
        self.inflight: dict[str, set[asyncio.Task]] = {}
        self.closed: set[str] = set()
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app[ADAPTER_KEY] = self
        self.app[TOKENIZER_KEY] = tokenizer
        self.app[SGLANG_URL_KEY] = sglang_url.rstrip("/") if isinstance(sglang_url, str) else sglang_url
        self.app[TOOL_PARSER_KEY] = tool_parser
        self.app[REASONING_PARSER_KEY] = reasoning_parser

    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
    ) -> None:
        register_session(
            self.store,
            sid,
            self.session_cls,
            sampling_defaults=sampling_defaults,
            max_context_tokens=max_context_tokens,
        )

    async def shutdown_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        await shutdown_session_tasks(sid, self.closed, self.inflight, wait_timeout=wait_timeout)

    async def finish_session(self, sid: str, *, wait_timeout: float = 5.0) -> list[TokenSegment]:
        raise NotImplementedError


def strip_cache_control(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [strip_cache_control(x) for x in obj]
    return obj


def stable_hash(obj: Any) -> str:
    payload = json.dumps(strip_cache_control(obj), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def json_arguments(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def render_token_ids(chain: AdapterChain, tokenizer) -> list[int]:
    # Qwen3 dense models default to thinking mode. For tool-use agents this may
    # consume most of a short generation before emitting a tool call. Keep the
    # behavior opt-in through an environment variable so Qwen3-Coder/A3B and
    # other model families are unaffected by default.
    disable_thinking = os.environ.get(
        "CODE_AGENT_DISABLE_THINKING", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}

    chat_template_kwargs = {}
    if disable_thinking:
        chat_template_kwargs["enable_thinking"] = False

    enc = tokenizer.apply_chat_template(
        chain.chat_messages,
        tools=chain.tools_schema,
        tokenize=True,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    ids = list(ids)

    if disable_thinking:
        assistant_prefix = tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        empty_think = tokenizer.encode(
            "<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )

        # Only enable this repair for templates whose current generation prompt
        # already ends with assistant-prefix + empty-think.
        generation_suffix = assistant_prefix + empty_think
        if (
            len(ids) >= len(generation_suffix)
            and ids[-len(generation_suffix):] == generation_suffix
        ):
            repaired = []
            i = 0

            while i < len(ids):
                if ids[i:i + len(assistant_prefix)] == assistant_prefix:
                    repaired.extend(assistant_prefix)
                    i += len(assistant_prefix)

                    # Historical assistant blocks omit empty-think under the
                    # Qwen3 dense nothink template. Restore the exact four-token
                    # span, while avoiding duplication for the final generation
                    # prompt or an already-correct historical block.
                    if ids[i:i + len(empty_think)] != empty_think:
                        repaired.extend(empty_think)
                    continue

                repaired.append(ids[i])
                i += 1

            ids = repaired

    return ids


def request_session_id(
    request: web.Request,
    *,
    body: dict | None = None,
    include_x_api_key: bool = False,
) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        sid = auth[7:].strip()
        if sid:
            return sid

    if body is not None:
        metadata = body.get("metadata")
        if isinstance(metadata, dict) and metadata.get("session_id"):
            return str(metadata["session_id"])
        if body.get("user"):
            return str(body["user"])

    if include_x_api_key:
        api_key = request.headers.get("X-Api-Key")
        if api_key:
            return api_key.strip()

    return "default"


def register_session(
    store: dict[str, Any],
    sid: str,
    session_factory: Callable[[], Any],
    *,
    sampling_defaults: dict | None = None,
    max_context_tokens: int = 0,
) -> None:
    if sid in store:
        raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
    session = store[sid] = session_factory()
    session.sampling_defaults = dict(sampling_defaults or {})
    session.max_context_tokens = int(max_context_tokens or 0)


def _sampling_params(session: Any, body: dict, *, max_token_keys: tuple[str, ...], stop_keys: tuple[str, ...]) -> dict:
    sp: dict[str, Any] = {
        "skip_special_tokens": True,
        "spaces_between_special_tokens": False,
        "no_stop_trim": False,
        "max_new_tokens": int(os.environ.get("CODE_AGENT_MAX_NEW_TOKENS", "4096")),
        **(session.sampling_defaults or {}),
    }

    # By default OpenAI clients such as OpenHands may send a small max_tokens
    # value, often 1000. For code-agent rollouts this can truncate long JSON
    # tool calls. When CODE_AGENT_FORCE_MAX_NEW_TOKENS=1, treat
    # CODE_AGENT_MAX_NEW_TOKENS as the hard generation budget and do not let
    # client-side max_tokens lower it.
    force_env_max_new_tokens = os.environ.get("CODE_AGENT_FORCE_MAX_NEW_TOKENS", "0") == "1"
    for key in max_token_keys:
        if body.get(key) is not None:
            if force_env_max_new_tokens:
                continue
            sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", body[key])), int(body[key]))
            break

    for src_k, dst_k in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if src_k in body:
            sp[dst_k] = body[src_k]

    for key in stop_keys:
        if body.get(key):
            sp["stop"] = body[key]
            break

    return sp


async def call_sglang_generate(
    prompt_ids: list[int],
    session: Any,
    body: dict,
    app,
    *,
    max_token_keys: tuple[str, ...],
    stop_keys: tuple[str, ...],
    log_prefix: str,
    logger: logging.Logger,
    session_id: str | None = None,
) -> TurnRecord:
    sp = _sampling_params(session, body, max_token_keys=max_token_keys, stop_keys=stop_keys)

    if session.max_context_tokens > 0:
        remaining_context = session.max_context_tokens - len(prompt_ids)
        if remaining_context <= 0:
            logger.warning(
                "[%s] prompt exceeds max_context_tokens (%d >= %d)",
                log_prefix,
                len(prompt_ids),
                session.max_context_tokens,
            )
            return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[], finish_reason="length")
        sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", remaining_context)), remaining_context)

    sglang_url = app[SGLANG_URL_KEY]
    rid = uuid.uuid4().hex
    headers = {"X-SMG-Routing-Key": session_id} if session_id and session_id != "default" else None
    timeout = aiohttp.ClientTimeout(total=None, sock_read=float(os.environ.get("CODE_AGENT_SGLANG_SOCK_READ_TIMEOUT_SEC", "900")))
    try:
        async with aiohttp.ClientSession(timeout=timeout) as sess, sess.post(
            f"{sglang_url}/generate",
            json={
                "rid": rid,
                "input_ids": prompt_ids,
                "sampling_params": sp,
                "return_logprob": True,
            },
            headers=headers,
        ) as r:
            if r.status >= 400:
                text = await r.text()
                raise RuntimeError(f"sglang upstream {r.status}: {text[:400]}")
            data = await r.json(content_type=None)
        meta = data.get("meta_info") or {}
        output_token_logprobs = meta.get("output_token_logprobs") or []
        output_ids = [x[1] for x in output_token_logprobs]
        output_log_probs = [float(x[0]) for x in output_token_logprobs]
        finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
    except (asyncio.CancelledError, aiohttp.ClientError, asyncio.TimeoutError):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s2:
                await s2.post(f"{sglang_url}/abort_request", json={"rid": rid})
        except Exception:
            pass
        raise

    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=output_ids,
        finish_reason=finish,
        output_log_probs=output_log_probs,
    )


def _consume_detached_task_result(task: asyncio.Task) -> None:
    """Consume a detached task result so asyncio does not emit warnings."""
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def _drain_tasks_on_owner_loop(
    tasks: list[asyncio.Task],
    *,
    wait_timeout: float,
    cancel_timeout: float = 2.0,
) -> int:
    """Drain tasks on their owner loop without waiting forever."""
    if not tasks:
        return 0

    # First allow normal in-flight responses a short time to finish.
    _, pending = await asyncio.wait(
        tasks,
        timeout=max(0.0, wait_timeout),
    )

    if not pending:
        return 0

    # The agent may have exited while its last model request was still active.
    # Ask those request tasks to stop.
    for task in pending:
        task.cancel()

    # Cancellation of network/streaming requests is not guaranteed to finish
    # immediately. Never use an unbounded gather here.
    _, still_pending = await asyncio.wait(
        pending,
        timeout=max(0.0, cancel_timeout),
    )

    # A Python Task cannot be force-killed. Detach any cancellation-resistant
    # leftovers; the session is already closed and its id will not be reused.
    for task in still_pending:
        task.add_done_callback(_consume_detached_task_result)

    return len(still_pending)


async def shutdown_session_tasks(
    sid: str,
    closed: set[str],
    inflight: dict[str, set[asyncio.Task]],
    *,
    wait_timeout: float = 5.0,
) -> None:
    closed.add(sid)

    tasks = [
        task
        for task in inflight.pop(sid, ())
        if not task.done()
    ]
    if not tasks:
        return

    current_loop = asyncio.get_running_loop()
    tasks_by_loop: dict[
        asyncio.AbstractEventLoop,
        list[asyncio.Task],
    ] = {}

    for task in tasks:
        tasks_by_loop.setdefault(task.get_loop(), []).append(task)

    for owner_loop, owner_tasks in tasks_by_loop.items():
        if owner_loop.is_closed() or not owner_loop.is_running():
            _sampling_debug_logger.warning(
                "finish_session sid=%s: owner loop unavailable; "
                "detaching %d request task(s)",
                sid,
                len(owner_tasks),
            )
            continue

        if owner_loop is current_loop:
            try:
                still_pending = await _drain_tasks_on_owner_loop(
                    owner_tasks,
                    wait_timeout=wait_timeout,
                )
            except Exception:
                _sampling_debug_logger.exception(
                    "finish_session sid=%s: local task cleanup failed",
                    sid,
                )
                continue
        else:
            cross_loop_future = asyncio.run_coroutine_threadsafe(
                _drain_tasks_on_owner_loop(
                    owner_tasks,
                    wait_timeout=wait_timeout,
                ),
                owner_loop,
            )

            try:
                still_pending = await asyncio.wait_for(
                    asyncio.wrap_future(cross_loop_future),
                    timeout=max(1.0, wait_timeout + 4.0),
                )
            except asyncio.TimeoutError:
                cross_loop_future.cancel()
                _sampling_debug_logger.warning(
                    "finish_session sid=%s: cross-loop cleanup timed out; "
                    "detaching %d request task(s)",
                    sid,
                    len(owner_tasks),
                )
                continue
            except Exception:
                cross_loop_future.cancel()
                _sampling_debug_logger.exception(
                    "finish_session sid=%s: cross-loop cleanup failed",
                    sid,
                )
                continue

        if still_pending:
            _sampling_debug_logger.warning(
                "finish_session sid=%s: detached %d "
                "cancellation-resistant request task(s)",
                sid,
                still_pending,
            )

async def ok_response(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})
