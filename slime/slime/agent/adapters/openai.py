"""OpenAI-compatible adapters for agent rollouts.

The adapter exposes ``/v1/chat/completions`` and ``/v1/responses``. Both
endpoints render incoming messages with the served model's chat template, call
SGLang ``/generate`` with ``input_ids``, and record the exact sampled token
ids/logprobs as ``TurnRecord`` objects. New code should use ``OpenAIAdapter``
and call ``finish_session()`` at trajectory end to drain trainable
``TokenSegment`` objects.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import secrets
import time
from typing import Any

from aiohttp import web

from slime.agent.adapters.common import ADAPTER_KEY, REASONING_PARSER_KEY, TOKENIZER_KEY, TOOL_PARSER_KEY
from slime.agent.adapters.common import AdapterChain as Chain
from slime.agent.adapters.common import BaseAdapter, call_sglang_generate
from slime.agent.adapters.common import json_arguments as _json_arguments
from slime.agent.adapters.common import ok_response, render_token_ids, request_session_id
from slime.agent.adapters.common import stable_hash as _hash
from slime.agent.parsing import ParsedModelOutput, parse_model_output
from slime.agent.trajectory import TokenSegment, TurnRecord, TurnSegment, make_turn_segment, merge_turn_segments


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Session:
    main: Chain = dataclasses.field(default_factory=Chain)
    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    segments: list[TurnSegment] = dataclasses.field(default_factory=list)
    raw_outputs: list[str] = dataclasses.field(default_factory=list)


class OpenAIAdapter(BaseAdapter):
    """OpenAI-compatible HTTP adapter with session lifecycle helpers."""

    session_cls = Session

    def __init__(self, *, tokenizer, sglang_url, tool_parser=None, reasoning_parser=None) -> None:
        super().__init__(
            tokenizer=tokenizer,
            sglang_url=sglang_url,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
        )
        self.app.router.add_post("/v1/chat/completions", _handle_chat_completions)
        self.app.router.add_post("/v1/responses", _handle_responses)
        self.app.router.add_get("/healthz", _ok)
        self.app.router.add_get("/v1/models", _ok)

    async def finish_session(self, sid: str, *, wait_timeout: float = 5.0) -> list[TokenSegment]:
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        s = self.store.pop(sid, None)
        if s is None:
            return []
        if s.main.turns:
            s.segments.append(make_turn_segment(s.main.turns, kind="final"))
        return merge_turn_segments(s.segments)


def _flatten_content(content: Any) -> str:
    """Flatten OpenAI text/content parts into a chat-template string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        typ = item.get("type")
        if typ in {"text", "input_text", "output_text"}:
            parts.append(item.get("text", ""))
        elif typ in {"image_url", "input_image"}:
            parts.append("[image omitted]")
        elif "content" in item:
            parts.append(_flatten_content(item.get("content")))
        elif "text" in item:
            parts.append(str(item.get("text") or ""))
    return "\n".join(p for p in parts if p)



def _raw_output_for_replay(text: str) -> str:
    if not text:
        return ""
    return text.removesuffix("<|im_end|>")


def _raw_replay_chat_messages(messages: list[dict], raw_outputs: list[str]) -> list[dict]:
    if os.environ.get("CODE_AGENT_RAW_REPLAY", "0") != "1":
        return messages

    assistant_indices = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "assistant"
    ]

    if not raw_outputs:
        logger.warning("[adapter_raw_replay] no raw_outputs yet; messages=%d", len(messages))
        return messages

    if not assistant_indices:
        logger.warning(
            "[adapter_raw_replay] no assistant history: raw_outputs=%d messages=%d",
            len(raw_outputs),
            len(messages),
        )
        roles = []
        for m in messages[:8]:
            if isinstance(m, dict):
                roles.append({
                    "role": m.get("role"),
                    "keys": sorted(m.keys()),
                    "content_head": str(m.get("content"))[:300],
                    "has_tool_calls": bool(m.get("tool_calls")),
                    "tool_call_id": m.get("tool_call_id"),
                })
            else:
                roles.append({"type": type(m).__name__})
        logger.warning("[adapter_raw_replay_debug] message_heads=%r", roles)
        return messages

    if len(assistant_indices) > len(raw_outputs):
        logger.warning(
            "[adapter_raw_replay] skip: assistant_msgs=%d raw_outputs=%d messages=%d",
            len(assistant_indices),
            len(raw_outputs),
            len(messages),
        )
        return messages

    # Important:
    # OpenHands may crop/summarize old context and keep only recent assistant/tool history.
    # In that case assistant_msgs < raw_outputs. Use the suffix of raw_outputs.
    replay_outputs = raw_outputs[-len(assistant_indices):]

    out = []
    raw_i = 0
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            out.append({
                "role": "assistant",
                "content": _raw_output_for_history_replay(
                    _raw_output_for_replay(replay_outputs[raw_i])
                ),
            })
            raw_i += 1
        else:
            out.append(m)

    if len(assistant_indices) != len(raw_outputs):
        logger.warning(
            "[adapter_raw_replay] suffix replay assistant_history=%d raw_outputs=%d",
            len(assistant_indices),
            len(raw_outputs),
        )
    else:
        logger.warning("[adapter_raw_replay] replaced assistant history=%d", raw_i)

    return out


def _normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    name = function.get("name") or call.get("name") or "tool"
    arguments = function.get("arguments", call.get("arguments", {}))
    out = {
        "type": "function",
        "function": {
            "name": name,
            "arguments": _json_arguments(arguments),
        },
    }
    if call.get("id"):
        out["id"] = call["id"]
    return out


def _translate_chat_messages(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages -> tokenizer chat-template messages."""
    translated: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "developer":
            role = "system"

        if role in {"system", "user"}:
            translated.append({"role": role, "content": _flatten_content(content)})
        elif role == "tool":
            tool_msg = {"role": "tool", "content": _flatten_content(content)}
            if msg.get("tool_call_id"):
                tool_msg["tool_call_id"] = msg["tool_call_id"]
            translated.append(tool_msg)
        elif role == "assistant":
            assistant: dict[str, Any] = {"role": "assistant", "content": _flatten_content(content)}
            if msg.get("reasoning_content"):
                assistant["reasoning_content"] = msg["reasoning_content"]
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                assistant["tool_calls"] = [_normalize_tool_call(c) for c in tool_calls if isinstance(c, dict)]
            translated.append(assistant)
    return translated


def _normalize_tool(tool: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    if tool.get("type") != "function":
        return None
    if isinstance(tool.get("function"), dict):
        function = tool["function"]
        name = function.get("name")
        if not name:
            return None
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": function.get("description", ""),
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            },
        }
    name = tool.get("name")
    if not name:
        return None
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        },
    }


def _normalize_tools(tools: list[dict] | None) -> list[dict] | None:
    normalized = [_normalize_tool(t) for t in tools or []]
    return [t for t in normalized if t is not None] or None


def _responses_input_to_messages(input_value: Any, instructions: Any = None) -> list[dict]:
    """Responses API input -> OpenAI chat message list.

    This intentionally covers the common message/function-call shapes used by
    agent SDKs. Unknown input items are preserved as user text where possible.
    """
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": _flatten_content(instructions)})

    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages

    if not isinstance(input_value, list):
        messages.append({"role": "user", "content": _flatten_content(input_value)})
        return messages

    for item in input_value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            messages.append({"role": "user", "content": str(item)})
            continue

        typ = item.get("type")
        if typ == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id") or "",
                    "content": item.get("output", ""),
                }
            )
        elif typ == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id") or f"call_{secrets.token_hex(8)}",
                            "type": "function",
                            "function": {
                                "name": item.get("name", "tool"),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    ],
                }
            )
        elif item.get("role"):
            messages.append({"role": item.get("role"), "content": item.get("content", "")})
        elif typ == "message":
            messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})
        else:
            messages.append({"role": "user", "content": _flatten_content(item)})
    return messages

def _select_kind(s: Session, messages: list[dict]) -> str:
    target = s.main
    msg_hashes = [_hash(m) for m in messages]
    if target.seen_msgs == 0:
        kind = "new"
    else:
        is_append = len(msg_hashes) >= target.seen_msgs and msg_hashes[: target.seen_msgs] == target.msg_hashes
        if is_append:
            kind = "append"
        else:
            if target.turns:
                s.segments.append(make_turn_segment(target.turns, kind="wipe"))
            kind = "wipe"
    return kind


def _replace_chat_messages(target: Chain, messages: list[dict], tools_schema: list[dict] | None) -> None:
    target.chat_messages = _translate_chat_messages(messages)
    target.turns.clear()
    target.seen_msgs = len(messages)
    target.msg_hashes = [_hash(m) for m in messages]
    if tools_schema is not None:
        target.tools_schema = tools_schema


def _extend_chat_messages(target: Chain, messages: list[dict], tools_schema: list[dict] | None) -> None:
    translated = _translate_chat_messages(messages[target.seen_msgs :])
    target.chat_messages.extend(translated)
    target.seen_msgs = len(messages)
    target.msg_hashes = [_hash(m) for m in messages]
    if tools_schema is not None:
        target.tools_schema = tools_schema


def _normalize_openai_obj_for_qwen_chat_template(x):
    """Normalize OpenAI tool-call objects for Qwen3-Coder chat_template.

    Qwen3-Coder chat_template iterates tool_call.arguments with |items,
    so every arguments field that appears in the rendered target must be a dict.
    """
    import json

    def parse_arguments(v):
        if isinstance(v, dict):
            return _normalize_openai_obj_for_qwen_chat_template(v)
        if isinstance(v, str):
            try:
                parsed = json.loads(v) if v.strip() else {}
                if isinstance(parsed, dict):
                    return _normalize_openai_obj_for_qwen_chat_template(parsed)
                return {"value": parsed}
            except Exception:
                return {"value": v}
        if v is None:
            return {}
        return {"value": _normalize_openai_obj_for_qwen_chat_template(v)}

    def parse_parameters(v):
        if isinstance(v, dict):
            y = _normalize_openai_obj_for_qwen_chat_template(v)
        elif isinstance(v, str):
            try:
                y = json.loads(v) if v.strip() else {}
            except Exception:
                y = {}
        else:
            y = {}

        if not isinstance(y, dict):
            y = {}
        y.setdefault("type", "object")
        if not isinstance(y.get("properties", {}), dict):
            y["properties"] = {}
        if "required" in y and not isinstance(y["required"], list):
            y["required"] = []
        return y

    if isinstance(x, list):
        return [_normalize_openai_obj_for_qwen_chat_template(v) for v in x]

    if not isinstance(x, dict):
        return x

    y = {}
    for k, v in x.items():
        if k == "arguments":
            y[k] = parse_arguments(v)
        elif k == "parameters":
            y[k] = parse_parameters(v)
        else:
            y[k] = _normalize_openai_obj_for_qwen_chat_template(v)

    # 只在真正的 message 层把 null content 改成空字符串
    if y.get("role") in ("system", "user", "assistant", "tool") and y.get("content") is None:
        y["content"] = ""


    return y



def _debug_to_jsonable(x, depth=0):
    if depth > 8:
        return repr(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, list):
        return [_debug_to_jsonable(v, depth + 1) for v in x[:50]]
    if isinstance(x, tuple):
        return [_debug_to_jsonable(v, depth + 1) for v in x[:50]]
    if isinstance(x, dict):
        return {str(k): _debug_to_jsonable(v, depth + 1) for k, v in list(x.items())[:100]}
    if hasattr(x, "__dict__"):
        return {
            "__class__": x.__class__.__name__,
            **{str(k): _debug_to_jsonable(v, depth + 1) for k, v in list(vars(x).items())[:100] if not str(k).startswith("_")},
        }
    return repr(x)


def _force_normalize_arguments_inplace(x, seen=None):
    """In-place fallback: convert every dict['arguments'] JSON string to dict.

    This catches cached target.chat_messages, which Qwen chat_template actually renders.
    """
    import json

    if seen is None:
        seen = set()

    oid = id(x)
    if oid in seen:
        return x
    seen.add(oid)

    if isinstance(x, dict):
        if "arguments" in x and isinstance(x["arguments"], str):
            raw = x["arguments"]
            try:
                parsed = json.loads(raw) if raw.strip() else {}
                if isinstance(parsed, dict):
                    x["arguments"] = parsed
                else:
                    x["arguments"] = {"value": parsed}
            except Exception:
                x["arguments"] = {"value": raw}

        for v in list(x.values()):
            _force_normalize_arguments_inplace(v, seen)
        return x

    if isinstance(x, list):
        for v in x:
            _force_normalize_arguments_inplace(v, seen)
        return x

    if isinstance(x, tuple):
        for v in x:
            _force_normalize_arguments_inplace(v, seen)
        return x

    if hasattr(x, "__dict__"):
        for k, v in list(vars(x).items()):
            if str(k).startswith("_"):
                continue
            _force_normalize_arguments_inplace(v, seen)
        return x

    return x


def _dump_openhands_bad_prompt_for_debug(*, body, messages, tools_schema, prompt_messages, prompt_tools_schema, kind, error, prompt_target=None):
    import json
    import os
    import time
    import traceback

    os.makedirs("/tmp/openhands_openai_bad_prompt", exist_ok=True)
    dump_path = f"/tmp/openhands_openai_bad_prompt/bad_{int(time.time() * 1000)}.json"
    with open(dump_path, "w") as f:
        json.dump(
            {
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "kind": kind,
                "prompt_target": _debug_to_jsonable(prompt_target),
                "body": body,
                "messages": messages,
                "tools_schema": tools_schema,
                "prompt_messages": prompt_messages,
                "prompt_tools_schema": prompt_tools_schema,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"[OpenAIAdapter] dumped bad prompt to {dump_path}", flush=True)


def _build_prompt(target: Chain, messages: list[dict], tools_schema: list[dict] | None, kind: str, tok) -> list[int]:
    (_extend_chat_messages if kind == "append" else _replace_chat_messages)(target, messages, tools_schema)
    _force_normalize_arguments_inplace(target)
    return render_token_ids(target, tok)


async def _generate(
    prompt_ids: list[int], s: Session, body: dict, app, *, session_id: str | None = None
) -> TurnRecord:
    return await call_sglang_generate(
        prompt_ids,
        s,
        body,
        app,
        max_token_keys=("max_output_tokens", "max_completion_tokens", "max_tokens"),
        stop_keys=("stop",),
        log_prefix="openai_adapter",
        logger=logger,
        session_id=session_id,
    )


def _extract_qwen_text_tool_uses(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse Qwen-style textual <tool_call> blocks into OpenAIAdapter tool_uses."""
    import re

    if not text or "<tool_call>" not in text:
        return text, []

    tool_uses: list[dict[str, Any]] = []

    block_re = re.compile(
        r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
        re.DOTALL,
    )
    param_re = re.compile(
        r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )

    first_start = None
    for m in block_re.finditer(text):
        if first_start is None:
            first_start = m.start()

        name = m.group(1).strip()
        body = m.group(2)
        args: dict[str, Any] = {}

        for pm in param_re.finditer(body):
            k = pm.group(1).strip()
            v = pm.group(2).strip()
            # Strip common Qwen special token leakage.
            v = v.replace("<|im_end|>", "").strip()
            args[k] = v

        if name:
            tool_uses.append({"name": name, "input": args})

    # Dense Qwen3 chat templates emit JSON directly inside <tool_call>:
    #
    # <tool_call>
    # {"name": "terminal", "arguments": {"command": "pwd"}}
    # </tool_call>
    #
    # This differs from the Qwen3-Coder <function=...> format handled above.
    if not tool_uses:
        import json

        json_block_re = re.compile(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            re.DOTALL,
        )

        for m in json_block_re.finditer(text):
            raw_payload = m.group(1).strip()
            raw_payload = raw_payload.replace("<|im_end|>", "").strip()

            try:
                payload = json.loads(raw_payload)
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue

            name = payload.get("name")
            arguments = payload.get("arguments", {})

            if not isinstance(name, str) or not name.strip():
                continue

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {"value": arguments}

            if not isinstance(arguments, dict):
                arguments = {"value": arguments}

            if first_start is None:
                first_start = m.start()

            tool_uses.append({
                "name": name.strip(),
                "input": arguments,
            })

    if not tool_uses:
        return text, []

    assistant_text = text[:first_start].strip() if first_start is not None else ""
    assistant_text = assistant_text.replace("<|im_end|>", "").strip()
    return assistant_text, tool_uses



def _extract_fenced_json_tool_calls(raw_output: str) -> list[dict]:
    """Extract Qwen-style tool calls emitted as Markdown fenced JSON.

    Example:
        ```json
        {"name": "terminal", "arguments": {"command": "pwd"}}
        ```
    """
    import json
    import re

    if not raw_output:
        return []

    candidates: list[str] = []

    # Prefer explicit Markdown JSON/code fences.
    for match in re.finditer(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        candidates.append(match.group(1))

    # Also tolerate a bare JSON object when the whole response is effectively
    # just a tool call. This intentionally stays conservative.
    stripped = raw_output.strip()
    stripped = stripped.removesuffix("<|im_end|>").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    tool_calls: list[dict] = []
    seen: set[str] = set()

    for text in candidates:
        try:
            payload = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

        payloads = payload if isinstance(payload, list) else [payload]

        for item in payloads:
            if not isinstance(item, dict):
                continue

            name = item.get("name")
            arguments = item.get("arguments", item.get("input", {}))

            if not isinstance(name, str) or not name.strip():
                continue

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue

            if not isinstance(arguments, dict):
                continue

            key = json.dumps(
                {"name": name, "arguments": arguments},
                sort_keys=True,
                ensure_ascii=False,
            )
            if key in seen:
                continue
            seen.add(key)

            tool_calls.append(
                {
                    "name": name,
                    "input": arguments,
                }
            )

    return tool_calls


def _inject_qwen_text_tool_uses(parsed: ParsedModelOutput, raw_output: str) -> ParsedModelOutput:
    """Fallback when parse_model_output does not parse textual Qwen tool calls."""
    if getattr(parsed, "tool_uses", None):
        return parsed

    # First try the native/Qwen <tool_call> textual format.
    text, tool_uses = _extract_qwen_text_tool_uses(raw_output)

    # Qwen2.5-Coder may emit the same JSON payload inside a Markdown
    # code fence instead of <tool_call> tags. Normalize only for parsing.
    # The original raw_output is still retained for raw replay, so this
    # does not change the generated-token prefix used by later turns.
    if not tool_uses:
        import re

        normalized_output = re.sub(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            r"<tool_call>\1</tool_call>",
            raw_output,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if normalized_output != raw_output:
            text, tool_uses = _extract_qwen_text_tool_uses(normalized_output)

    if not tool_uses:
        return parsed

    import dataclasses

    # Frozen dataclass / named dataclass style.
    try:
        if dataclasses.is_dataclass(parsed):
            fields = {f.name for f in dataclasses.fields(parsed)}
            updates = {}
            if "text" in fields:
                updates["text"] = text
            if "tool_uses" in fields:
                updates["tool_uses"] = tool_uses
            if updates:
                return dataclasses.replace(parsed, **updates)
    except Exception:
        pass

    # Mutable object fallback.
    try:
        parsed.text = text
        parsed.tool_uses = tool_uses
        return parsed
    except Exception:
        return parsed



def _raw_output_for_history_replay(raw_output: str) -> str:
    """Return the exact assistant span that should appear in later prompts.

    Dense Qwen3 with enable_thinking=False puts an empty thinking block in the
    generation prompt rather than in output_ids. Historical assistant messages
    do not recreate that block, so prepend it to raw replay text. This changes
    only the model-facing replay; OpenHands still receives standard OpenAI
    tool_calls.
    """
    import os

    disable_thinking = os.environ.get(
        "CODE_AGENT_DISABLE_THINKING", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}

    if not disable_thinking:
        return raw_output

    empty_think = "<think>\n\n</think>\n\n"

    if raw_output.startswith(empty_think):
        return raw_output

    return empty_think + raw_output

def _parse_turn(target: Chain, turn: TurnRecord, app) -> ParsedModelOutput:
    tok = app[TOKENIZER_KEY]
    raw_output = tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
    parsed = parse_model_output(
        raw_output,
        tools_schema=target.tools_schema,
        tool_parser_name=app[TOOL_PARSER_KEY],
        reasoning_parser_name=app[REASONING_PARSER_KEY],
    )
    return _inject_qwen_text_tool_uses(parsed, raw_output)


def _openai_tool_calls(tool_uses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for tool_use in tool_uses:
        call_id = f"call_{secrets.token_hex(12)}"
        name = tool_use.get("name") or tool_use.get("tool_name") or "tool"
        arguments = tool_use.get("input") or {}

        # OpenHands terminal tool schema requires security_risk.
        # Some smaller json-format models emit terminal calls with only {"command": "..."}.
        # Patch it at the adapter boundary so OpenHands validation does not reject the action.
        if name == "terminal" and isinstance(arguments, dict) and "security_risk" not in arguments:
            arguments = dict(arguments)
            arguments["security_risk"] = "LOW"

        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _json_arguments(arguments),
                },
            }
        )
    return calls


def _finish_reason(parsed: ParsedModelOutput, finish: str) -> str:
    if parsed.tool_uses:
        return "tool_calls"
    if finish == "length":
        return "length"
    return "stop"


def _chat_message(parsed: ParsedModelOutput) -> dict[str, Any]:
    tool_calls = _openai_tool_calls(parsed.tool_uses)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": parsed.text if parsed.text else None,
    }
    if parsed.reasoning:
        message["reasoning_content"] = parsed.reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _usage(in_tok: int, out_tok: int) -> dict[str, int]:
    return {
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
    }


def _responses_usage(in_tok: int, out_tok: int) -> dict[str, int]:
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
    }


def _request_session_id(request: web.Request, body: dict) -> str:
    return request_session_id(request, body=body)


async def _run_turn(
    request: web.Request, body: dict, messages: list[dict]
) -> tuple[TurnRecord, ParsedModelOutput, int, int]:
    sid = _request_session_id(request, body)
    adapter = request.app[ADAPTER_KEY]
    if sid in adapter.closed:
        raise web.HTTPServiceUnavailable(text="session closed")
    app = request.app
    s = adapter.store.setdefault(sid, Session())
    task = asyncio.current_task()
    adapter.inflight.setdefault(sid, set()).add(task)
    try:
        async with s.lock:
            target = s.main
            tools_schema = _normalize_tools(body.get("tools"))
            kind = _select_kind(s, messages)

            prompt_target = copy.deepcopy(target)
            prompt_source_messages = _raw_replay_chat_messages(messages, s.raw_outputs)
            prompt_messages = _normalize_openai_obj_for_qwen_chat_template(prompt_source_messages)
            prompt_tools_schema = _normalize_openai_obj_for_qwen_chat_template(tools_schema)
            try:
                prompt_ids = _build_prompt(prompt_target, prompt_messages, prompt_tools_schema, kind, app[TOKENIZER_KEY])
            except Exception as e:
                _dump_openhands_bad_prompt_for_debug(
                    body=body,
                    messages=messages,
                    tools_schema=tools_schema,
                    prompt_messages=prompt_messages,
                    prompt_tools_schema=prompt_tools_schema,
                    kind=kind,
                    error=e,
                    prompt_target=prompt_target,
                )
                raise
            turn = await _generate(prompt_ids, s, body, app, session_id=sid)
            parsed = _parse_turn(target, turn, app)
            if os.environ.get("CODE_AGENT_RAW_REPLAY", "0") == "1":
                raw_output = app[TOKENIZER_KEY].decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
                s.raw_outputs.append(raw_output)
                logger.warning("[adapter_raw_replay] saved raw_output idx=%d chars=%d", len(s.raw_outputs) - 1, len(raw_output))
            target.turns.append(turn)
            return turn, parsed, len(prompt_ids), len(turn.output_ids)
    finally:
        adapter.inflight.get(sid, set()).discard(task)


async def _handle_chat_completions(request: web.Request) -> web.StreamResponse:
    body = await request.json()

    # OpenHands may send a small max_tokens, which truncates long JSON tool calls.
    # For SWE-agent rollouts we need enough room for complete tool_call JSON.
    force_max_tokens = int(os.environ.get("CODE_AGENT_ADAPTER_FORCE_MAX_TOKENS", "0") or "0")
    if force_max_tokens > 0:
        body["max_tokens"] = force_max_tokens
        body["max_completion_tokens"] = force_max_tokens
        body["max_output_tokens"] = force_max_tokens

    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise web.HTTPBadRequest(text="messages must be a list")
    turn, parsed, in_tok, out_tok = await _run_turn(request, body, messages)
    if body.get("stream"):
        return await _stream_chat_completion(request, body, parsed, turn.finish_reason, in_tok, out_tok)
    return web.json_response(_chat_completion_response(body, parsed, turn.finish_reason, in_tok, out_tok))


def _chat_completion_response(
    body: dict,
    parsed: ParsedModelOutput,
    finish: str,
    in_tok: int,
    out_tok: int,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl_{secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "slime-actor"),
        "choices": [
            {
                "index": 0,
                "message": _chat_message(parsed),
                "finish_reason": _finish_reason(parsed, finish),
            }
        ],
        "usage": _usage(in_tok, out_tok),
    }


async def _stream_chat_completion(
    request: web.Request,
    body: dict,
    parsed: ParsedModelOutput,
    finish: str,
    in_tok: int,
    out_tok: int,
) -> web.StreamResponse:
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)
    completion_id = f"chatcmpl_{secrets.token_hex(12)}"
    created = int(time.time())

    async def emit(choice_delta: dict[str, Any], finish_reason: str | None = None, usage: dict | None = None) -> None:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.get("model", "slime-actor"),
            "choices": [{"index": 0, "delta": choice_delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            chunk["usage"] = usage
        await out.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())

    await emit({"role": "assistant"})
    if parsed.reasoning:
        await emit({"reasoning_content": parsed.reasoning})
    if parsed.text:
        await emit({"content": parsed.text})
    for idx, call in enumerate(_openai_tool_calls(parsed.tool_uses)):
        await emit({"tool_calls": [{**call, "index": idx}]})
    await emit({}, finish_reason=_finish_reason(parsed, finish), usage=_usage(in_tok, out_tok))
    await out.write(b"data: [DONE]\n\n")
    return out


async def _handle_responses(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    messages = _responses_input_to_messages(body.get("input", ""), body.get("instructions"))
    turn, parsed, in_tok, out_tok = await _run_turn(request, body, messages)
    if body.get("stream"):
        return await _stream_response(request, body, parsed, turn.finish_reason, in_tok, out_tok)
    return web.json_response(_response_response(body, parsed, turn.finish_reason, in_tok, out_tok))


def _response_output(parsed: ParsedModelOutput) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if parsed.reasoning:
        output.append(
            {
                "id": f"rs_{secrets.token_hex(12)}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": parsed.reasoning}],
            }
        )
    if parsed.text:
        output.append(
            {
                "id": f"msg_{secrets.token_hex(12)}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": parsed.text, "annotations": []}],
            }
        )
    for call in _openai_tool_calls(parsed.tool_uses):
        output.append(
            {
                "id": f"fc_{secrets.token_hex(12)}",
                "type": "function_call",
                "status": "completed",
                "call_id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            }
        )
    if not output:
        output.append(
            {
                "id": f"msg_{secrets.token_hex(12)}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            }
        )
    return output


def _response_response(
    body: dict,
    parsed: ParsedModelOutput,
    finish: str,
    in_tok: int,
    out_tok: int,
) -> dict[str, Any]:
    status = "incomplete" if finish == "length" else "completed"
    return {
        "id": f"resp_{secrets.token_hex(12)}",
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": body.get("model", "slime-actor"),
        "output": _response_output(parsed),
        "usage": _responses_usage(in_tok, out_tok),
    }


async def _stream_response(
    request: web.Request,
    body: dict,
    parsed: ParsedModelOutput,
    finish: str,
    in_tok: int,
    out_tok: int,
) -> web.StreamResponse:
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)
    response = _response_response(body, parsed, finish, in_tok, out_tok)
    created = {"type": "response.created", "response": response}
    await out.write(f"event: response.created\ndata: {json.dumps(created, ensure_ascii=False)}\n\n".encode())
    if parsed.text:
        delta = {"type": "response.output_text.delta", "delta": parsed.text}
        await out.write(
            f"event: response.output_text.delta\ndata: {json.dumps(delta, ensure_ascii=False)}\n\n".encode()
        )
    completed = {"type": "response.completed", "response": response}
    await out.write(f"event: response.completed\ndata: {json.dumps(completed, ensure_ascii=False)}\n\n".encode())
    return out


async def _ok(request: web.Request) -> web.Response:
    return await ok_response(request)
