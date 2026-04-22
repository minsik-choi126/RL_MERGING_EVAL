import json
import os
import re

from bfcl.model_handler.local_inference.qwen import QwenHandler
from bfcl.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)
from overrides import override

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Match <tool_call> with JSON inside; also handle malformed <tool_call></tool_call>\n{...}\n</tool_call>
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*</tool_call>\s*(\{.*?\})\s*</tool_call>"  # malformed: empty open tag
    r"|<tool_call>\s*(\{.*?\})\s*</tool_call>",               # normal
    re.DOTALL,
)


def _strip_think(text: str) -> str:
    """Strip Qwen3-style <think> blocks. Safe no-op for Qwen2.5."""
    return _THINK_RE.sub("", text)


def _extract_tool_call_jsons(text: str) -> list:
    """Extract JSON objects from various tool_call formats.

    Handles:
    1. <tool_call>{"name": ..., "arguments": ...}</tool_call>  (normal)
    2. <tool_call></tool_call>{"name": ...}</tool_call>        (malformed)
    3. [{"name": ..., "arguments": ...}]                        (JSON array, no tags)
    4. {"name": ..., "arguments": ...}                          (bare JSON)
    """
    # Try regex first (handles cases 1 & 2)
    matches = _TOOL_CALL_RE.findall(text)
    jsons = []
    for groups in matches:
        # groups is a tuple from alternation; pick the non-empty one
        raw = groups[0] or groups[1]
        if raw:
            jsons.append(raw)

    if jsons:
        return jsons

    # Try JSON array format: [{"name": ..., "arguments": ...}, ...]
    text_stripped = text.strip()
    if text_stripped.startswith("["):
        try:
            arr = json.loads(text_stripped)
            if isinstance(arr, list) and all(isinstance(x, dict) and "name" in x for x in arr):
                return [json.dumps(x) for x in arr]
        except json.JSONDecodeError:
            pass

    # Try bare JSON: {"name": ..., "arguments": ...}
    if text_stripped.startswith("{"):
        try:
            obj = json.loads(text_stripped)
            if isinstance(obj, dict) and "name" in obj:
                return [text_stripped]
        except json.JSONDecodeError:
            pass

    return []


def _parse_tool_calls(text: str):
    """Parse tool call responses into BFCL's expected decode_ast format.

    Returns list of dicts like [{"func_name": {"arg": val}}, ...].
    Returns None if no tool calls found (fallback to default parser).
    """
    jsons = _extract_tool_call_jsons(text)
    if not jsons:
        return None

    result = []
    for raw in jsons:
        try:
            call = json.loads(raw)
            name = call.get("name", "")
            args = call.get("arguments", {})
            result.append({name: args})
        except (json.JSONDecodeError, AttributeError):
            continue

    return result if result else None


def _tool_calls_to_execute(text: str):
    """Convert tool call responses to Python-style execution strings.

    Returns list of strings like ["func_name(arg=val, ...)", ...].
    Returns None if no tool calls found.
    """
    jsons = _extract_tool_call_jsons(text)
    if not jsons:
        return None

    result = []
    for raw in jsons:
        try:
            call = json.loads(raw)
            name = call.get("name", "")
            args = call.get("arguments", {})
            args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
            result.append(f"{name}({args_str})")
        except (json.JSONDecodeError, AttributeError):
            continue

    return result if result else None


class QwenGenericHandler(QwenHandler):
    """Generic handler that reads model path from BFCL_MODEL_PATH env var.

    Env vars:
        BFCL_MODEL_PATH  — (required) local path to model
        BFCL_DTYPE       — vLLM serve dtype (default: bfloat16)
        BFCL_PROMPT_MODE — "tool" uses Qwen tool-calling template with <tools>;
                           "plain" uses simple ChatML (default: tool)
    """

    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.dtype = os.environ.get("BFCL_DTYPE", "bfloat16")
        self.prompt_mode = os.environ.get("BFCL_PROMPT_MODE", "tool")
        model_path = os.environ.get("BFCL_MODEL_PATH")
        if not model_path:
            raise RuntimeError(
                "BFCL_MODEL_PATH env var must be set. "
                "Example: BFCL_MODEL_PATH=/path/to/model bash run_eval.sh"
            )
        self.model_name_huggingface = model_path

    @override
    def _format_prompt(self, messages, function):
        if self.prompt_mode == "plain" or not function:
            return super()._format_prompt(messages, function)

        # Qwen2.5 tool-calling format
        system_content = ""
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)

        if not system_content:
            system_content = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

        tools_block = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        tools_block += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        for func in function:
            tools_block += "\n" + json.dumps(func)
        tools_block += "\n</tools>\n\n"
        tools_block += (
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

        prompt = f"<|im_start|>system\n{system_content}{tools_block}<|im_end|>\n"
        for msg in user_messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    @override
    def decode_ast(self, result, language="Python"):
        result = _strip_think(result)
        # Try tool_call format first (XML tags, JSON array, bare JSON)
        parsed = _parse_tool_calls(result)
        if parsed is not None:
            return parsed
        # Fallback to default Python-style parser
        return default_decode_ast_prompting(result, language)

    @override
    def decode_execute(self, result):
        result = _strip_think(result)
        # Try tool_call format first
        parsed = _tool_calls_to_execute(result)
        if parsed is not None:
            return parsed
        # Fallback to default parser
        return default_decode_execute_prompting(result)
