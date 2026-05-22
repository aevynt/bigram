"""Wire schema for tool-using Bigram conversations."""

from dataclasses import dataclass
import json
import re


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ToolCall:
    tool: str
    args: dict


def render_tool_call(tool: str, args: dict) -> str:
    payload = json.dumps({"tool": tool, "args": args}, ensure_ascii=False, separators=(",", ":"))
    return f"<tool_call>{payload}</tool_call>"


def parse_tool_call(text: str) -> ToolCall | None:
    match = TOOL_CALL_RE.search(text or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except (TypeError, json.JSONDecodeError):
        return None
    tool = payload.get("tool")
    args = payload.get("args", {})
    if not isinstance(tool, str) or not isinstance(args, dict):
        return None
    return ToolCall(tool=tool, args=args)


def render_tool_result(text: str) -> str:
    return f"<tool_result>{text}</tool_result>"


def is_abstain(text: str) -> bool:
    lowered = (text or "").lower()
    markers = ["không đủ căn cứ", "khong du can cu", "tôi không chắc", "toi khong chac", "abstain"]
    return any(marker in lowered for marker in markers)
