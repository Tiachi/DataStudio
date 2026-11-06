"""Reject tool protocol use, without banning ordinary questions about hand tools."""

import re

TOOL_FIELDS = {
    "tool",
    "tools",
    "tool_calls",
    "tool_call",
    "tool_call_id",
    "tool_use",
    "tool_result",
    "tool_response",
    "function_call",
    "functions",
}
TOOL_MARKERS = re.compile(
    r"<\s*/?\s*(?:tool(?:_call|_response|_result|_use)?|function_call)\b"
    r"|\[/?TOOL(?:_CALL|_RESULT|_RESPONSE)?\]"
    r"|<\|(?:tool|tool_call|tool_response|python_tag)\|>",
    re.I,
)


def contains_tool(value, *, strict_fields=True):
    """Presence of a protocol field counts even when it is null or an empty list."""
    if type(value) is dict:
        return (
            any(
                str(k).casefold() in TOOL_FIELDS and (strict_fields or bool(v))
                for k, v in value.items()
            )
            or any(
                str(value.get(k, "")).casefold()
                in {
                    "tool",
                    "function",
                    "tool_use",
                    "tool_result",
                    "tool_call",
                    "tool_response",
                    "function_call",
                }
                for k in ("from", "role", "type")
            )
            or any(
                contains_tool(v, strict_fields=strict_fields) for v in value.values()
            )
        )
    if type(value) is list:
        return any(contains_tool(v, strict_fields=strict_fields) for v in value)
    return type(value) is str and bool(TOOL_MARKERS.search(value))
