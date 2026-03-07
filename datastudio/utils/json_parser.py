"""JSON parsing utilities for MLLM responses."""

import json
from typing import Any, Dict, Union

import json_repair


def load_str_to_dict(response_str: str) -> Union[Dict[str, Any], str]:
    """
    Parse MLLM response string to dictionary.

    Tries in order (fast to slow):
    1. Standard json.loads (fastest)
    2. json_repair.loads (handles markdown wrappers, trailing commas,
       single quotes, missing quotes, comments, truncated JSON,
       unbalanced brackets, etc.)

    Args:
        response_str: Response string to parse.

    Returns:
        Parsed dictionary or original string if all parsing fails.
    """
    # Strategy 1: Standard json (fastest path)
    try:
        return json.loads(response_str)
    except Exception:
        pass

    # Strategy 2: json_repair (handles all common LLM output issues)
    try:
        result = json_repair.loads(response_str)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    # All strategies failed
    return response_str
