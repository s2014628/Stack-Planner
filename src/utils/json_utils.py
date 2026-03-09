# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from src.utils.logger import logger
import json
import re
import json_repair


def extract_json_from_text(content: str) -> str:
    """
    Extract JSON from LLM output that may contain <think> tags,
    natural language text, or markdown code blocks around the JSON.

    Args:
        content: Raw LLM output string

    Returns:
        Extracted JSON string
    """
    if not content:
        return content

    # Strip <think>...</think> tags (qwen3 etc.)
    content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()

    # Extract from markdown code blocks (try all code-block pairs)
    for m in re.finditer(r"```(?:json|ts)?\s*\n?([\s\S]*?)```", content):
        extracted = m.group(1).strip()
        if extracted and extracted not in ("```", "``", "`"):
            return extracted

    # Find outermost JSON object {...} via brace matching
    start_idx = content.find("{")
    if start_idx != -1:
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start_idx, len(content)):
            ch = content[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return content[start_idx : i + 1]

    # Last resort: strip stray backticks so callers don't receive bare ```
    cleaned = re.sub(r"^`+|`+$", "", content).strip()
    return cleaned if cleaned else content


def repair_json_output(content: str) -> str:
    """
    Repair and normalize JSON output.

    Args:
        content (str): String content that may contain JSON

    Returns:
        str: Repaired JSON string, or original content if not JSON
    """
    content = content.strip()
    if content.startswith(("{", "[")) or "```json" in content or "```ts" in content:
        try:
            # If content is wrapped in ```json code block, extract the JSON part
            if content.startswith("```json"):
                content = content.removeprefix("```json")

            if content.startswith("```ts"):
                content = content.removeprefix("```ts")

            if content.endswith("```"):
                content = content.removesuffix("```")

            # Try to repair and parse JSON
            repaired_content = json_repair.loads(content)
            return json.dumps(repaired_content, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"JSON repair failed: {e}")
    return content
