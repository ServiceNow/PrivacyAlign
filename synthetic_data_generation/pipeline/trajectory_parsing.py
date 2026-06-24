"""Stateless trajectory parsing and serialization functions."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pipeline.utils import normalize_key, normalize_text, strip_code_fence

_TOOL_USE_EVENT_TYPES = {"tool_use", "tool_call", "toolcall"}
_TOOL_RESULT_EVENT_TYPES = {"tool_result", "tool_output", "tool_response", "result"}
_TOOL_EVENT_TYPES = _TOOL_USE_EVENT_TYPES | _TOOL_RESULT_EVENT_TYPES


def _normalized_event_type(event: Dict[str, Any]) -> str:
    return normalize_key(str(event.get("type", "")))


def extract_json_objects(text: str) -> List[Any]:
    """Best-effort extraction of JSON objects/arrays from mixed text."""
    decoder = json.JSONDecoder()
    objects: List[Any] = []
    idx = 0
    while idx < len(text):
        match = re.search(r"[\{\[]", text[idx:])
        if not match:
            break
        start = idx + match.start()
        try:
            obj, end = decoder.raw_decode(text[start:])
            objects.append(obj)
            idx = start + end
        except json.JSONDecodeError:
            idx = start + 1
    return objects


def step_from_action_dict(item: Dict[str, Any]) -> Optional[Dict[str, str]]:
    action = normalize_text(
        item.get("action", item.get("name", item.get("tool", item.get("tool_name", ""))))
    )
    if not action:
        return None
    action_input = item.get(
        "action_input",
        item.get("arguments", item.get("input", item.get("args", {}))),
    )
    observation = item.get(
        "observation",
        item.get("output", item.get("content", item.get("result", ""))),
    )
    if not isinstance(action_input, str):
        action_input = json.dumps(action_input, ensure_ascii=False)
    if not isinstance(observation, str):
        observation = json.dumps(observation, ensure_ascii=False)
    return {
        "action": action,
        "action_input": action_input,
        "observation": observation,
    }


def parse_tool_events(events: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = []
    idx = 0
    while idx < len(events):
        event = events[idx]
        if not isinstance(event, dict):
            idx += 1
            continue
        event_type = _normalized_event_type(event)
        if event_type not in _TOOL_USE_EVENT_TYPES:
            idx += 1
            continue
        tool_use = event
        tool_use_id = normalize_text(tool_use.get("id", ""))
        tool_result = None

        # Strict pairing: the immediately following event must be the matching tool_result.
        if idx + 1 < len(events):
            follow = events[idx + 1]
            if isinstance(follow, dict):
                follow_type = _normalized_event_type(follow)
                if follow_type in _TOOL_RESULT_EVENT_TYPES:
                    follow_id = normalize_text(follow.get("tool_use_id", ""))
                    if tool_use_id and follow_id and follow_id == tool_use_id:
                        tool_result = follow
                        idx += 1  # consume the paired tool_result

        steps.append(
            {
                "action": normalize_text(tool_use.get("name", "")),
                "action_input": json.dumps(tool_use.get("arguments", tool_use.get("input", {})), ensure_ascii=False),
                "observation": json.dumps(tool_result.get("content", {}), ensure_ascii=False) if tool_result else "",
            }
        )
        idx += 1
    return steps


def parse_steps(executable_trajectory: Any) -> List[Dict[str, str]]:
    """Parse trajectory steps from several possible model-output shapes."""
    if executable_trajectory is None:
        return []

    if isinstance(executable_trajectory, dict):
        if isinstance(executable_trajectory.get("steps"), list):
            parsed_steps: List[Dict[str, str]] = []
            for item in executable_trajectory["steps"]:
                if isinstance(item, dict):
                    step = step_from_action_dict(item)
                    if step:
                        parsed_steps.append(step)
            if parsed_steps:
                return parsed_steps
        if "executable_trajectory" in executable_trajectory:
            return parse_steps(executable_trajectory.get("executable_trajectory"))
        if _normalized_event_type(executable_trajectory) in _TOOL_EVENT_TYPES:
            return parse_tool_events([executable_trajectory])
        step = step_from_action_dict(executable_trajectory)
        return [step] if step else []

    if isinstance(executable_trajectory, list):
        events = []
        for item in executable_trajectory:
            if not isinstance(item, dict):
                continue
            item_type = _normalized_event_type(item)
            if item_type in _TOOL_EVENT_TYPES:
                events.append(item)
        if events:
            steps = parse_tool_events(events)
            if steps:
                return steps
        parsed_steps: List[Dict[str, str]] = []
        for item in executable_trajectory:
            if isinstance(item, dict):
                step = step_from_action_dict(item)
                if step:
                    parsed_steps.append(step)
            elif isinstance(item, str):
                parsed_steps.extend(parse_steps(item))
        return parsed_steps

    if not isinstance(executable_trajectory, str):
        executable_trajectory = str(executable_trajectory)

    text = strip_code_fence(executable_trajectory).strip()
    if not text:
        return []

    # Fast path: one JSON object per line with blank-line separators.
    steps: List[Dict[str, str]] = []
    for block in re.split(r'\n\s*\n', text):
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        tool_use = tool_result = None
        for line in lines:
            try:
                obj = json.loads(line)
                if obj.get("type") == "tool_use":
                    tool_use = obj
                elif obj.get("type") == "tool_result":
                    tool_result = obj
            except json.JSONDecodeError:
                continue
        if tool_use:
            tool_use_id = normalize_text(tool_use.get("id", ""))
            tool_result_id = normalize_text(tool_result.get("tool_use_id", "")) if tool_result else ""
            matched_result = (
                tool_result
                if tool_use_id and tool_result_id and tool_result_id == tool_use_id
                else None
            )
            steps.append({
                "action": tool_use.get("name", ""),
                "action_input": json.dumps(tool_use.get("arguments", tool_use.get("input", {})), ensure_ascii=False),
                "observation": json.dumps(matched_result.get("content", {}), ensure_ascii=False) if matched_result else "",
            })
    if steps:
        return steps

    # Fallback: extract JSON objects from mixed text and parse event stream.
    objects = extract_json_objects(text)
    events: List[Dict[str, Any]] = []
    for obj in objects:
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and _normalized_event_type(item) in _TOOL_EVENT_TYPES:
                    events.append(item)
        elif isinstance(obj, dict):
            if _normalized_event_type(obj) in _TOOL_EVENT_TYPES:
                events.append(obj)
            elif isinstance(obj.get("steps"), list):
                for item in obj["steps"]:
                    if isinstance(item, dict):
                        step = step_from_action_dict(item)
                        if step:
                            steps.append(step)
    if steps:
        return steps
    if events:
        return parse_tool_events(events)
    return steps


def steps_to_executable_trajectory(steps: List[Dict[str, str]]) -> str:
    blocks: List[str] = []
    for idx, step in enumerate(steps, start=1):
        action_input_raw = step.get("action_input", "")
        observation_raw = step.get("observation", "")
        try:
            action_input = json.loads(action_input_raw) if isinstance(action_input_raw, str) and action_input_raw.strip() else {}
        except Exception:
            action_input = {"raw": normalize_text(action_input_raw)}
        try:
            observation = json.loads(observation_raw) if isinstance(observation_raw, str) and observation_raw.strip() else {}
        except Exception:
            observation = {"raw": normalize_text(observation_raw)}
        tool_use = {"type": "tool_use", "id": f"call_{idx}", "name": step.get("action", ""), "arguments": action_input}
        tool_result = {"type": "tool_result", "tool_use_id": f"call_{idx}", "content": observation}
        blocks.append(f"{json.dumps(tool_use, ensure_ascii=False)}\n{json.dumps(tool_result, ensure_ascii=False)}")
    return "\n\n".join(blocks)
