"""ToolkitRegistry: loads toolkits from all_toolkits.json and exposes lookups."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pipeline.utils import normalize_key, normalize_text

LOGGER = logging.getLogger("scenario_generator")


class ToolkitRegistry:
    """Single source of truth for toolkit/action definitions."""

    # Curated allowlist of actions that represent outbound communication tasks.
    FINAL_ACTION_ALLOWLIST = {
        "AmazonPostReview",
        "DiscordSendMessage",
        "FacebookManagerCreateComment",
        "FacebookManagerCreatePost",
        "FacebookManagerSendMessage",
        "FacebookManagerUpdatePost",
        "GitHubPostIssue",
        "GmailSendEmail",
        "JiraAddComment",
        "LinkedInManagerPostUpdate",
        "LinkedInManagerSendMessage",
        "MicrosoftTeamsSendChannelMessage",
        "MicrosoftTeamsSendDirectMessage",
        "Outlook365SendEmail",
        "RedditManagerCreateComment",
        "RedditManagerCreatePost",
        "SignalSendMessage",
        "SlackSendMessage",
        "TeladocLeaveReview",
        "TeladocSendMessage",
        "TwilioSendSms",
        "TwitterManagerPostTweet",
        "TwitterManagerReplyToTweet",
        "WhatsAppSendMessage",
        "ZendeskManagerReplyToTicket",
        "ZoomSendChatMessage",
    }

    def __init__(
        self,
        toolkit_specs_path: Path,
    ) -> None:
        self.toolkit_param_specs: Dict[str, List[Dict[str, Any]]] = {}
        self.toolkit_return_specs: Dict[str, List[Dict[str, Any]]] = {}
        self.toolkit_summaries: Dict[str, str] = {}
        self._load_from_toolkit_specs(toolkit_specs_path)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_from_toolkit_specs(self, path: Path) -> None:
        if not path.exists():
            raise RuntimeError(f"Toolkit specs file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as fh:
                toolkits = json.load(fh)
        except Exception as exc:
            raise RuntimeError(f"Failed to load toolkit specs from {path}: {exc}") from exc

        if not isinstance(toolkits, list):
            raise RuntimeError("Toolkit specs JSON must be a list.")

        self.toolkit_actions: Dict[str, List[str]] = {}
        self.toolkit_aliases: Dict[str, str] = {}
        self.action_to_toolkit: Dict[str, str] = {}
        self.final_action_to_toolkit: Dict[str, str] = {}

        for toolkit in toolkits:
            if not isinstance(toolkit, dict):
                continue
            toolkit_name = normalize_text(toolkit.get("toolkit", ""))
            if not toolkit_name:
                continue

            raw_tools = toolkit.get("tools", [])
            if not isinstance(raw_tools, list):
                continue

            actions: List[str] = []
            for tool in raw_tools:
                if not isinstance(tool, dict):
                    continue
                tool_name = normalize_text(tool.get("name", ""))
                if not tool_name:
                    continue

                raw_params = tool.get("parameters", [])
                if not isinstance(raw_params, list):
                    raw_params = []
                raw_returns = tool.get("returns", [])
                if not isinstance(raw_returns, list):
                    raw_returns = []

                action_name = toolkit_name + tool_name
                actions.append(action_name)
                self.toolkit_param_specs[action_name] = raw_params
                self.toolkit_return_specs[action_name] = raw_returns
                self.action_to_toolkit[action_name] = toolkit_name
                summary = tool.get("summary", "")
                if summary:
                    self.toolkit_summaries[action_name] = summary

            if not actions:
                continue

            self.toolkit_actions[toolkit_name] = actions

            primary_alias = normalize_key(toolkit_name)
            if primary_alias:
                self.toolkit_aliases[primary_alias] = toolkit_name
            if toolkit_name.endswith("Manager") and len(toolkit_name) > len("Manager"):
                stripped = normalize_key(toolkit_name[:-len("Manager")])
                if stripped:
                    self.toolkit_aliases[stripped] = toolkit_name

        self.all_actions_list: List[str] = sorted(self.action_to_toolkit.keys())
        missing_allowlist_actions = sorted(
            action for action in self.FINAL_ACTION_ALLOWLIST
            if action not in self.action_to_toolkit
        )
        if missing_allowlist_actions:
            raise RuntimeError(
                "Final-action allowlist contains actions missing from toolkit specs: "
                + ", ".join(missing_allowlist_actions)
            )
        self.final_action_to_toolkit = {
            action: self.action_to_toolkit[action]
            for action in sorted(self.FINAL_ACTION_ALLOWLIST)
        }
        self.final_actions_list: List[str] = sorted(self.final_action_to_toolkit.keys())

        if not self.toolkit_actions:
            raise RuntimeError("No toolkits loaded from toolkit specs.")
        if not self.final_actions_list:
            raise RuntimeError("No outgoing communication actions available for final_action selection.")

        LOGGER.info(
            "Loaded %d toolkits, %d actions (%d final actions) from %s",
            len(self.toolkit_actions),
            len(self.all_actions_list),
            len(self.final_actions_list),
            path,
        )

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def sample_final_action(self) -> str:
        if not self.final_actions_list:
            raise RuntimeError("No final actions configured in toolkit registry.")
        return random.choice(self.final_actions_list)

    def canonicalize_toolkit(self, value: str) -> Optional[str]:
        cleaned = normalize_text(value).replace(" ", "")
        key = normalize_key(cleaned)
        if key in self.toolkit_aliases:
            return self.toolkit_aliases[key]
        for toolkit in self.toolkit_actions:
            if normalize_key(toolkit) == key:
                return toolkit
        return None

    def canonicalize_toolkit_list(self, raw_toolkits: Any) -> List[str]:
        """Coerce raw toolkit value to list, canonicalize each, and deduplicate."""
        if not isinstance(raw_toolkits, list):
            raw_toolkits = [raw_toolkits]
        toolkits: List[str] = []
        for item in raw_toolkits:
            toolkit = self.canonicalize_toolkit(str(item))
            if toolkit and toolkit not in toolkits:
                toolkits.append(toolkit)
        return toolkits

    def infer_toolkit_from_action(self, action_name: str) -> Optional[str]:
        action_name = normalize_text(action_name)
        if action_name in self.action_to_toolkit:
            return self.action_to_toolkit[action_name]
        for toolkit, action_list in self.toolkit_actions.items():
            if action_name in action_list:
                return toolkit
        for toolkit in self.toolkit_actions:
            if action_name.startswith(toolkit):
                return toolkit
        return None

    def action_specs_for_toolkits(self, toolkits: List[str], final_action: str) -> List[Dict[str, Any]]:
        """Return action specs with parameter and return schemas for the given toolkits.

        Each entry has 'name', 'parameters' (list of params with
        name/type/description/required), and 'returns' (list of return fields
        with name/type/description).  The final_action is excluded since it is
        handled separately by the pipeline.
        """
        specs: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for toolkit in toolkits:
            for action in self.toolkit_actions.get(toolkit, []):
                if action == final_action or action in seen:
                    continue
                seen.add(action)
                raw_params = self.toolkit_param_specs.get(action, [])
                params = [
                    {k: v for k, v in p.items() if k in ("name", "type", "required", "description")}
                    for p in raw_params
                ]
                raw_returns = self.toolkit_return_specs.get(action, [])
                returns = [
                    {k: v for k, v in r.items() if k in ("name", "type", "description")}
                    for r in raw_returns
                ]
                spec: Dict[str, Any] = {"name": action, "parameters": params}
                summary = self.toolkit_summaries.get(action)
                if summary:
                    spec["summary"] = summary
                if returns:
                    spec["returns"] = returns
                specs.append(spec)
        return specs

    def format_action_input_schema(self, action: str) -> Dict[str, Any]:
        """Build a compact JSON schema for an action's input parameters."""
        param_specs: List[Dict[str, Any]] = self.toolkit_param_specs.get(action, [])
        properties: Dict[str, Dict[str, str]] = {}
        required: List[str] = []

        for param in param_specs:
            name = str(param.get("name", "")).strip()
            if not name:
                continue

            prop: Dict[str, str] = {"type": str(param.get("type", "string"))}
            description = param.get("description")
            if description:
                prop["description"] = str(description)
            is_required = bool(param.get("required", False))
            if not is_required:
                prop["optional"] = "true"
            properties[name] = prop

            if is_required:
                required.append(name)

        schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def build_toolkit_descriptions(self, toolkits: List[str]) -> str:
        """Build a human-readable list of toolkit names and their actions."""
        lines: List[str] = []
        for toolkit in toolkits:
            actions = self.toolkit_actions.get(toolkit, [])
            if actions:
                lines.append(f"- **{toolkit}**: {', '.join(actions)}")
            else:
                lines.append(f"- **{toolkit}**")
        return "\n".join(lines) if lines else "(no toolkits)"

    def validate_step_params(
        self, action: str, action_input: str, step_idx: int
    ) -> List[str]:
        """Validate step parameters against toolkit specs.

        Returns list of hard errors (missing required params).
        Type mismatches are logged as warnings only.
        """
        if action not in self.toolkit_param_specs:
            return []

        param_specs = self.toolkit_param_specs[action]
        try:
            parsed_input = json.loads(action_input) if isinstance(action_input, str) else action_input
        except (json.JSONDecodeError, ValueError):
            return []

        if not isinstance(parsed_input, dict):
            return []

        errors: List[str] = []
        for param in param_specs:
            param_name = param.get("name", "")
            is_required = param.get("required", False)
            expected_type = param.get("type", "")

            if is_required and param_name not in parsed_input:
                errors.append(
                    f"step {step_idx + 1}: action {action} missing required param '{param_name}'"
                )
                continue

            if param_name in parsed_input and expected_type:
                value = parsed_input[param_name]
                type_ok = True
                if expected_type == "string":
                    type_ok = isinstance(value, str)
                elif expected_type == "integer":
                    type_ok = isinstance(value, (int, str))
                elif expected_type == "number":
                    type_ok = isinstance(value, (int, float, str))
                elif expected_type == "boolean":
                    type_ok = isinstance(value, (bool, str))
                elif expected_type == "array":
                    type_ok = isinstance(value, (list, str))
                elif expected_type == "object":
                    type_ok = isinstance(value, (dict, str))

                if not type_ok:
                    LOGGER.warning(
                        "Step %d: action %s param '%s' expected type %s, got %s",
                        step_idx + 1, action, param_name, expected_type, type(value).__name__,
                    )

        return errors
