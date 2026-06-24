"""PipelineContext: shared state passed to all pipeline stage functions."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from string import Template
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pipeline.utils import NameRow, StageGenerationError, parse_json_loose

if TYPE_CHECKING:
    from pipeline.diversity import DiversityTracker
    from pipeline.toolkit_registry import ToolkitRegistry

LOGGER = logging.getLogger("scenario_generator")


@dataclass
class PipelineContext:
    """Shared state passed to all stage functions."""

    args: argparse.Namespace
    registry: ToolkitRegistry
    prompt_templates: Dict[str, Template]
    seed_options: Dict[str, Any]
    model: Optional[Any]
    diversity: DiversityTracker
    name_rows: List[NameRow]

    def _ensure_model(self) -> Any:
        if self.model is None:
            raise StageGenerationError("Model is unavailable.")
        return self.model

    def _log_prompt(self, prompt: str) -> None:
        if self.args.print_prompts:
            LOGGER.info("Prompt:\n%s", prompt)

    def _log_batch_prompts(self, prompts: List[str]) -> None:
        if self.args.print_prompts:
            for idx, prompt in enumerate(prompts):
                LOGGER.info("Batch prompt [%d]:\n%s", idx, prompt)

    @staticmethod
    def _clean_text_response(response: Any) -> Optional[str]:
        if isinstance(response, str):
            stripped = response.strip()
            if stripped:
                return stripped
        return None

    def _resolve_temperature(self, filter: bool = False) -> float:
        """Resolve the sampling temperature for generation vs filter calls."""
        if filter:
            return float(getattr(self.args, "filter_temperature", 1.0))
        return float(getattr(self.args, "diverse_generation_temperature", 1.0))

    def _resolve_reasoning_effort(self, filter: bool = False) -> Optional[str]:
        """Resolve the reasoning effort for generation vs filter calls.

        Returns a per-call override string, or None to use the model default.
        """
        if filter:
            override = getattr(self.args, "filter_reasoning_effort", None)
            if override is not None:
                return override
        # For generation stages, return the base reasoning_effort only if a
        # filter override exists (so generation stages use the explicit value
        # instead of the model default which is the same thing).
        # When no filter override is set, return None to let the model default apply.
        if getattr(self.args, "filter_reasoning_effort", None) is not None:
            return getattr(self.args, "reasoning_effort", None)
        return None

    def _resolve_sampling_kwargs(self, filter: bool = False) -> Dict[str, Any]:
        """Resolve top_p/top_k kwargs — only non-default for filter stages."""
        if filter:
            top_p = float(getattr(self.args, "filter_top_p", 1.0))
            top_k = int(getattr(self.args, "filter_top_k", -1))
            kwargs: Dict[str, Any] = {}
            if top_p != 1.0:
                kwargs["top_p"] = top_p
            if top_k != -1:
                kwargs["top_k"] = top_k
            return kwargs
        return {}

    def render_prompt(self, template_name: str, **kwargs: Any) -> str:
        template = self.prompt_templates.get(template_name)
        if template is None:
            raise StageGenerationError(f"Prompt template not loaded: {template_name}")
        rendered_vars: Dict[str, str] = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                rendered_vars[key] = value
            elif isinstance(value, (dict, list)):
                rendered_vars[key] = json.dumps(value, indent=2)
            else:
                rendered_vars[key] = str(value)
        return template.safe_substitute(**rendered_vars)

    def call_json(self, prompt: str, max_tokens: int, filter: bool = False) -> Any:
        model = self._ensure_model()
        self._log_prompt(prompt)
        temperature = self._resolve_temperature(filter=filter)
        sampling_kwargs = self._resolve_sampling_kwargs(filter=filter)
        reasoning_effort = self._resolve_reasoning_effort(filter=filter)
        response = model.interact(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            reasoning_effort=reasoning_effort,
            **sampling_kwargs,
        )
        if isinstance(response, str):
            cleaned = response.strip()
            if not cleaned:
                raise ValueError("Model returned empty text response.")
            response = cleaned
        return parse_json_loose(response)

    def call_text(self, prompt: str, max_tokens: int, filter: bool = False) -> str:
        """Call the model and return raw text (no JSON parsing)."""
        model = self._ensure_model()
        self._log_prompt(prompt)
        temperature = self._resolve_temperature(filter=filter)
        sampling_kwargs = self._resolve_sampling_kwargs(filter=filter)
        reasoning_effort = self._resolve_reasoning_effort(filter=filter)
        response = model.interact(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            **sampling_kwargs,
        )
        cleaned = self._clean_text_response(response)
        if cleaned is not None:
            return cleaned
        raise ValueError("Model returned empty text response.")

    def batch_call_text(
        self,
        prompts: List[str],
        max_tokens: int,
        filter: bool = False,
        enable_thinking: Optional[bool] = None,
    ) -> List[Optional[str]]:
        """Call the model with multiple prompts in parallel and return raw text."""
        model = self._ensure_model()
        self._log_batch_prompts(prompts)
        temperature = self._resolve_temperature(filter=filter)
        sampling_kwargs = self._resolve_sampling_kwargs(filter=filter)
        reasoning_effort = self._resolve_reasoning_effort(filter=filter)
        responses = model.batch_interact(
            prompts=prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            **sampling_kwargs,
        )
        results: List[Optional[str]] = []
        for resp in responses:
            results.append(self._clean_text_response(resp))
        return results

    def batch_call_json(
        self,
        prompts: List[str],
        max_tokens: int,
        filter: bool = False,
        enable_thinking: Optional[bool] = None,
    ) -> List[Optional[Any]]:
        """Call the model with multiple prompts in parallel and JSON-parse each response."""
        model = self._ensure_model()
        self._log_batch_prompts(prompts)
        temperature = self._resolve_temperature(filter=filter)
        sampling_kwargs = self._resolve_sampling_kwargs(filter=filter)
        reasoning_effort = self._resolve_reasoning_effort(filter=filter)
        responses = model.batch_interact(
            prompts=prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            **sampling_kwargs,
        )
        results: List[Optional[Any]] = []
        for resp in responses:
            cleaned = self._clean_text_response(resp)
            if cleaned is not None:
                try:
                    results.append(parse_json_loose(cleaned))
                except Exception:
                    results.append(None)
            else:
                results.append(None)
        return results
