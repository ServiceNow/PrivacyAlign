#!/usr/bin/env python3
"""
Mine pre-generated samples for privacy-handling differences across model pairs.

For each input sample, this script:
1. Generates fresh naive-agent actions with each response model, reusing the
   stored generated final action when it already came from that same model.
2. Runs sensibility and comparative judging with each judge model, including
   sensibility checks on reused source actions from the originating model.
3. Requires majority sensibility approval before a response can participate in
   pair selection.
4. Records every evaluated response pair in the output, even when comparative
   judges disagree or do not unanimously identify a leaking response, and
   preserves a majority-supported representative comparative output when one
   exists.
5. Requires comparative-judge unanimity that at least one response in the pair
   leaks privacy-sensitive information before a pair can be selected.
6. Requires comparative-judge unanimity on which response is better before a
   pair can be selected.
7. Keeps at most one (pair, primary_judge) selection per sample, preferring
   pairs whose responses are unanimously judged sensible before judge and
   pair-family diversity.

The pipeline is deliberately sequential by model to avoid repeated vLLM reloads:
all generation for model A happens together, then all generation for model B,
then all judging for judge A, and so on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pipeline.context import PipelineContext
from pipeline.diversity import DiversityTracker
from pipeline.leakage_filter import (
    build_naive_agent_prompt,
    build_sensibility_check_prompt,
    parse_naive_agent_result,
    parse_sensibility_check_result,
)
from pipeline.toolkit_registry import ToolkitRegistry
from pipeline.utils import normalize_text, write_json_atomic

LOGGER = logging.getLogger("mine_samples")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_DIR = SCRIPT_DIR / "resources" / "prompts"
DEFAULT_TOOLKIT_SPECS_PATH = SCRIPT_DIR / "resources" / "assets" / "all_toolkits.json"
COMPARATIVE_JUDGE_TEMPLATE = "comparative_judge_prompt.txt"
REQUIRED_PROMPT_TEMPLATE_FILES = (
    "naive_agent_prompt.txt",
    "sensibility_check_prompt.txt",
    COMPARATIVE_JUDGE_TEMPLATE,
)

GPT_OSS_MODEL = "openai/gpt-oss-120b"
NVIDIA_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
STEP_MODEL = "stepfun-ai/Step-3.5-Flash-FP8"
QWEN_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"

DEFAULT_RESPONSE_MODELS = [GPT_OSS_MODEL, NVIDIA_MODEL, QWEN_MODEL]

MODEL_LABELS = {
    GPT_OSS_MODEL: "gpt",
    NVIDIA_MODEL: "nvidia",
    STEP_MODEL: "step",
    QWEN_MODEL: "qwen",
}

MODEL_NAME_ALIASES = {
    "gpt-oss-120b": GPT_OSS_MODEL,
    "openai/gpt-oss-120b": GPT_OSS_MODEL,
    "nvidia/nvidia-nemotron-3-super-120b-a12b-fp8": NVIDIA_MODEL,
    "nvidia-nemotron-3-super-120b-a12b-fp8": NVIDIA_MODEL,
    "nemotron-3-super-120b-a12b-fp8": NVIDIA_MODEL,
    "step-3.5-flash-fp8": STEP_MODEL,
    "stepfun-ai/step-3.5-flash-fp8": STEP_MODEL,
    "qwen3.5-397b-a17b-fp8": QWEN_MODEL,
    "qwen/qwen3.5-397b-a17b-fp8": QWEN_MODEL,
}

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    GPT_OSS_MODEL: {
        "reasoning_effort": "high",
        "temperature": 1.0,
    },
    NVIDIA_MODEL: {
        "language_model_only": True,
        "top_p": 0.95,
        "temperature": 1.0,
    },
    STEP_MODEL: {
        "enable_expert_parallel": True,
        "language_model_only": True,
        "top_p": 0.95,
        "temperature": 1.0,
    },
    QWEN_MODEL: {
        "language_model_only": True,
        "repetition_penalty": 1.0,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    },
}


def reconstruct_inputs(
    sample: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Convert a stored sample into the prompt-builder inputs."""
    trajectory = sample["trajectory"]
    seed = sample["seed"]
    trajectory_payload = {"trajectory": trajectory}

    raw_domains = (
        sample.get("generation_metadata", {}).get("domains")
        or sample.get("seed", {}).get("domains")
    )
    if not raw_domains:
        legacy_domain = (
            sample.get("generation_metadata", {}).get("scenario_domain", "")
            or sample.get("seed", {}).get("scenario_domain", "")
        )
        raw_domains = [legacy_domain] if legacy_domain else []

    seed_candidate = {
        "data_subject": seed["data_subject"],
        "data_sender": seed["data_sender"],
        "data_recipient": seed["data_recipient"],
        "final_action": trajectory["final_action"],
        "toolkits": trajectory["toolkits"],
        "domains": raw_domains,
    }

    vignette = {
        **sample["vignette"],
        "memories": sample.get("memories", []),
    }
    return trajectory_payload, seed_candidate, vignette


def build_comparative_judge_prompt(
    ctx: Any,
    trajectory_payload: Dict[str, Any],
    seed_candidate: Dict[str, Any],
    vignette: Dict[str, Any],
    agent_a_action: str,
    agent_b_action: str,
) -> str:
    trajectory = trajectory_payload["trajectory"]
    return ctx.render_prompt(
        COMPARATIVE_JUDGE_TEMPLATE,
        story=vignette.get("story", ""),
        data_subject=vignette.get("data_subject_concrete", seed_candidate.get("data_subject", "")),
        data_sender=vignette.get("data_sender_concrete", seed_candidate.get("data_sender", "")),
        data_recipient=vignette.get("data_recipient_concrete", seed_candidate.get("data_recipient", "")),
        user_name=trajectory["user_name"],
        user_instruction=trajectory["user_instruction"],
        original_action=agent_a_action,
        new_action=agent_b_action,
    )


def _strip_reasoning(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    return cleaned


_COMPARATIVE_AGENT_A_HEADER_RE = r"(?:(?:ASSISTANT|AGENT)\s*_?\s*A|A)"
_COMPARATIVE_AGENT_B_HEADER_RE = r"(?:(?:ASSISTANT|AGENT)\s*_?\s*B|B)"
_COMPARATIVE_COMPARISON_HEADER_RE = r"(?:COMPARISON)"
_COMPARATIVE_EXPLANATION_HEADER_RE = r"(?:EXPLANATION)"
_COMPARATIVE_RESPONSE_A_LEAKS_HEADER_RE = r"(?:RESPONSE\s*A\s*LEAKS)"
_COMPARATIVE_RESPONSE_A_LEAKS_DETAILS_HEADER_RE = r"(?:RESPONSE\s*A\s*LEAKS\s*DETAILS)"
_COMPARATIVE_RESPONSE_A_OMITS_HEADER_RE = r"(?:RESPONSE\s*A\s*OMITS)"
_COMPARATIVE_RESPONSE_A_OMITS_DETAILS_HEADER_RE = r"(?:RESPONSE\s*A\s*OMITS\s*DETAILS)"
_COMPARATIVE_RESPONSE_B_LEAKS_HEADER_RE = r"(?:RESPONSE\s*B\s*LEAKS)"
_COMPARATIVE_RESPONSE_B_LEAKS_DETAILS_HEADER_RE = r"(?:RESPONSE\s*B\s*LEAKS\s*DETAILS)"
_COMPARATIVE_RESPONSE_B_OMITS_HEADER_RE = r"(?:RESPONSE\s*B\s*OMITS)"
_COMPARATIVE_RESPONSE_B_OMITS_DETAILS_HEADER_RE = r"(?:RESPONSE\s*B\s*OMITS\s*DETAILS)"
_COMPARATIVE_PREFERENCE_HEADER_RE = r"(?:PREFERENCE|VERDICT|ANSWER|FINAL\s+ANSWER)"
_COMPARATIVE_ANY_HEADER_RE = (
    rf"(?:{_COMPARATIVE_AGENT_A_HEADER_RE}|{_COMPARATIVE_AGENT_B_HEADER_RE}|"
    rf"{_COMPARATIVE_COMPARISON_HEADER_RE}|{_COMPARATIVE_EXPLANATION_HEADER_RE}|"
    rf"{_COMPARATIVE_RESPONSE_A_LEAKS_HEADER_RE}|"
    rf"{_COMPARATIVE_RESPONSE_A_LEAKS_DETAILS_HEADER_RE}|"
    rf"{_COMPARATIVE_RESPONSE_A_OMITS_HEADER_RE}|"
    rf"{_COMPARATIVE_RESPONSE_A_OMITS_DETAILS_HEADER_RE}|"
    rf"{_COMPARATIVE_RESPONSE_B_LEAKS_HEADER_RE}|"
    rf"{_COMPARATIVE_RESPONSE_B_LEAKS_DETAILS_HEADER_RE}|"
    rf"{_COMPARATIVE_RESPONSE_B_OMITS_HEADER_RE}|"
    rf"{_COMPARATIVE_RESPONSE_B_OMITS_DETAILS_HEADER_RE}|"
    rf"{_COMPARATIVE_PREFERENCE_HEADER_RE})"
)
_COMPARATIVE_HEADER_SUFFIX_RE = (
    r"(?:\*{0,2}\s*:\s*\*{0,2}|:\s*\*{0,2}|\*{0,2}\s*:|:|\*{1,2})?"
)


def _comparative_header_pattern(label: str) -> str:
    normalized = normalize_text(label).upper()
    if normalized in {"ASSISTANT A", "AGENT A", "A"}:
        return _COMPARATIVE_AGENT_A_HEADER_RE
    if normalized in {"ASSISTANT B", "AGENT B", "B"}:
        return _COMPARATIVE_AGENT_B_HEADER_RE
    if normalized == "COMPARISON":
        return _COMPARATIVE_COMPARISON_HEADER_RE
    if normalized == "EXPLANATION":
        return _COMPARATIVE_EXPLANATION_HEADER_RE
    if normalized in {"RESPONSE A LEAKS", "A LEAKS"}:
        return _COMPARATIVE_RESPONSE_A_LEAKS_HEADER_RE
    if normalized in {"RESPONSE A LEAKS DETAILS", "A LEAKS DETAILS"}:
        return _COMPARATIVE_RESPONSE_A_LEAKS_DETAILS_HEADER_RE
    if normalized in {"RESPONSE A OMITS", "A OMITS"}:
        return _COMPARATIVE_RESPONSE_A_OMITS_HEADER_RE
    if normalized in {"RESPONSE A OMITS DETAILS", "A OMITS DETAILS"}:
        return _COMPARATIVE_RESPONSE_A_OMITS_DETAILS_HEADER_RE
    if normalized in {"RESPONSE B LEAKS", "B LEAKS"}:
        return _COMPARATIVE_RESPONSE_B_LEAKS_HEADER_RE
    if normalized in {"RESPONSE B LEAKS DETAILS", "B LEAKS DETAILS"}:
        return _COMPARATIVE_RESPONSE_B_LEAKS_DETAILS_HEADER_RE
    if normalized in {"RESPONSE B OMITS", "B OMITS"}:
        return _COMPARATIVE_RESPONSE_B_OMITS_HEADER_RE
    if normalized in {"RESPONSE B OMITS DETAILS", "B OMITS DETAILS"}:
        return _COMPARATIVE_RESPONSE_B_OMITS_DETAILS_HEADER_RE
    if normalized in {"PREFERENCE", "VERDICT", "ANSWER", "FINAL ANSWER"}:
        return _COMPARATIVE_PREFERENCE_HEADER_RE
    return re.escape(label)


def _clean_comparative_verdict_candidate(text: str) -> str:
    candidate = text.strip()
    candidate = re.sub(r"^[\-\*\u2022]+\s*", "", candidate)
    candidate = candidate.splitlines()[0].strip()
    candidate = candidate.strip("`*_ \t")
    candidate = re.sub(r"\s+", " ", candidate)
    candidate = candidate.rstrip(".").strip()
    return candidate


def extract_comparative_judge_section(raw_text: str, label: str) -> Optional[str]:
    cleaned = _strip_reasoning(raw_text)
    header_pattern = _comparative_header_pattern(label)
    section_re = re.compile(
        rf"(?ims)^\s*\*{{0,2}}\s*{header_pattern}\s*{_COMPARATIVE_HEADER_SUFFIX_RE}\s*"
        r"(.*?)"
        rf"(?=^\s*\*{{0,2}}\s*(?:{_COMPARATIVE_ANY_HEADER_RE})\s*"
        rf"{_COMPARATIVE_HEADER_SUFFIX_RE}\s*|\Z)"
    )
    match = section_re.search(cleaned)
    if not match:
        return None
    section = match.group(1).strip()
    return section or None


def _interpret_comparative_verdict(verdict: str) -> Tuple[Optional[bool], Optional[str]]:
    normalized = normalize_text(verdict).upper()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if normalized in {"SAME", "TIE", "EQUAL"}:
        return False, None
    if normalized in {"A", "AGENT A", "ASSISTANT A"}:
        return True, "A"
    if normalized in {"B", "AGENT B", "ASSISTANT B"}:
        return True, "B"

    if re.search(r"\b(?:SAME|TIE|EQUAL|EQUIVALENT)\b", normalized):
        return False, None

    if re.search(
        r"\b(?:A|AGENT A|ASSISTANT A)\b.*\b(?:BETTER|WINS|WINNER|PREFERRED|BEST)\b",
        normalized,
    ):
        return True, "A"
    if re.search(
        r"\b(?:B|AGENT B|ASSISTANT B)\b.*\b(?:BETTER|WINS|WINNER|PREFERRED|BEST)\b",
        normalized,
    ):
        return True, "B"

    if re.search(
        r"\b(?:A|AGENT A|ASSISTANT A)\b.*\bLEAKS\b.*\b(?:WORSE|MORE)\b",
        normalized,
    ):
        return True, "B"
    if re.search(
        r"\b(?:B|AGENT B|ASSISTANT B)\b.*\bLEAKS\b.*\b(?:WORSE|MORE)\b",
        normalized,
    ):
        return True, "A"

    return None, None


def _interpret_comparative_boolean(value: str) -> Optional[bool]:
    normalized = normalize_text(value).upper()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in {"TRUE", "YES"}:
        return True
    if normalized in {"FALSE", "NO"}:
        return False
    return None


def extract_comparative_judge_boolean(raw_text: str, label: str) -> Optional[bool]:
    cleaned = _strip_reasoning(raw_text)
    header_pattern = _comparative_header_pattern(label)
    bool_re = re.compile(
        rf"(?im)^\s*\*{{0,2}}\s*{header_pattern}\s*{_COMPARATIVE_HEADER_SUFFIX_RE}\s*(true|false|yes|no)\b"
    )
    matches = list(bool_re.finditer(cleaned))
    if matches:
        return _interpret_comparative_boolean(matches[-1].group(1))

    section = extract_comparative_judge_section(cleaned, label)
    if section:
        return _interpret_comparative_boolean(_clean_comparative_verdict_candidate(section))
    return None


def parse_comparative_judge_result(
    raw_text: str,
) -> Tuple[Optional[bool], Optional[str], Optional[str]]:
    """Return (differ, better_agent, matched_verdict)."""
    cleaned = _strip_reasoning(raw_text)
    verdict_candidates: List[str] = []

    for label in ("PREFERENCE", "VERDICT"):
        verdict_section = extract_comparative_judge_section(cleaned, label)
        if verdict_section:
            verdict_candidates.append(_clean_comparative_verdict_candidate(verdict_section))

    verdict_patterns = [
        re.compile(
            rf"(?im)^\s*\*{{0,2}}\s*PREFERENCE\s*{_COMPARATIVE_HEADER_SUFFIX_RE}\s*([^\n\r]+)"
        ),
        re.compile(
            rf"(?im)^\s*\*{{0,2}}\s*VERDICT\s*{_COMPARATIVE_HEADER_SUFFIX_RE}\s*([^\n\r]+)"
        ),
        re.compile(
            rf"(?im)^\s*\*{{0,2}}\s*(?:Final\s+)?Answer\s*{_COMPARATIVE_HEADER_SUFFIX_RE}\s*([^\n\r]+)"
        ),
        re.compile(
            r"(?i)(?:answer|preference|verdict|judgment|conclusion|decision|winner|preferred|best\s+overall)\s*"
            r"(?:is|:)\s*\*{0,2}\s*([^\n\r]+)"
        ),
    ]

    matches = []
    for pattern in verdict_patterns:
        matches = list(pattern.finditer(cleaned))
        if matches:
            verdict_candidates.extend(
                _clean_comparative_verdict_candidate(match.group(1))
                for match in matches
            )
            break

    for verdict in verdict_candidates:
        if not verdict:
            continue
        differ, better_agent = _interpret_comparative_verdict(verdict)
        if differ is not None:
            return differ, better_agent, verdict

    if not verdict_candidates:
        LOGGER.warning("Could not parse comparative judge verdict: %s", cleaned[:300])
        return None, None, None

    LOGGER.warning("Could not interpret comparative judge verdict: %s", verdict_candidates[-1])
    return None, None, verdict_candidates[-1]


def canonicalize_model_name(model_name: str) -> str:
    normalized = normalize_text(model_name)
    if not normalized:
        raise ValueError("Model name cannot be empty.")
    return MODEL_NAME_ALIASES.get(normalized.lower(), normalized)


def extract_source_model_name(sample: Dict[str, Any]) -> Optional[str]:
    metadata = sample.get("generation_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    candidates = (
        sample.get("source_model_name"),
        sample.get("model_name"),
        metadata.get("combined_source_model_name"),
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return canonicalize_model_name(candidate)
        except ValueError:
            continue
    return None


def extract_source_generated_action(
    sample: Dict[str, Any],
    *,
    source_model_name: Optional[str],
) -> str:
    trajectory = sample.get("trajectory", {})
    if isinstance(trajectory, dict):
        raw_action = trajectory.get("generated_final_action", "")
        if isinstance(raw_action, str) and raw_action.strip():
            return raw_action

    raw_action = sample.get("source_generated_action_raw", "")
    if isinstance(raw_action, str) and raw_action.strip():
        return raw_action

    responses = sample.get("responses", {})
    if not isinstance(responses, dict):
        return ""

    for response_key, response in responses.items():
        if not isinstance(response, dict):
            continue
        action = response.get("action", "")
        if not isinstance(action, str) or not action.strip():
            continue

        if response.get("reused_source_action") is True:
            return action

        if source_model_name is None:
            continue

        try:
            canonical_response_key = canonicalize_model_name(str(response_key))
        except ValueError:
            continue
        if canonical_response_key == source_model_name:
            return action

    return ""


def unique_models(model_names: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for name in model_names:
        canonical = canonicalize_model_name(name)
        if canonical in seen:
            continue
        seen.add(canonical)
        ordered.append(canonical)
    return ordered


def model_label(model_name: str) -> str:
    if model_name in MODEL_LABELS:
        return MODEL_LABELS[model_name]
    normalized = normalize_text(model_name)
    if "/" in normalized:
        normalized = normalized.split("/")[-1]
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized or "model"


def stable_order_token(seed: int, name: str) -> str:
    payload = f"{seed}:{name}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


class SampleMiner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        random.seed(args.seed)

        self.response_models = unique_models(args.response_models)
        judge_models = args.judge_models or self.response_models
        self.judge_models = unique_models(judge_models)
        self._response_model_index = {
            model_name: idx for idx, model_name in enumerate(self.response_models)
        }

        self.registry = ToolkitRegistry(
            toolkit_specs_path=Path(args.toolkit_specs_path),
        )
        self.prompt_templates = self._load_prompt_templates(Path(args.prompts_dir))

        self._model_cfgs = {
            model_name: self._build_model_cfg(model_name)
            for model_name in unique_models(self.response_models + self.judge_models)
        }

        self._active_model = None
        self._active_model_name: Optional[str] = None

        self.ctx = PipelineContext(
            args=self._make_ctx_args(self.response_models[0]),
            registry=self.registry,
            prompt_templates=self.prompt_templates,
            seed_options={},
            model=None,
            diversity=DiversityTracker(),
            name_rows=[],
        )

        self._completed_response_models: List[str] = []
        self._completed_sensibility_judge_models: List[str] = []
        self._completed_comparative_judge_models: List[str] = []
        self._selection_complete = False

    def _build_model_cfg(self, model_name: str) -> Dict[str, Any]:
        speculative_config = None
        if self.args.speculative_config:
            speculative_config = json.loads(self.args.speculative_config)

        cfg: Dict[str, Any] = {
            "model_name": model_name,
            "backend": self.args.backend,
            "tensor_parallel_size": self.args.vllm_tensor_parallel_size,
            "pipeline_parallel_size": self.args.vllm_pipeline_parallel_size,
            "enable_expert_parallel": self.args.vllm_enable_expert_parallel,
            "enforce_eager": self.args.vllm_enforce_eager,
            "language_model_only": self.args.vllm_language_model_only,
            "enable_prefix_caching": self.args.vllm_enable_prefix_caching,
            "gpu_memory_utilization": self.args.vllm_gpu_memory_utilization,
            "max_model_len": self.args.vllm_max_model_len,
            "kv_cache_dtype": self.args.vllm_kv_cache_dtype,
            "presence_penalty": self.args.presence_penalty,
            "repetition_penalty": self.args.repetition_penalty,
            "hf_cache_dir": self.args.hf_cache_dir,
            "reasoning_effort": self.args.reasoning_effort,
            "enable_thinking": self.args.enable_thinking,
            "speculative_config": speculative_config,
        }
        if not self.args.disable_model_presets:
            for key, value in MODEL_PRESETS.get(model_name, {}).items():
                if key in cfg:
                    cfg[key] = value
        return cfg

    def _make_ctx_args(self, model_name: str) -> argparse.Namespace:
        preset = MODEL_PRESETS.get(model_name, {}) if not self.args.disable_model_presets else {}
        temperature = float(preset.get("temperature", self.args.temperature))
        top_p = float(preset.get("top_p", self.args.top_p))
        top_k = int(preset.get("top_k", self.args.top_k))
        reasoning_effort = preset.get("reasoning_effort", self.args.reasoning_effort)
        fields = {
            "print_prompts": self.args.print_prompts,
            "diverse_generation_temperature": temperature,
            "filter_temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "filter_reasoning_effort": reasoning_effort,
            "filter_top_p": top_p,
            "filter_top_k": top_k,
        }
        if preset:
            for key, value in preset.items():
                if key in fields:
                    fields[key] = value
        return argparse.Namespace(**fields)

    def _load_prompt_templates(self, prompts_dir: Path) -> Dict[str, Template]:
        templates: Dict[str, Template] = {}
        missing: List[str] = []
        for filename in REQUIRED_PROMPT_TEMPLATE_FILES:
            path = prompts_dir / filename
            if not path.exists():
                missing.append(str(path))
                continue
            with path.open("r", encoding="utf-8") as handle:
                templates[filename] = Template(handle.read())

        if missing:
            raise RuntimeError("Missing prompt template files:\n" + "\n".join(missing))
        return templates

    def _unload_model(self) -> None:
        if self._active_model is None:
            return
        LOGGER.info("Unloading model: %s", self._active_model_name)
        if hasattr(self._active_model, "close"):
            try:
                self._active_model.close()
            except Exception:
                pass
        self._active_model = None
        self._active_model_name = None
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    def _activate_model(self, model_name: str) -> None:
        if self._active_model_name == model_name and self._active_model is not None:
            self.ctx.args = self._make_ctx_args(model_name)
            self.ctx.model = self._active_model
            return

        self._unload_model()
        cfg = self._model_cfgs[model_name]
        LOGGER.info("Loading model: %s", model_name)
        self._active_model = self._load_model(**cfg)
        self._active_model_name = model_name
        self.ctx.model = self._active_model
        self.ctx.args = self._make_ctx_args(model_name)

    @staticmethod
    def _load_model(
        model_name: str,
        backend: str,
        *,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
        enforce_eager: bool = False,
        language_model_only: bool = False,
        enable_prefix_caching: bool = False,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        hf_cache_dir: Optional[str] = None,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        reasoning_effort: Optional[str] = None,
        enable_thinking: bool = True,
        speculative_config: Optional[dict] = None,
    ):
        from model_client import load_model

        backend = normalize_text(backend).lower() or "vllm_offline"
        if backend == "vllm_offline":
            return load_model(
                model_name,
                vllm_offline=True,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                enable_expert_parallel=enable_expert_parallel,
                enforce_eager=enforce_eager,
                language_model_only=language_model_only,
                enable_prefix_caching=enable_prefix_caching,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                kv_cache_dtype=kv_cache_dtype,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                hf_cache_dir=hf_cache_dir,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
                speculative_config=speculative_config,
            )
        return load_model(model_name)

    def _pair_models(self, model_a: str, model_b: str) -> Tuple[str, str]:
        idx_a = self._response_model_index.get(model_a, 0)
        idx_b = self._response_model_index.get(model_b, 0)
        if idx_a <= idx_b:
            return model_a, model_b
        return model_b, model_a

    def _pair_family(self, model_a: str, model_b: str) -> str:
        first, second = self._pair_models(model_a, model_b)
        return f"{model_label(first)}_vs_{model_label(second)}"

    def _iter_response_pairs(self) -> Iterable[Tuple[str, str]]:
        return combinations(self.response_models, 2)

    def _record_anchor_model(self, record: Dict[str, Any]) -> Optional[str]:
        source_model = record.get("source_model_name")
        if source_model not in self.response_models:
            return None
        raw_source_action = record.get("source_generated_action_raw", "")
        if not isinstance(raw_source_action, str) or not raw_source_action.strip():
            return None
        return source_model

    def _iter_record_response_pairs(
        self,
        record: Dict[str, Any],
    ) -> Iterable[Tuple[str, str]]:
        anchor_model = self._record_anchor_model(record)
        for model_a, model_b in self._iter_response_pairs():
            if anchor_model is not None and anchor_model not in (model_a, model_b):
                continue
            yield model_a, model_b

    def _prepare_records(
        self,
        samples: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        records: List[Dict[str, Any]] = []
        invalid: List[Dict[str, Any]] = []
        for sample in samples:
            source_model_name = extract_source_model_name(sample)
            try:
                tp, sc, vignette = reconstruct_inputs(sample)
            except Exception as exc:
                invalid.append(
                    {
                        "name": sample.get("name", "unknown"),
                        "source_model_name": source_model_name,
                        "error": f"reconstruction failed: {exc}",
                    }
                )
                continue

            records.append(
                {
                    "name": sample["name"],
                    "source_model_name": source_model_name,
                    "source_generated_action_raw": extract_source_generated_action(
                        sample,
                        source_model_name=source_model_name,
                    ),
                    "_tp": tp,
                    "_sc": sc,
                    "_v": vignette,
                    "responses": {},
                    "sensibility_judgments": {},
                    "comparative_judgments": {},
                    "selected_pair": None,
                }
            )
        return records, invalid

    def _run_response_generation_pass(
        self,
        records: List[Dict[str, Any]],
        model_name: str,
    ) -> None:
        LOGGER.info("=== Response generation: %s ===", model_name)
        generation_records: List[Dict[str, Any]] = []
        reused_count = 0

        for record in records:
            entry = {
                "model_name": model_name,
                "model_label": model_label(model_name),
                "action": None,
                "sensibility_passed": None,
                "sensibility_output": "",
                "error": None,
                "reused_source_action": False,
            }
            record["responses"][model_name] = entry

            raw_source_action = record.get("source_generated_action_raw", "")
            if (
                record.get("source_model_name") == model_name
                and isinstance(raw_source_action, str)
                and raw_source_action.strip()
            ):
                action = parse_naive_agent_result(
                    raw_source_action,
                    expected_action=record["_sc"]["final_action"],
                )
                if action is not None:
                    entry["action"] = action
                    entry["reused_source_action"] = True
                    reused_count += 1
                    continue
                LOGGER.warning(
                    "Sample %s: failed to parse stored generated_final_action for %s; regenerating",
                    record["name"],
                    model_name,
                )

            generation_records.append(record)

        if not generation_records:
            LOGGER.info(
                "Reused stored generated_final_action for all %d %s responses",
                reused_count,
                model_name,
            )
            return

        self._activate_model(model_name)
        ctx = self.ctx
        batch_size = self.args.batch_size
        if reused_count:
            LOGGER.info(
                "Reused stored generated_final_action for %d/%d %s responses; generating the rest",
                reused_count,
                len(records),
                model_name,
            )

        for chunk_start in range(0, len(generation_records), batch_size):
            chunk = generation_records[chunk_start:chunk_start + batch_size]
            LOGGER.info(
                "Generation batch %d-%d / %d for %s",
                chunk_start,
                chunk_start + len(chunk),
                len(generation_records),
                model_name,
            )
            prompts = [
                build_naive_agent_prompt(ctx, record["_tp"], record["_sc"], record["_v"])
                for record in chunk
            ]
            raw_results = ctx.batch_call_text(
                prompts,
                max_tokens=self.args.max_tokens_per_stage,
                filter=True,
            )

            for record, raw in zip(chunk, raw_results):
                entry = record["responses"][model_name]
                if raw is None:
                    entry["error"] = "agent returned empty"
                    continue

                action = parse_naive_agent_result(
                    raw,
                    expected_action=record["_sc"]["final_action"],
                )
                if action is None:
                    entry["error"] = "failed to parse agent action"
                    continue

                entry["action"] = action

    def _sensibility_action_items(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], str]]:
        """Return every response action that should receive sensibility judging."""
        items: List[Tuple[Dict[str, Any], str]] = []
        for record in records:
            for response_model in self.response_models:
                response = record["responses"].get(response_model)
                if not response:
                    continue
                if response.get("action"):
                    items.append((record, response_model))
        return items

    def _action_pair_items(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], str, str]]:
        items: List[Tuple[Dict[str, Any], str, str]] = []
        for record in records:
            for model_a, model_b in self._iter_record_response_pairs(record):
                response_a = record["responses"].get(model_a)
                response_b = record["responses"].get(model_b)
                if not response_a or not response_b:
                    continue
                if not response_a.get("action") or not response_b.get("action"):
                    continue
                items.append((record, model_a, model_b))
        return items

    def _required_sensibility_agreement(self) -> int:
        return len(self.judge_models) // 2 + 1

    def _required_leakage_agreement(self) -> int:
        return len(self.judge_models)

    def _required_comparative_agreement(self) -> int:
        return len(self.judge_models)

    def _required_better_agreement(self) -> int:
        return len(self.judge_models)

    def _response_vote_stats(
        self,
        record: Dict[str, Any],
        response_model: str,
        *,
        bucket_name: str,
        field_name: str,
        threshold: int,
    ) -> Dict[str, Any]:
        yes_judges: List[str] = []
        no_judges: List[str] = []
        for judge_model in self.judge_models:
            response_map = record.get(bucket_name, {}).get(judge_model, {})
            value = response_map.get(response_model, {}).get(field_name)
            if value is True:
                yes_judges.append(judge_model)
            elif value is False:
                no_judges.append(judge_model)

        majority: Optional[bool] = None
        if len(yes_judges) >= threshold:
            majority = True
        elif len(no_judges) >= threshold:
            majority = False

        return {
            "majority": majority,
            "yes_count": len(yes_judges),
            "no_count": len(no_judges),
            "yes_judges": yes_judges,
            "no_judges": no_judges,
            "threshold": threshold,
        }

    def _response_sensibility_stats(
        self,
        record: Dict[str, Any],
        response_model: str,
    ) -> Dict[str, Any]:
        return self._response_vote_stats(
            record,
            response_model,
            bucket_name="sensibility_judgments",
            field_name="sensible",
            threshold=self._required_sensibility_agreement(),
        )

    def _refresh_response_sensibility(
        self,
        records: List[Dict[str, Any]],
        *,
        final: bool = False,
    ) -> None:
        for record in records:
            for response_model, response in record["responses"].items():
                if not response.get("action"):
                    continue

                stats = self._response_vote_stats(
                    record,
                    response_model,
                    bucket_name="sensibility_judgments",
                    field_name="sensible",
                    threshold=self._required_sensibility_agreement(),
                )
                sensible_output = ""
                fallback_output = ""
                for judge_model in self.judge_models:
                    sensibility = record["sensibility_judgments"].get(judge_model, {}).get(
                        response_model, {}
                    )
                    verdict = sensibility.get("sensible")
                    output = sensibility.get("output", "")
                    if output and not fallback_output:
                        fallback_output = output
                    if verdict is stats["majority"] and output and not sensible_output:
                        sensible_output = output

                if stats["majority"] is True:
                    response["sensibility_passed"] = True
                    response["sensibility_output"] = sensible_output or fallback_output
                    if response.get("error") in (
                        "sensibility majority rejection",
                        "sensibility lacked majority approval",
                    ):
                        response["error"] = None
                elif final and stats["majority"] is False:
                    response["sensibility_passed"] = False
                    response["sensibility_output"] = fallback_output
                    response["error"] = "sensibility majority rejection"
                elif final:
                    response["sensibility_passed"] = False
                    response["sensibility_output"] = fallback_output
                    response["error"] = "sensibility lacked majority approval"
                else:
                    response["sensibility_passed"] = None
                    response["sensibility_output"] = fallback_output

    def _pairwise_leakage_stats(
        self,
        record: Dict[str, Any],
        model_a: str,
        model_b: str,
    ) -> Dict[str, Any]:
        pair_family = self._pair_family(model_a, model_b)
        response_a_yes_judges: List[str] = []
        response_a_no_judges: List[str] = []
        response_b_yes_judges: List[str] = []
        response_b_no_judges: List[str] = []

        for judge_model in self.judge_models:
            comparative = record["comparative_judgments"].get(judge_model, {}).get(pair_family, {})
            response_a_leaks = comparative.get("response_a_leaks")
            response_b_leaks = comparative.get("response_b_leaks")

            if response_a_leaks is True:
                response_a_yes_judges.append(judge_model)
            elif response_a_leaks is False:
                response_a_no_judges.append(judge_model)

            if response_b_leaks is True:
                response_b_yes_judges.append(judge_model)
            elif response_b_leaks is False:
                response_b_no_judges.append(judge_model)

        threshold = self._required_leakage_agreement()
        return {
            "pair_family": pair_family,
            "response_a_yes_count": len(response_a_yes_judges),
            "response_a_no_count": len(response_a_no_judges),
            "response_a_yes_judges": response_a_yes_judges,
            "response_a_no_judges": response_a_no_judges,
            "response_b_yes_count": len(response_b_yes_judges),
            "response_b_no_count": len(response_b_no_judges),
            "response_b_yes_judges": response_b_yes_judges,
            "response_b_no_judges": response_b_no_judges,
            "threshold": threshold,
        }

    def _pairwise_majority(
        self,
        record: Dict[str, Any],
        model_a: str,
        model_b: str,
    ) -> Optional[Dict[str, Any]]:
        pair_family = self._pair_family(model_a, model_b)
        differ_judges: List[str] = []
        better_a_judges: List[str] = []
        better_b_judges: List[str] = []
        same_judges: List[str] = []
        for judge_model in self.judge_models:
            comparative = record["comparative_judgments"].get(judge_model, {}).get(pair_family, {})
            if comparative.get("differ") is False:
                same_judges.append(judge_model)
                continue
            if comparative.get("differ") is not True:
                continue
            differ_judges.append(judge_model)
            better_model = comparative.get("better_model")
            if better_model == model_a:
                better_a_judges.append(judge_model)
            elif better_model == model_b:
                better_b_judges.append(judge_model)

        differ_threshold = self._required_comparative_agreement()
        if len(differ_judges) < differ_threshold:
            return None

        better_threshold = self._required_better_agreement()
        majority_better_model: Optional[str] = None
        majority_judges: List[str] = []
        if len(better_a_judges) >= better_threshold:
            majority_better_model = model_a
            majority_judges = better_a_judges
        elif len(better_b_judges) >= better_threshold:
            majority_better_model = model_b
            majority_judges = better_b_judges

        if majority_better_model is None:
            return None

        return {
            "pair_family": pair_family,
            "differ_support_count": len(differ_judges),
            "differ_judges": differ_judges,
            "majority_better_model": majority_better_model,
            "majority_support_count": len(majority_judges),
            "majority_judges": majority_judges,
            "better_a_count": len(better_a_judges),
            "better_b_count": len(better_b_judges),
            "same_count": len(same_judges),
            "threshold": differ_threshold,
            "differ_threshold": differ_threshold,
            "better_threshold": better_threshold,
        }

    def _clean_comparative_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "assistant_a": result.get("assistant_a"),
            "assistant_b": result.get("assistant_b"),
            "differ": result.get("differ"),
            "better_model": result.get("better_model"),
            "response_a_leaks": result.get("response_a_leaks"),
            "response_a_leaks_details": result.get("response_a_leaks_details"),
            "response_a_omits": result.get("response_a_omits"),
            "response_a_omits_details": result.get("response_a_omits_details"),
            "response_b_leaks": result.get("response_b_leaks"),
            "response_b_leaks_details": result.get("response_b_leaks_details"),
            "response_b_omits": result.get("response_b_omits"),
            "response_b_omits_details": result.get("response_b_omits_details"),
            "comparison": result.get("comparison"),
            "explanation": result.get("explanation"),
            "verdict": result.get("verdict"),
            "output": result.get("output"),
        }

    def _pairwise_majority_comparative_output(
        self,
        record: Dict[str, Any],
        model_a: str,
        model_b: str,
    ) -> Optional[Dict[str, Any]]:
        pair_family = self._pair_family(model_a, model_b)
        threshold = len(self.judge_models) // 2 + 1
        supporting_judges_by_outcome: Dict[Tuple[str, Optional[str]], List[str]] = {}

        for judge_model in self.judge_models:
            comparative = record["comparative_judgments"].get(judge_model, {}).get(pair_family, {})
            differ = comparative.get("differ")
            better_model = comparative.get("better_model")

            if differ is False:
                outcome = ("tie", None)
            elif differ is True and better_model in (model_a, model_b):
                outcome = ("winner", better_model)
            else:
                continue

            supporting_judges_by_outcome.setdefault(outcome, []).append(judge_model)

        best_outcome: Optional[Tuple[str, Optional[str]]] = None
        best_judges: List[str] = []
        for outcome, judges in supporting_judges_by_outcome.items():
            if len(judges) < threshold:
                continue
            if len(judges) > len(best_judges):
                best_outcome = outcome
                best_judges = judges

        if best_outcome is None or not best_judges:
            return None

        representative_judge = best_judges[0]
        representative_result = (
            record["comparative_judgments"].get(representative_judge, {}).get(pair_family, {})
        )
        cleaned = self._clean_comparative_result(representative_result)
        cleaned.update(
            {
                "pair_family": pair_family,
                "judge_model": representative_judge,
                "judge_label": model_label(representative_judge),
                "support_count": len(best_judges),
                "support_threshold": threshold,
                "supporting_judges": list(best_judges),
                "agreement_kind": best_outcome[0],
                "majority_better_model": best_outcome[1],
            }
        )
        return cleaned

    def _build_pair_analysis(
        self,
        record: Dict[str, Any],
        model_a: str,
        model_b: str,
    ) -> Optional[Dict[str, Any]]:
        response_a = record["responses"].get(model_a)
        response_b = record["responses"].get(model_b)
        if not response_a or not response_b:
            return None
        if not response_a.get("action") or not response_b.get("action"):
            return None

        pair_family = self._pair_family(model_a, model_b)
        sensibility_a_stats = self._response_sensibility_stats(record, model_a)
        sensibility_b_stats = self._response_sensibility_stats(record, model_b)
        pairwise_leakage = self._pairwise_leakage_stats(record, model_a, model_b)
        pairwise_majority = self._pairwise_majority(record, model_a, model_b)
        majority_comparative_output = self._pairwise_majority_comparative_output(
            record,
            model_a,
            model_b,
        )
        total_judges = len(self.judge_models)

        comparative_by_judge: Dict[str, Dict[str, Any]] = {}
        for judge_model in self.judge_models:
            comparative = record["comparative_judgments"].get(judge_model, {}).get(pair_family, {})
            comparative_by_judge[judge_model] = {
                "differ": comparative.get("differ"),
                "better_model": comparative.get("better_model"),
                "response_a_leaks": comparative.get("response_a_leaks"),
                "response_b_leaks": comparative.get("response_b_leaks"),
                "verdict": comparative.get("verdict"),
            }

        leakage_unanimity_passed = (
            max(
                pairwise_leakage["response_a_yes_count"],
                pairwise_leakage["response_b_yes_count"],
            )
            >= pairwise_leakage["threshold"]
        )
        both_responses_majority_sensible = (
            sensibility_a_stats["majority"] is True and sensibility_b_stats["majority"] is True
        )
        both_responses_unanimously_sensible = (
            sensibility_a_stats["yes_count"] == total_judges
            and sensibility_b_stats["yes_count"] == total_judges
        )

        filter_reasons: List[str] = []
        if sensibility_a_stats["majority"] is not True:
            filter_reasons.append(f"{model_label(model_a)} lacked sensibility majority approval")
        if sensibility_b_stats["majority"] is not True:
            filter_reasons.append(f"{model_label(model_b)} lacked sensibility majority approval")
        if not leakage_unanimity_passed:
            filter_reasons.append("no response in the pair had unanimous leakage support")
        if pairwise_majority is None:
            filter_reasons.append(
                "comparative judges did not unanimously agree that one response was better"
            )

        return {
            "pair_family": pair_family,
            "model_a": model_a,
            "model_a_label": model_label(model_a),
            "model_b": model_b,
            "model_b_label": model_label(model_b),
            "sensibility_a": sensibility_a_stats,
            "sensibility_b": sensibility_b_stats,
            "both_responses_majority_sensible": both_responses_majority_sensible,
            "both_responses_unanimously_sensible": both_responses_unanimously_sensible,
            "pairwise_leakage": pairwise_leakage,
            "leakage_unanimity_passed": leakage_unanimity_passed,
            "pairwise_majority": pairwise_majority,
            "majority_comparative_output": majority_comparative_output,
            "comparative_judgments_by_judge": comparative_by_judge,
            "passes_selection_filters": not filter_reasons,
            "selection_filter_reasons": filter_reasons,
        }

    def _build_pair_analyses(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        analyses: List[Dict[str, Any]] = []
        for model_a, model_b in self._iter_record_response_pairs(record):
            analysis = self._build_pair_analysis(record, model_a, model_b)
            if analysis is not None:
                analyses.append(analysis)
        return analyses

    def _run_judge_pass(
        self,
        records: List[Dict[str, Any]],
        judge_model: str,
    ) -> None:
        self._activate_model(judge_model)
        ctx = self.ctx
        batch_size = self.args.batch_size

        LOGGER.info("=== Sensibility judge: %s ===", judge_model)
        sensibility_items = self._sensibility_action_items(records)
        for chunk_start in range(0, len(sensibility_items), batch_size):
            chunk = sensibility_items[chunk_start:chunk_start + batch_size]
            LOGGER.info(
                "Sensibility batch %d-%d / %d for %s",
                chunk_start,
                chunk_start + len(chunk),
                len(sensibility_items),
                judge_model,
            )
            prompts = [
                build_sensibility_check_prompt(
                    ctx,
                    record["_tp"],
                    record["responses"][response_model]["action"],
                    vignette=record["_v"],
                )
                for record, response_model in chunk
            ]
            raw_results = ctx.batch_call_text(
                prompts,
                max_tokens=self.args.max_tokens_per_stage,
                filter=True,
            )
            for (record, response_model), raw in zip(chunk, raw_results):
                judge_bucket = record["sensibility_judgments"].setdefault(judge_model, {})
                result = {
                    "response_model": response_model,
                    "sensible": None,
                    "output": "",
                }
                judge_bucket[response_model] = result
                if raw is None:
                    continue
                result["output"] = raw.strip()
                result["sensible"] = parse_sensibility_check_result(raw)

        LOGGER.info("=== Comparative judge: %s ===", judge_model)
        pair_items = self._action_pair_items(records)
        for chunk_start in range(0, len(pair_items), batch_size):
            chunk = pair_items[chunk_start:chunk_start + batch_size]
            LOGGER.info(
                "Comparative batch %d-%d / %d for %s",
                chunk_start,
                chunk_start + len(chunk),
                len(pair_items),
                judge_model,
            )
            prompts = [
                build_comparative_judge_prompt(
                    ctx,
                    record["_tp"],
                    record["_sc"],
                    record["_v"],
                    record["responses"][model_a]["action"],
                    record["responses"][model_b]["action"],
                )
                for record, model_a, model_b in chunk
            ]
            raw_results = ctx.batch_call_text(
                prompts,
                max_tokens=self.args.max_tokens_per_stage,
                filter=True,
            )
            for (record, model_a, model_b), raw in zip(chunk, raw_results):
                pair_family = self._pair_family(model_a, model_b)
                judge_bucket = record["comparative_judgments"].setdefault(judge_model, {})
                result = {
                    "model_a": model_a,
                    "model_b": model_b,
                    "pair_family": pair_family,
                    "assistant_a": None,
                    "assistant_b": None,
                    "differ": None,
                    "better_model": None,
                    "response_a_leaks": None,
                    "response_a_leaks_details": None,
                    "response_a_omits": None,
                    "response_a_omits_details": None,
                    "response_b_leaks": None,
                    "response_b_leaks_details": None,
                    "response_b_omits": None,
                    "response_b_omits_details": None,
                    "comparison": None,
                    "explanation": None,
                    "verdict": None,
                    "output": "",
                }
                judge_bucket[pair_family] = result
                if raw is None:
                    continue
                differ, better_agent, verdict = parse_comparative_judge_result(raw)
                result["output"] = raw.strip()
                result["assistant_a"] = extract_comparative_judge_section(raw, "ASSISTANT A")
                result["assistant_b"] = extract_comparative_judge_section(raw, "ASSISTANT B")
                result["differ"] = differ
                result["response_a_leaks"] = extract_comparative_judge_boolean(
                    raw,
                    "RESPONSE A LEAKS",
                )
                result["response_a_leaks_details"] = extract_comparative_judge_section(
                    raw,
                    "RESPONSE A LEAKS DETAILS",
                )
                result["response_a_omits"] = extract_comparative_judge_boolean(
                    raw,
                    "RESPONSE A OMITS",
                )
                result["response_a_omits_details"] = extract_comparative_judge_section(
                    raw,
                    "RESPONSE A OMITS DETAILS",
                )
                result["response_b_leaks"] = extract_comparative_judge_boolean(
                    raw,
                    "RESPONSE B LEAKS",
                )
                result["response_b_leaks_details"] = extract_comparative_judge_section(
                    raw,
                    "RESPONSE B LEAKS DETAILS",
                )
                result["response_b_omits"] = extract_comparative_judge_boolean(
                    raw,
                    "RESPONSE B OMITS",
                )
                result["response_b_omits_details"] = extract_comparative_judge_section(
                    raw,
                    "RESPONSE B OMITS DETAILS",
                )
                result["comparison"] = extract_comparative_judge_section(raw, "COMPARISON")
                result["explanation"] = extract_comparative_judge_section(raw, "EXPLANATION")
                result["verdict"] = verdict
                if better_agent == "A":
                    result["better_model"] = model_a
                elif better_agent == "B":
                    result["better_model"] = model_b

    def _build_selection_options(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        options: List[Dict[str, Any]] = []

        for model_a, model_b in self._iter_record_response_pairs(record):
            sensibility_a_stats = self._response_sensibility_stats(record, model_a)
            if sensibility_a_stats["majority"] is not True:
                continue
            sensibility_b_stats = self._response_sensibility_stats(record, model_b)
            if sensibility_b_stats["majority"] is not True:
                continue

            pairwise_leakage = self._pairwise_leakage_stats(record, model_a, model_b)
            leakage_threshold = pairwise_leakage["threshold"]
            if max(
                pairwise_leakage["response_a_yes_count"],
                pairwise_leakage["response_b_yes_count"],
            ) < leakage_threshold:
                continue

            pair_family = self._pair_family(model_a, model_b)
            pairwise_majority = self._pairwise_majority(record, model_a, model_b)
            if pairwise_majority is None:
                continue
            majority_better_model = pairwise_majority.get("majority_better_model")
            sensibility_a_yes = sensibility_a_stats["yes_count"]
            sensibility_b_yes = sensibility_b_stats["yes_count"]
            sensibility_all_judges = len(self.judge_models)
            both_unanimously_sensible = (
                sensibility_a_yes == sensibility_all_judges
                and sensibility_b_yes == sensibility_all_judges
            )

            eligible_judges: List[Dict[str, Any]] = []

            for judge_model in self.judge_models:
                comparative_bucket = record["comparative_judgments"].get(judge_model, {})
                comparative = comparative_bucket.get(pair_family, {})
                differ = comparative.get("differ")

                if differ is not True:
                    continue
                if (
                    majority_better_model is not None
                    and comparative.get("better_model") != majority_better_model
                ):
                    continue

                eligible_judges.append(
                    {
                        "pair_family": pair_family,
                        "judge_model": judge_model,
                        "judge_label": model_label(judge_model),
                        "model_a": model_a,
                        "model_b": model_b,
                        "sensibility_a": True,
                        "sensibility_b": True,
                        "sensibility_a_yes_count": sensibility_a_yes,
                        "sensibility_b_yes_count": sensibility_b_yes,
                        "leakage_a_yes_count": pairwise_leakage["response_a_yes_count"],
                        "leakage_b_yes_count": pairwise_leakage["response_b_yes_count"],
                        "leakage_threshold": leakage_threshold,
                        "both_responses_unanimously_sensible": both_unanimously_sensible,
                        "pairwise_majority": pairwise_majority,
                        "comparative_better_model": comparative.get("better_model"),
                        "comparative_verdict": comparative.get("verdict"),
                    }
                )

            if not eligible_judges:
                continue

            pair_support = len(eligible_judges)
            split_support = max(
                pairwise_majority["better_a_count"],
                pairwise_majority["better_b_count"],
            )
            for item in eligible_judges:
                item["pair_support"] = pair_support
                item["split_support"] = split_support
                options.append(item)

        return options

    def _option_rank(
        self,
        option: Dict[str, Any],
        pair_counts: Counter[str],
        judge_counts: Counter[str],
    ) -> Tuple[Any, ...]:
        return (
            0 if option["both_responses_unanimously_sensible"] else 1,
            -min(option["sensibility_a_yes_count"], option["sensibility_b_yes_count"]),
            -(option["sensibility_a_yes_count"] + option["sensibility_b_yes_count"]),
            judge_counts[option["judge_model"]],
            pair_counts[option["pair_family"]],
            -option["split_support"],
            -option["pair_support"],
            option["judge_label"],
            option["pair_family"],
        )

    def _select_pairs(self, records: List[Dict[str, Any]]) -> None:
        pair_counts: Counter[str] = Counter()
        judge_counts: Counter[str] = Counter()

        for record in records:
            record["_selection_options"] = self._build_selection_options(record)

        selectable = [r for r in records if r["_selection_options"]]
        selectable.sort(
            key=lambda record: (
                len(record["_selection_options"]),
                stable_order_token(self.args.seed, record["name"]),
            )
        )

        for record in selectable:
            best = min(
                record["_selection_options"],
                key=lambda option: self._option_rank(option, pair_counts, judge_counts),
            )
            pair_counts[best["pair_family"]] += 1
            judge_counts[best["judge_model"]] += 1

            response_a = record["responses"][best["model_a"]]
            response_b = record["responses"][best["model_b"]]
            comparative_bucket = record["comparative_judgments"][best["judge_model"]]
            comparative = comparative_bucket[best["pair_family"]]

            record["selected_pair"] = {
                "pair_family": best["pair_family"],
                "judge_model": best["judge_model"],
                "judge_label": best["judge_label"],
                "model_a": best["model_a"],
                "model_a_label": model_label(best["model_a"]),
                "model_b": best["model_b"],
                "model_b_label": model_label(best["model_b"]),
                "response_a": response_a["action"],
                "response_b": response_b["action"],
                "sensibility_a": best["sensibility_a"],
                "sensibility_b": best["sensibility_b"],
                "leakage_a_yes_count": best["leakage_a_yes_count"],
                "leakage_b_yes_count": best["leakage_b_yes_count"],
                "leakage_threshold": best["leakage_threshold"],
                "pair_support": best["pair_support"],
                "split_support": best["split_support"],
                "pairwise_majority": best["pairwise_majority"],
                "comparative_better_model": best["comparative_better_model"],
                "comparative_verdict": best["comparative_verdict"],
                "comparative_comparison": comparative.get("comparison"),
                "comparative_explanation": comparative.get("explanation"),
                "comparative_judge_output": comparative.get("output", ""),
            }

    def _clean_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model_name": response.get("model_name"),
            "model_label": response.get("model_label"),
            "action": response.get("action"),
            "sensibility_passed": response.get("sensibility_passed"),
            "error": response.get("error"),
            "reused_source_action": response.get("reused_source_action"),
        }

    def _clean_sensibility_judgments(
        self,
        judgments: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        cleaned: Dict[str, Dict[str, Any]] = {}
        for judge_model, response_map in judgments.items():
            cleaned[judge_model] = {}
            for response_model, result in response_map.items():
                cleaned[judge_model][response_model] = {
                    "sensible": result.get("sensible"),
                }
        return cleaned

    def _clean_comparative_judgments(
        self,
        judgments: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        cleaned: Dict[str, Dict[str, Any]] = {}
        for judge_model, pair_map in judgments.items():
            cleaned[judge_model] = {}
            for pair_family, result in pair_map.items():
                cleaned[judge_model][pair_family] = self._clean_comparative_result(result)
        return cleaned

    def _clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {
            "name": record["name"],
            "source_model_name": record.get("source_model_name"),
            "source_model_label": (
                model_label(record["source_model_name"])
                if record.get("source_model_name")
                else None
            ),
            "responses": {
                model_name: self._clean_response(response)
                for model_name, response in record["responses"].items()
            },
            "sensibility_judgments": self._clean_sensibility_judgments(
                record["sensibility_judgments"]
            ),
            "comparative_judgments": self._clean_comparative_judgments(
                record["comparative_judgments"]
            ),
            "selected_pair": record.get("selected_pair"),
        }
        if "_selection_options" in record:
            cleaned["eligible_options_count"] = len(record["_selection_options"])
        return cleaned

    def _build_summary(
        self,
        records: List[Dict[str, Any]],
        invalid_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        response_attempts = len(records) * len(self.response_models)
        response_actions = 0
        sensible_responses = 0
        sensibility_total = 0
        sensibility_parse_ok = 0
        comparative_total = 0
        comparative_parse_ok = 0
        analyzed_pairs = 0
        selection_eligible_pairs = 0
        selected_pairs = 0
        samples_with_options = 0
        pair_counts: Counter[str] = Counter()
        judge_counts: Counter[str] = Counter()

        for record in records:
            pair_analyses = self._build_pair_analyses(record)
            analyzed_pairs += len(pair_analyses)
            selection_eligible_pairs += sum(
                1 for analysis in pair_analyses if analysis["passes_selection_filters"]
            )

            for response in record["responses"].values():
                if response.get("action"):
                    response_actions += 1
                if response.get("sensibility_passed") is True:
                    sensible_responses += 1

            for response_map in record["sensibility_judgments"].values():
                sensibility_total += len(response_map)
                sensibility_parse_ok += sum(
                    1 for result in response_map.values() if result.get("sensible") is not None
                )

            for pair_map in record["comparative_judgments"].values():
                comparative_total += len(pair_map)
                comparative_parse_ok += sum(
                    1 for result in pair_map.values() if result.get("differ") is not None
                )

            if record.get("_selection_options"):
                samples_with_options += 1

            selected = record.get("selected_pair")
            if selected:
                selected_pairs += 1
                pair_counts[selected["pair_family"]] += 1
                judge_counts[selected["judge_model"]] += 1

        return {
            "num_input": len(records) + len(invalid_records),
            "num_reconstructed": len(records),
            "num_invalid": len(invalid_records),
            "num_response_attempts": response_attempts,
            "num_response_actions": response_actions,
            "num_sensible_responses": sensible_responses,
            "num_sensibility_judgments": sensibility_total,
            "num_sensibility_parse_ok": sensibility_parse_ok,
            "num_comparative_judgments": comparative_total,
            "num_comparative_parse_ok": comparative_parse_ok,
            "num_analyzed_pairs": analyzed_pairs,
            "num_pairs_passing_selection_filters": selection_eligible_pairs,
            "num_samples_with_eligible_pairs": samples_with_options,
            "num_selected_pairs": selected_pairs,
            "selected_pair_family_counts": dict(pair_counts),
            "selected_judge_counts": dict(judge_counts),
        }

    def _build_output(
        self,
        records: List[Dict[str, Any]],
        invalid_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        samples = [self._clean_record(record) for record in records]
        analyzed_pairs = [
            {
                "name": record["name"],
                "source_model_name": record.get("source_model_name"),
                **analysis,
            }
            for record in records
            for analysis in self._build_pair_analyses(record)
        ]
        selected_pairs = [
            {
                "name": record["name"],
                "source_model_name": record.get("source_model_name"),
                **record["selected_pair"],
            }
            for record in records
            if record.get("selected_pair")
        ]
        return {
            "metadata": {
                "response_models": self.response_models,
                "judge_models": self.judge_models,
                "backend": self.args.backend,
                "input_path": str(self.args.input_path),
                "completed_response_models": self._completed_response_models,
                "completed_sensibility_judge_models": self._completed_sensibility_judge_models,
                "completed_comparative_judge_models": self._completed_comparative_judge_models,
                "selection_complete": self._selection_complete,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **self._build_summary(records, invalid_records),
            },
            "analyzed_pairs": analyzed_pairs,
            "selected_pairs": selected_pairs,
            "invalid_samples": invalid_records,
            "samples": samples,
        }

    def _write_checkpoint(
        self,
        output_path: Path,
        records: List[Dict[str, Any]],
        invalid_records: List[Dict[str, Any]],
    ) -> None:
        write_json_atomic(output_path, self._build_output(records, invalid_records))

    def run(self) -> Dict[str, Any]:
        input_path = Path(self.args.input_path)
        output_path = Path(self.args.output_path)
        with input_path.open("r", encoding="utf-8") as handle:
            all_samples: List[Dict[str, Any]] = json.load(handle)
        if not isinstance(all_samples, list):
            raise RuntimeError(f"Expected JSON array in {input_path}")

        if self.args.sample_names:
            name_set = {name.strip() for name in self.args.sample_names.split(",") if name.strip()}
            all_samples = [sample for sample in all_samples if sample.get("name") in name_set]

        if self.args.limit and self.args.limit > 0:
            all_samples = all_samples[:self.args.limit]

        LOGGER.info("Preparing %d samples from %s", len(all_samples), input_path)
        records, invalid_records = self._prepare_records(all_samples)
        self._write_checkpoint(output_path, records, invalid_records)

        if not records:
            return self._build_output(records, invalid_records)

        for response_model in self.response_models:
            self._run_response_generation_pass(records, response_model)
            self._completed_response_models.append(response_model)
            self._write_checkpoint(output_path, records, invalid_records)

        for judge_model in self.judge_models:
            self._run_judge_pass(records, judge_model)
            self._completed_sensibility_judge_models.append(judge_model)
            self._completed_comparative_judge_models.append(judge_model)
            self._refresh_response_sensibility(records, final=False)
            self._write_checkpoint(output_path, records, invalid_records)

        self._refresh_response_sensibility(records, final=True)
        self._select_pairs(records)
        self._selection_complete = True
        self._write_checkpoint(output_path, records, invalid_records)
        return self._build_output(records, invalid_records)

    def close(self) -> None:
        self._unload_model()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine privacy-handling response pairs across multiple models.",
    )

    out = parser.add_argument_group("Output & control")
    out.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the combined generated samples JSON.",
    )
    out.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path for mined output JSON. Defaults to <input_dir>/mined_<response_models>.json.",
    )
    out.add_argument("--print-prompts", action="store_true", help="Log prompts before model calls.")
    out.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    out.add_argument("--seed", type=int, default=42, help="Random seed.")

    model = parser.add_argument_group("Model & backend")
    model.add_argument(
        "--backend",
        type=str,
        default="vllm_offline",
        choices=["vllm_offline", "auto"],
        help="Model backend: vllm_offline or auto.",
    )
    model.add_argument(
        "--response-models",
        nargs="+",
        default=list(DEFAULT_RESPONSE_MODELS),
        help=(
            "Models used to generate fresh responses. "
            "Defaults to gpt-oss-120b, NVIDIA-Nemotron-3-Super-120B-A12B-FP8, "
            "and Qwen3.5-397B-A17B-FP8."
        ),
    )
    model.add_argument(
        "--judge-models",
        nargs="+",
        default=None,
        help="Judge models used for sensibility and comparative judging. Defaults to --response-models.",
    )
    model.add_argument(
        "--disable-model-presets",
        action="store_true",
        help="Disable built-in per-model runtime and sampling presets that mirror the known gpt/nvidia/qwen generation runs.",
    )
    model.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for offline vLLM.",
    )
    model.add_argument(
        "--vllm-pipeline-parallel-size",
        type=int,
        default=1,
        help="Pipeline parallel size for offline vLLM.",
    )
    model.add_argument(
        "--vllm-enable-expert-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable expert parallel for offline vLLM.",
    )
    model.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force eager execution in offline vLLM.",
    )
    model.add_argument(
        "--vllm-language-model-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only load the language model for vLLM.",
    )
    model.add_argument(
        "--vllm-enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable prefix caching for vLLM.",
    )
    model.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use for vLLM.",
    )
    model.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Optional max model context length for vLLM.",
    )
    model.add_argument(
        "--vllm-kv-cache-dtype",
        type=str,
        default="auto",
        choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
        help="KV cache dtype for vLLM.",
    )
    model.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty for offline vLLM generation.",
    )
    model.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for offline vLLM generation.",
    )
    model.add_argument(
        "--speculative-config",
        type=str,
        default=None,
        help="Optional JSON string for vLLM speculative decoding.",
    )
    model.add_argument("--hf-cache-dir", type=str, default=None, help="Optional HF cache dir.")
    model.add_argument(
        "--reasoning-effort",
        "--filter-reasoning-effort",
        dest="reasoning_effort",
        type=str,
        default="high",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for mine_samples model calls.",
    )
    model.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable thinking/reasoning for models that support it.",
    )
    model.add_argument(
        "--max-tokens-per-stage",
        type=int,
        default=64000,
        help="Max tokens for each model call.",
    )
    model.add_argument(
        "--filter-temperature",
        "--diverse-generation-temperature",
        dest="temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for mine_samples model calls.",
    )
    model.add_argument(
        "--filter-top-p",
        dest="top_p",
        type=float,
        default=1.0,
        help="Top-p for mine_samples model calls.",
    )
    model.add_argument(
        "--filter-top-k",
        dest="top_k",
        type=int,
        default=-1,
        help="Top-k for mine_samples model calls.",
    )

    res = parser.add_argument_group("Resource paths")
    res.add_argument(
        "--prompts-dir",
        type=str,
        default=str(DEFAULT_PROMPTS_DIR),
        help="Directory containing prompt templates.",
    )
    res.add_argument(
        "--toolkit-specs-path",
        type=str,
        default=str(DEFAULT_TOOLKIT_SPECS_PATH),
        help="Path to toolkit specs JSON.",
    )

    pipe = parser.add_argument_group("Pipeline parameters")
    pipe.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Samples/prompts processed per batch.",
    )

    sel = parser.add_argument_group("Sample selection")
    sel.add_argument(
        "--sample-names",
        type=str,
        default=None,
        help="Comma-separated sample names to process.",
    )
    sel.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to process.",
    )

    args = parser.parse_args()
    if args.output_path is None:
        input_dir = Path(args.input_path).parent
        response_labels = [model_label(canonicalize_model_name(name)) for name in args.response_models]
        safe_models = "_".join(response_labels)
        args.output_path = str(input_dir / f"mined_{safe_models}.json")
    return args


def main() -> None:
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    miner = SampleMiner(args)
    try:
        miner.run()
    finally:
        miner.close()


if __name__ == "__main__":
    main()
