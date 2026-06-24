#!/usr/bin/env python3
"""
Staged generator for privacy-scenario benchmark data.

Pipeline (batch-parallelized):
1) Profile generation from sampled first names and constrained attribute options.
2) Scenario generation (combined seed+vignette — picks CI parameters, toolkits, final action,
   and generates story, instruction, sensitive items in one pass).
3) Vignette quality check (LLM judge filters bad scenarios before trajectory generation).
4) Trajectory + memories generation (one per surviving scenario).
5) Quality filter (optional LLM judge).
6) Leakage filter (naive agent action + zero-shot judgment).

All stages are batched across samples for maximum throughput
with vLLM offline inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from pipeline.constants import (
    PROMPT_TEMPLATE_FILES,
    REQUIRED_SEED_OPTION_KEYS,
)
from pipeline.context import PipelineContext
from pipeline.diversity import DiversityTracker
from pipeline.identifiers import IdentifierRegistry, reconcile_identifiers
from pipeline.leakage_filter import (
    build_leakage_judge_prompt,
    build_naive_agent_prompt,
    build_sensibility_check_prompt,
    parse_leakage_judge_result,
    parse_naive_agent_result,
    parse_sensibility_check_result,
)
from pipeline.link_passthrough_filter import is_link_passthrough
from pipeline.output import build_output_sample
from pipeline.profile import build_profile_prompt, parse_profile_result
from pipeline.quality_filter import build_quality_filter_prompt, parse_quality_filter_result
from pipeline.seed_vignette import (
    build_scenario_prompt,
    parse_scenario_result,
)
from pipeline.trajectory_parsing import parse_steps
from pipeline.toolkit_registry import ToolkitRegistry
from pipeline.vignette_trajectory import (
    build_trajectory_memories_prompt,
    build_vignette_quality_check_prompt,
    parse_and_validate_candidate,
    parse_vignette_quality_check_result,
)
from pipeline.utils import (
    NameRow,
    append_jsonl,
    normalize_text,
    write_json_atomic,
)


LOGGER = logging.getLogger("scenario_generator")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NAMES_PATH = SCRIPT_DIR / "resources" / "names" / "name_frequency.jsonl"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "outputs" / "main_data_generated.json"
DEFAULT_ERROR_PATH = SCRIPT_DIR / "outputs" / "generation_errors.jsonl"
RESOURCES_DIR = SCRIPT_DIR / "resources"
DEFAULT_SEED_OPTIONS_PATH = RESOURCES_DIR / "seed_options.json"
DEFAULT_PROMPTS_DIR = RESOURCES_DIR / "prompts"
DEFAULT_TOOLKIT_SPECS_PATH = RESOURCES_DIR / "assets" / "all_toolkits.json"


@dataclass
class SampleState:
    """Tracks per-sample progress through the batch pipeline."""
    name: str
    first_name: str
    min_tool_calls: int = 4
    min_toolkits: int = 2
    profile_seed: Optional[Dict[str, Any]] = None
    profile: Optional[Dict[str, Any]] = None
    seed_candidate: Optional[Dict[str, Any]] = None
    vignette: Optional[Dict[str, Any]] = None
    trajectory_payload: Optional[Dict[str, Any]] = None
    agent_action: Optional[str] = None
    sensitive_memories: bool = True
    # Stage rejection reasons for diagnostics
    candidate_errors: List[Dict[str, Any]] = field(default_factory=list)
    leakage_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    diversity_counted: bool = False
    # Identifier replacement map from scenario stage (old -> new)
    id_replacements: Dict[str, str] = field(default_factory=dict)


class ScenarioGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        random.seed(args.seed)

        self.registry = ToolkitRegistry(
            toolkit_specs_path=Path(args.toolkit_specs_path),
        )

        self.seed_options = self._load_seed_options(Path(args.seed_options_path))
        self.sex_options = self.seed_options["sex_options"]
        self.ethnicity_options = self.seed_options["ethnicity_options"]
        self.religion_options = self.seed_options["religion_options"]
        self.prompt_templates = self._load_prompt_templates(Path(args.prompts_dir))

        self.model = self._load_model()

        self.name_rows = self._load_names(args.names_path)

        self.diversity = DiversityTracker()
        self.identifier_registry = IdentifierRegistry()

        if args.bootstrap_data_path and Path(args.bootstrap_data_path).exists():
            self.diversity.warm_from_dataset(Path(args.bootstrap_data_path))

        self.ctx = PipelineContext(
            args=args,
            registry=self.registry,
            prompt_templates=self.prompt_templates,
            seed_options=self.seed_options,
            model=self.model,
            diversity=self.diversity,
            name_rows=self.name_rows,
        )

    def _load_model(self):
        try:
            from model_client import load_model  # pylint: disable=import-error
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import model loader from {SCRIPT_DIR / 'model_client.py'}"
            ) from exc

        backend = normalize_text(getattr(self.args, "backend", "auto")).lower() or "auto"
        if backend == "vllm_offline":
            speculative_config = None
            raw_spec = getattr(self.args, "speculative_config", None)
            if raw_spec:
                speculative_config = json.loads(raw_spec)
            return load_model(
                self.args.generator_model,
                vllm_offline=True,
                tensor_parallel_size=self.args.vllm_tensor_parallel_size,
                enable_expert_parallel=self.args.vllm_enable_expert_parallel,
                enforce_eager=self.args.vllm_enforce_eager,
                language_model_only=self.args.vllm_language_model_only,
                enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                max_model_len=self.args.vllm_max_model_len,
                kv_cache_dtype=self.args.vllm_kv_cache_dtype,
                presence_penalty=self.args.presence_penalty,
                repetition_penalty=self.args.repetition_penalty,
                hf_cache_dir=self.args.hf_cache_dir,
                reasoning_effort=self.args.reasoning_effort,
                enable_thinking=self.args.enable_thinking,
                speculative_config=speculative_config,
            )

        # auto: fall back to OpenRouter
        return load_model(self.args.generator_model)

    def _load_seed_options(self, path: Path) -> Dict[str, List[str]]:
        if not path.exists():
            raise RuntimeError(f"Seed options file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse seed options file at {path}: {exc}") from exc

        if not isinstance(loaded, dict):
            raise RuntimeError("Seed options JSON must be an object.")

        normalized: Dict[str, List[str]] = {}
        for key in REQUIRED_SEED_OPTION_KEYS:
            raw_values = loaded.get(key)
            if not isinstance(raw_values, list):
                raise RuntimeError(f"Seed options key '{key}' must be a non-empty list.")

            deduped_values: List[str] = []
            seen = set()
            for item in raw_values:
                value = normalize_text(item)
                if not value or value in seen:
                    continue
                seen.add(value)
                deduped_values.append(value)

            if not deduped_values:
                raise RuntimeError(f"Seed options key '{key}' has no usable values.")
            normalized[key] = deduped_values

        return normalized

    def _load_prompt_templates(self, prompts_dir: Path) -> Dict[str, Template]:
        templates: Dict[str, Template] = {}
        missing: List[str] = []
        for filename in PROMPT_TEMPLATE_FILES:
            path = prompts_dir / filename
            if not path.exists():
                missing.append(str(path))
                continue
            with path.open("r", encoding="utf-8") as handle:
                templates[filename] = Template(handle.read())
        if missing:
            raise RuntimeError("Missing prompt template files:\n" + "\n".join(missing))
        return templates

    def _load_names(self, path: str) -> List[NameRow]:
        rows: List[NameRow] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                name = normalize_text(item.get("name", ""))
                if not name:
                    continue
                frequency = int(item.get("frequency", 1))
                rows.append(NameRow(name=name, frequency=max(1, frequency)))
        if not rows:
            raise RuntimeError(f"No names loaded from {path}")
        return rows

    def _sample_first_name(self) -> str:
        return self.diversity.sample_first_name(
            self.name_rows,
            float(self.args.name_frequency_alpha),
            self.args.name_repeat_penalty,
        )

    def _update_diversity_counters(
        self,
        seed_candidate: Dict[str, Any],
        vignette: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.diversity.update_seed(seed_candidate, vignette=vignette)

    # ------------------------------------------------------------------
    # Batch pipeline
    # ------------------------------------------------------------------

    def _append_sample_states(
        self,
        states: List[SampleState],
        sample_names: List[str],
    ) -> None:
        """Initialize per-sample state for a batch.

        Mutates ``states`` in place so partially built batches can still be
        rolled back if a catastrophic exception interrupts generation.
        """
        sensitive_mem_prob = float(getattr(self.args, "sensitive_memory_prob", 0.5))
        for name in sample_names:
            states.append(
                SampleState(
                    name=name,
                    first_name=self._sample_first_name(),
                    min_tool_calls=random.randint(self.args.min_tool_calls, self.args.max_tool_calls),
                    min_toolkits=random.randint(self.args.min_toolkits, self.args.max_toolkits),
                    sensitive_memories=random.random() < sensitive_mem_prob,
                )
            )

    def _rollback_partial_batch_state(self, states: List[SampleState]) -> None:
        """Rollback diversity counters for a catastrophically failed batch."""
        for sample in states:
            if sample.diversity_counted and sample.seed_candidate is not None:
                self.diversity.rollback_seed(sample.seed_candidate, vignette=sample.vignette)
                sample.diversity_counted = False
            if sample.profile is not None:
                self.diversity.rollback_profile(sample.profile)
            self.diversity.rollback_name(sample.first_name)

    def _run_profile_stage(self, states: List[SampleState]) -> None:
        """Stage 1: profile generation with overgeneration support."""
        ctx = self.ctx
        K = self.args.overgeneration_factor
        LOGGER.info("Stage 1: Generating %d profiles (K=%d)", len(states), K)

        # Build K prompts per sample (each call shuffles options + samples a new action hint)
        all_prompts: List[str] = []
        all_seeds: List[Dict[str, Any]] = []
        for sample in states:
            for _ in range(K):
                prompt, seed = build_profile_prompt(ctx, sample.first_name)
                all_prompts.append(prompt)
                all_seeds.append(seed)

        all_results = ctx.batch_call_json(all_prompts, max_tokens=self.args.max_tokens_per_stage, enable_thinking=False)

        for sample_idx, sample in enumerate(states):
            if sample.error is not None:
                continue

            # Collect valid candidates from this sample's K results
            candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []  # (profile, seed)
            for k in range(K):
                flat_idx = sample_idx * K + k
                raw = all_results[flat_idx]
                seed = all_seeds[flat_idx]
                if raw is None:
                    continue
                try:
                    profile = parse_profile_result(ctx, raw, seed)
                    candidates.append((profile, seed))
                except Exception:
                    continue

            if not candidates:
                sample.error = "Profile: all candidates failed parsing/validation"
                LOGGER.warning("Sample %s profile failed: no valid candidates out of %d", sample.name, K)
                continue

            if len(candidates) == 1 or K == 1:
                best_profile, best_seed = candidates[0]
            else:
                scored = [(self.diversity.score_profile(p), p, s) for p, s in candidates]
                scored.sort(key=lambda x: x[0], reverse=True)
                _, best_profile, best_seed = scored[0]
                LOGGER.debug(
                    "Sample %s: selected profile candidate 1/%d (score=%.3f)",
                    sample.name, len(candidates), scored[0][0],
                )

            sample.profile_seed = best_seed
            sample.profile = best_profile
            self.diversity.update_profile(best_profile)

    def _run_scenario_stage(self, states: List[SampleState]) -> None:
        """Stage 2: generate combined scenario (seed + vignette) with overgeneration."""
        ctx = self.ctx
        K = self.args.overgeneration_factor
        active_states = [sample for sample in states if sample.error is None]
        LOGGER.info(
            "Stage 2: Generating scenarios for %d samples (K=%d)",
            len(active_states), K,
        )
        if not active_states:
            return

        # Build K prompts per sample (each call shuffles action specs + toolkit lists)
        all_prompts: List[str] = []
        for sample in active_states:
            for _ in range(K):
                all_prompts.append(
                    build_scenario_prompt(ctx, sample.profile, min_toolkits=sample.min_toolkits)
                )

        all_results = ctx.batch_call_json(all_prompts, max_tokens=self.args.max_tokens_per_stage, enable_thinking=False)

        for state_idx, sample in enumerate(active_states):
            # Collect valid candidates from this sample's K results
            candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []  # (seed, vignette)
            for k in range(K):
                flat_idx = state_idx * K + k
                raw = all_results[flat_idx]
                if raw is None:
                    sample.candidate_errors.append(
                        {"stage": "scenario", "reason": f"candidate {k+1}/{K}: JSON parse failed"}
                    )
                    continue
                try:
                    seed, vignette = parse_scenario_result(
                        ctx, raw, min_toolkits=sample.min_toolkits, profile=sample.profile,
                    )
                    candidates.append((seed, vignette))
                except Exception as exc:
                    sample.candidate_errors.append(
                        {"stage": "scenario", "reason": f"candidate {k+1}/{K}: {exc}"}
                    )

            if not candidates:
                sample.error = "Scenario generation failed validation"
                LOGGER.warning("Sample %s: %s (0/%d valid)", sample.name, sample.error, K)
                for ce in sample.candidate_errors:
                    LOGGER.warning("  %s: %s: %s", sample.name, ce["stage"], ce["reason"])
                continue

            if len(candidates) == 1 or K == 1:
                best_seed, best_vignette = candidates[0]
            else:
                scored = [(self.diversity.score_scenario(s, v), s, v) for s, v in candidates]
                scored.sort(key=lambda x: x[0], reverse=True)
                _, best_seed, best_vignette = scored[0]
                LOGGER.debug(
                    "Sample %s: selected scenario candidate 1/%d (score=%.3f)",
                    sample.name, len(candidates), scored[0][0],
                )

            sample.seed_candidate = best_seed
            sample.vignette = best_vignette
            # Normalize scenario identifiers before counting diversity so rollback
            # uses the same canonical values if this sample later fails.
            self._fix_scenario_identifiers([sample])
            self.diversity.update_seed(sample.seed_candidate, vignette=sample.vignette)
            sample.diversity_counted = True

    def _fix_scenario_identifiers(self, states: List[SampleState]) -> None:
        """Replace lazy/duplicate identifiers in scenario fields with unique random values."""
        registry = self.identifier_registry
        for sample in states:
            if sample.error is not None or sample.vignette is None:
                continue
            registry.begin_sample()
            # Fix sensitive_info_items (primary source of identifiers)
            sample.vignette["sensitive_info_items"] = registry.fix_string_list(
                sample.vignette["sensitive_info_items"]
            )
            # Fix relevant_info_items (same treatment as sensitive items)
            sample.vignette["relevant_info_items"] = registry.fix_string_list(
                sample.vignette.get("relevant_info_items", [])
            )
            # Fix story (often echoes the same identifiers)
            sample.vignette["story"] = registry.fix_identifiers(
                sample.vignette["story"]
            )
            # Fix descriptor fields that may echo the same identifiers
            for desc_key in ("data_subject_concrete", "data_sender_concrete", "data_recipient_concrete"):
                if desc_key in sample.vignette:
                    sample.vignette[desc_key] = registry.fix_identifiers(
                        sample.vignette[desc_key]
                    )
            # Propagate to seed_candidate so both views stay in sync
            if sample.seed_candidate is not None:
                for seed_key in ("data_subject", "data_sender", "data_recipient"):
                    if seed_key in sample.seed_candidate:
                        sample.seed_candidate[seed_key] = registry.fix_identifiers(
                            sample.seed_candidate[seed_key]
                        )
            # Fix user_instruction (may reference identifiers)
            sample.vignette["user_instruction"] = registry.fix_identifiers(
                sample.vignette["user_instruction"]
            )
            # Store the replacement map on the sample for trajectory fixing later
            sample.id_replacements = registry.replacements
            if sample.id_replacements:
                LOGGER.debug(
                    "Sample %s: replaced %d lazy identifiers",
                    sample.name, len(sample.id_replacements),
                )

    def _run_vignette_quality_check(self, states: List[SampleState]) -> None:
        """Stage 3: quality-check vignettes before trajectory generation."""
        ctx = self.ctx
        active_states = [
            sample for sample in states
            if (
                sample.error is None
                and sample.seed_candidate is not None
                and sample.vignette is not None
            )
        ]
        LOGGER.info(
            "Stage 3: Quality-checking %d vignettes",
            len(active_states),
        )
        if not active_states:
            return

        prompts: List[str] = [
            build_vignette_quality_check_prompt(ctx, sample.vignette, sample.seed_candidate)
            for sample in active_states
        ]

        results = ctx.batch_call_json(
            prompts, max_tokens=self.args.max_tokens_per_stage, filter=True,
        )

        for sample, raw in zip(active_states, results):
            if raw is None:
                # Parse failure -> reject to avoid silently passing malformed judge outputs.
                reason = "parse_failed: model returned unparseable JSON"
                sample.candidate_errors.append(
                    {"stage": "vignette_quality_check", "reason": reason}
                )
                sample.error = "Vignette rejected by quality check"
                LOGGER.info("Sample %s: quality check rejected: %s", sample.name, reason)
                continue
            passed, issues = parse_vignette_quality_check_result(raw)
            if passed:
                continue
            reason = "; ".join(issues[:3])
            sample.candidate_errors.append(
                {"stage": "vignette_quality_check", "reason": reason}
            )
            sample.error = "Vignette rejected by quality check"
            LOGGER.info("Sample %s: quality check rejected: %s", sample.name, reason)

    def _run_trajectory_memories_stage(self, states: List[SampleState]) -> None:
        """Stage 4: generate trajectory + memories for each surviving sample."""
        ctx = self.ctx
        active_states = [
            sample for sample in states
            if (
                sample.error is None
                and sample.seed_candidate is not None
                and sample.vignette is not None
            )
        ]
        LOGGER.info(
            "Stage 4: Generating trajectories for %d samples",
            len(active_states),
        )
        if not active_states:
            return

        prompts: List[str] = [
            build_trajectory_memories_prompt(
                ctx, sample.profile, sample.seed_candidate, sample.vignette,
                min_steps=sample.min_tool_calls,
                sensitive_memories=sample.sensitive_memories,
            )
            for sample in active_states
        ]

        results = ctx.batch_call_json(prompts, max_tokens=self.args.max_tokens_per_stage)

        for sample, raw in zip(active_states, results):
            if raw is None:
                sample.candidate_errors.append(
                    {"stage": "trajectory_memories", "reason": "JSON parse failed"}
                )
                LOGGER.debug("Sample %s: trajectory JSON parse failed", sample.name)
                continue
            try:
                vignette_out, trajectory_payload = parse_and_validate_candidate(
                    ctx, raw, sample.seed_candidate, sample.vignette,
                    sample.profile, min_steps=sample.min_tool_calls,
                )
                sample.vignette = vignette_out
                sample.trajectory_payload = trajectory_payload
            except Exception as exc:
                sample.candidate_errors.append(
                    {"stage": "trajectory_memories", "reason": str(exc)}
                )
                LOGGER.debug("Sample %s: trajectory rejected: %s", sample.name, exc)

        # Fix identifiers in trajectory and memories
        self._fix_trajectory_identifiers(active_states)

        for sample in active_states:
            if sample.trajectory_payload is not None:
                continue
            sample.error = "Trajectory generation failed validation"
            LOGGER.warning("Sample %s: %s", sample.name, sample.error)

    @staticmethod
    def _normalize_hyphens(text: str) -> str:
        """Replace unicode dash variants with ASCII hyphen-minus."""
        for ch in ('\u2011', '\u2010', '\u2012', '\u2013', '\u2014', '\u2212'):
            text = text.replace(ch, '-')
        return text

    def _fix_trajectory_identifiers(self, states: List[SampleState]) -> None:
        """Apply identifier replacements to trajectory text and memories.

        The trajectory LLM sees the already-fixed sensitive_info_items, so it
        mostly echoes the correct values.  But it may also independently invent
        new lazy identifiers in tool-result JSON.  We:
          1. Re-seed the registry with the scenario-stage replacement map so
             those substitutions are applied first.
          2. Run fix_identifiers on the trajectory string to catch any NEW
             lazy identifiers introduced by the trajectory LLM.
          3. Apply all replacements to memories.
        """
        registry = self.identifier_registry
        for sample in states:
            if sample.trajectory_payload is None or sample.vignette is None:
                continue
            registry.begin_sample()
            # Re-load the scenario-stage replacements
            registry.restore_replacements(sample.id_replacements)
            # Fix trajectory text (catches both old replacements and new lazy IDs)
            traj = sample.trajectory_payload.get("trajectory", {})
            raw_traj = traj.get("executable_trajectory", "")
            # Normalize unicode hyphens to ASCII before any matching
            if isinstance(raw_traj, str) and raw_traj:
                raw_traj = self._normalize_hyphens(raw_traj)
                traj["executable_trajectory"] = raw_traj
            if isinstance(raw_traj, str) and raw_traj:
                traj["executable_trajectory"] = registry.fix_identifiers(raw_traj)
            # Reconcile: replace trajectory identifiers that diverge from
            # the canonical values in sensitive_info_items / relevant_info_items
            # (the LLM sometimes generates the right format but different digits).
            canonical = sample.vignette.get("sensitive_info_items", [])
            canonical_relevant = sample.vignette.get("relevant_info_items", [])
            if canonical or canonical_relevant:
                cur_traj = traj.get("executable_trajectory", "")
                if isinstance(cur_traj, str) and cur_traj:
                    if canonical:
                        cur_traj = reconcile_identifiers(canonical, cur_traj)
                    if canonical_relevant:
                        cur_traj = reconcile_identifiers(canonical_relevant, cur_traj)
                    traj["executable_trajectory"] = cur_traj
            # Fix memories — normalize unicode hyphens first
            memories = sample.vignette.get("memories", [])
            if memories:
                memories = [self._normalize_hyphens(m) for m in memories]
                sample.vignette["memories"] = registry.apply_replacements_to_list(memories)
            # Reconcile memories against canonical sensitive items too
            if canonical and memories:
                sample.vignette["memories"] = [
                    reconcile_identifiers(canonical, m)
                    for m in sample.vignette["memories"]
                ]
            # Keep memory identifiers aligned with relevant items as well.
            if canonical_relevant and memories:
                sample.vignette["memories"] = [
                    reconcile_identifiers(canonical_relevant, m)
                    for m in sample.vignette["memories"]
                ]
            # Keep parsed steps in sync with the reconciled trajectory text.
            # Downstream quality filtering reads `steps`, not the raw trajectory string.
            updated_traj = traj.get("executable_trajectory", "")
            reparsed_steps = parse_steps(updated_traj)
            if reparsed_steps:
                sample.trajectory_payload["steps"] = reparsed_steps
            else:
                LOGGER.debug(
                    "Sample %s: could not reparse trajectory after identifier reconciliation; "
                    "retaining original parsed steps",
                    sample.name,
                )

    def generate_batch(self, sample_names: List[str]) -> List[SampleState]:
        """Process a batch of samples through all pipeline stages using batched LLM calls.

        Stages: (1) profile generation, (2) scenario generation (combined seed+vignette),
        (3) vignette quality check, (4) trajectory+memories generation for survivors,
        (5) batch filtering (quality → naive agent → sensibility → leakage judge).
        """
        states: List[SampleState] = []
        try:
            self._append_sample_states(states, sample_names)
            self._run_profile_stage(states)
            self._run_scenario_stage(states)
            self._run_vignette_quality_check(states)
            self._run_trajectory_memories_stage(states)

            active_states = [sample for sample in states if sample.error is None]
            if active_states:
                self._batch_filter_all_samples(active_states)
        except Exception:
            self._rollback_partial_batch_state(states)
            raise
        return states

    def _batch_filter_all_samples(self, active: List[SampleState]) -> None:
        """Run quality and leakage filters in batched mode across all active samples."""
        quality_disabled = getattr(self.args, "disable_quality_filter", False)
        leakage_disabled = getattr(self.args, "disable_leakage_filter", False)
        link_passthrough_disabled = getattr(self.args, "disable_link_passthrough_filter", False)

        surviving = list(active)

        # Pipeline: quality filter → naive agent → sensibility check → leakage judge → link passthrough
        if not quality_disabled and surviving:
            surviving = self._run_quality_filter(surviving)
        if not leakage_disabled and surviving:
            surviving = self._run_naive_agent(surviving)
            if surviving:
                surviving = self._run_sensibility_check(surviving)
            if surviving:
                surviving = self._run_leakage_judge(surviving)
        if not link_passthrough_disabled and surviving:
            surviving = self._filter_link_passthrough(surviving)

        surviving_ids = {id(sample) for sample in surviving}

        for s in active:
            if id(s) not in surviving_ids:
                s.error = "Sample rejected by filters"
                LOGGER.warning("Sample %s: %s", s.name, s.error)
                continue

            if s.seed_candidate is None or s.vignette is None or s.trajectory_payload is None:
                s.error = "Internal state missing after filters"
                LOGGER.warning("Sample %s: %s", s.name, s.error)
                continue

            s.result = build_output_sample(
                self.args, s.name, s.profile, s.seed_candidate,
                s.vignette, s.trajectory_payload, s.leakage_info,
                sensitive_memories=s.sensitive_memories,
            )
            LOGGER.info("Sample %s: passed all filters", s.name)

    def _run_quality_filter(
        self, samples: List[SampleState],
    ) -> List[SampleState]:
        """Remove samples that fail the LLM quality judge."""
        ctx = self.ctx
        LOGGER.info("Quality filter: evaluating %d samples", len(samples))

        prompts: List[Optional[str]] = [
            build_quality_filter_prompt(ctx, s.trajectory_payload, s.vignette, s.seed_candidate)
            for s in samples
        ]

        real_indices: List[int] = []
        real_prompts: List[str] = []
        for idx, prompt in enumerate(prompts):
            if prompt is not None:
                real_indices.append(idx)
                real_prompts.append(prompt)

        if not real_prompts:
            return samples

        results = ctx.batch_call_json(
            real_prompts,
            max_tokens=self.args.max_tokens_per_stage,
            filter=True,
        )

        # Map results back
        parsed: List[Optional[Tuple[bool, List[str]]]] = [None] * len(prompts)
        for ri, raw in zip(real_indices, results):
            if raw is None:
                # Parse failure -> reject to avoid silently passing malformed judge outputs.
                parsed[ri] = (False, ["llm_check_failed: parse_failure"])
            else:
                parsed[ri] = parse_quality_filter_result(raw)

        surviving: List[SampleState] = []
        for idx, sample in enumerate(samples):
            result = parsed[idx]
            if result is None:
                surviving.append(sample)
            else:
                passed, issues = result
                if passed:
                    surviving.append(sample)
                else:
                    reason = "; ".join(issues[:3])
                    sample.candidate_errors.append({"stage": "quality_filter", "reason": reason})
                    LOGGER.info("Sample %s: quality filter rejected: %s", sample.name, reason)
        return surviving

    def _run_naive_agent(
        self, samples: List[SampleState],
    ) -> List[SampleState]:
        """Generate naive agent actions and keep samples with valid actions."""
        ctx = self.ctx
        LOGGER.info("Leakage phase 1: generating naive agent actions for %d samples", len(samples))

        prompts = [
            build_naive_agent_prompt(ctx, sample.trajectory_payload, sample.seed_candidate, sample.vignette)
            for sample in samples
        ]

        results = ctx.batch_call_text(
            prompts,
            max_tokens=self.args.max_tokens_per_stage,
            filter=True,
            enable_thinking=True,
        )

        surviving: List[SampleState] = []
        for sample, raw in zip(samples, results):
            if raw is None:
                sample.candidate_errors.append(
                    {"stage": "naive_agent", "reason": "agent returned empty"}
                )
                LOGGER.info("Sample %s: leakage filter: agent returned empty", sample.name)
                continue
            expected_action = sample.seed_candidate["final_action"]
            action = parse_naive_agent_result(raw, expected_action=expected_action)
            if action is not None:
                sample.agent_action = action
                surviving.append(sample)
            else:
                sample.candidate_errors.append(
                    {"stage": "naive_agent", "reason": "failed to parse agent action"}
                )
                LOGGER.info("Sample %s: leakage filter: failed to parse agent action", sample.name)
        return surviving

    def _run_sensibility_check(
        self, samples: List[SampleState],
    ) -> List[SampleState]:
        """Keep only samples whose agent action passes the sensibility check."""
        ctx = self.ctx
        LOGGER.info("Leakage phase 1.5: sensibility check for %d samples", len(samples))

        prompts = [
            build_sensibility_check_prompt(ctx, sample.trajectory_payload, sample.agent_action, vignette=sample.vignette)
            for sample in samples
        ]

        results = ctx.batch_call_text(
            prompts,
            max_tokens=self.args.max_tokens_per_stage,
            filter=True,
        )

        surviving: List[SampleState] = []
        for sample, raw in zip(samples, results):
            if raw is None:
                sample.candidate_errors.append(
                    {"stage": "sensibility_check", "reason": "returned empty"}
                )
                LOGGER.info("Sample %s: sensibility check returned empty, skipping", sample.name)
                continue
            sensible = parse_sensibility_check_result(raw)
            if sensible is None:
                sample.candidate_errors.append(
                    {"stage": "sensibility_check", "reason": f"unparseable: {raw.strip()}"}
                )
                LOGGER.warning("Sample %s: sensibility check unparseable, skipping", sample.name)
            elif sensible:
                surviving.append(sample)
                LOGGER.debug("Sample %s: sensibility PASSED", sample.name)
            else:
                sample.candidate_errors.append(
                    {"stage": "sensibility_check", "reason": raw.strip()}
                )
                LOGGER.info("Sample %s: sensibility REJECTED: %s", sample.name, raw.strip())
        return surviving

    def _run_leakage_judge(
        self, samples: List[SampleState],
    ) -> List[SampleState]:
        """Keep only samples judged as leaking privacy-sensitive info."""
        ctx = self.ctx
        LOGGER.info("Leakage phase 2: judging %d samples", len(samples))

        prompts = [
            build_leakage_judge_prompt(
                ctx, sample.trajectory_payload, sample.seed_candidate,
                sample.vignette, sample.agent_action,
            )
            for sample in samples
        ]

        results = ctx.batch_call_text(
            prompts,
            max_tokens=self.args.max_tokens_per_stage,
            filter=True,
        )

        surviving: List[SampleState] = []
        for sample, raw in zip(samples, results):
            if raw is None:
                sample.candidate_errors.append(
                    {"stage": "leakage_judge", "reason": "returned empty"}
                )
                LOGGER.info("Sample %s: leakage judge returned empty", sample.name)
                continue
            leaks = parse_leakage_judge_result(raw)
            info = {
                "generated_final_action": sample.agent_action,
                "leakage_judgment": leaks,
                "leakage_judge_output": raw,
            }
            if leaks is None:
                sample.candidate_errors.append(
                    {"stage": "leakage_judge", "reason": f"unparseable: {raw.strip()}"}
                )
                LOGGER.warning("Sample %s: leakage judge unparseable, skipping", sample.name)
            elif leaks:
                sample.leakage_info = info
                surviving.append(sample)
                LOGGER.debug("Sample %s: leakage PASSED (leaked)", sample.name)
            else:
                sample.candidate_errors.append(
                    {"stage": "leakage_judge", "reason": f"did not leak: {raw.strip()}"}
                )
                LOGGER.info("Sample %s: leakage REJECTED (did not leak)", sample.name)
        return surviving

    def _filter_link_passthrough(
        self, samples: List[SampleState],
    ) -> List[SampleState]:
        """Remove samples whose leakage is solely a forwarded pre-existing link."""
        LOGGER.info("Link-passthrough filter: checking %d samples", len(samples))
        surviving: List[SampleState] = []
        for sample in samples:
            traj = (sample.trajectory_payload or {}).get("trajectory", {})
            leakage_judge_output = (sample.leakage_info or {}).get("leakage_judge_output", "")
            generated_action = (sample.leakage_info or {}).get("generated_final_action", "")
            if is_link_passthrough(
                user_instruction=traj.get("user_instruction", ""),
                leakage_judge_output=leakage_judge_output,
                generated_final_action=generated_action,
                executable_trajectory=traj.get("executable_trajectory", ""),
                relevant_info_items=(sample.vignette or {}).get("relevant_info_items", []),
                memories=(sample.vignette or {}).get("memories", []),
                sensitive_info_items=(sample.vignette or {}).get("sensitive_info_items", []),
            ):
                sample.candidate_errors.append(
                    {"stage": "link_passthrough_filter", "reason": "leakage caused by forwarded link"}
                )
                LOGGER.info("Sample %s: link-passthrough filter rejected", sample.name)
            else:
                surviving.append(sample)
        return surviving

    def run(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        output_path = Path(self.args.output_path)
        error_path = Path(self.args.error_path)
        batch_size = self.args.batch_size

        existing: List[Dict[str, Any]] = []
        if self.args.resume and output_path.exists():
            with output_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                existing = loaded
                LOGGER.info("Resuming from %d existing samples in %s", len(existing), output_path)
                self.diversity.warm_from_samples(existing)

        generated = list(existing)
        errors: List[Dict[str, Any]] = []
        start_offset = len(existing)
        num_samples = self.args.num_samples

        # Process in chunks of batch_size
        for chunk_start in range(0, num_samples, batch_size):
            chunk_end = min(chunk_start + batch_size, num_samples)
            chunk_size = chunk_end - chunk_start

            sample_names = [
                f"{self.args.case_prefix}{self.args.start_id + start_offset + chunk_start + i}"
                for i in range(chunk_size)
            ]

            LOGGER.info(
                "Processing batch of %d samples (%s .. %s)",
                chunk_size, sample_names[0], sample_names[-1],
            )

            try:
                states = self.generate_batch(sample_names)
            except Exception as exc:
                # Catastrophic batch failure — record all samples as errors
                LOGGER.warning("Batch failed catastrophically: %s", exc)
                LOGGER.debug("Traceback:\n%s", traceback.format_exc())
                for name in sample_names:
                    error_payload = {"name": name, "error": f"{type(exc).__name__}: {exc}"}
                    errors.append(error_payload)
                    append_jsonl(error_path, error_payload)
                continue

            # Separate successes and errors; rollback diversity for failed
            # samples that were eagerly counted in Stage 2.
            successes = [(s, s.result) for s in states if s.result is not None]
            for s in states:
                if s.error:
                    if s.diversity_counted and s.seed_candidate is not None:
                        self.diversity.rollback_seed(s.seed_candidate, vignette=s.vignette)
                        s.diversity_counted = False
                    if s.profile is not None:
                        self.diversity.rollback_profile(s.profile)
                    self.diversity.rollback_name(s.first_name)
                    error_payload = {"name": s.name, "error": s.error}
                    if s.candidate_errors:
                        error_payload["candidate_errors"] = s.candidate_errors
                    errors.append(error_payload)
                    append_jsonl(error_path, error_payload)

            for s, r in successes:
                if not s.diversity_counted:
                    self._update_diversity_counters(s.seed_candidate, vignette=s.vignette)
                generated.append(r)

            # Write after each batch
            write_json_atomic(output_path, generated)

        return generated, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate privacy-scenario benchmark data with a staged pipeline.")

    # -- Output & control --
    out = parser.add_argument_group("Output & control")
    out.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output JSON path.")
    out.add_argument("--error-path", type=str, default=str(DEFAULT_ERROR_PATH), help="JSONL file for generation errors.")
    out.add_argument("--num-samples", type=int, default=20, help="Number of new samples to generate.")
    out.add_argument("--start-id", type=int, default=1, help="Starting numeric id for sample names.")
    out.add_argument("--case-prefix", type=str, default="main", help="Sample name prefix.")
    out.add_argument("--resume", action="store_true", help="Resume from existing output JSON if present.")
    out.add_argument("--source", type=str, default="synthetic_privacyalign_style", help="seed.source field value.")
    out.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include extended diagnostic metadata such as profile hashes in outputs.",
    )
    out.add_argument("--print-prompts", action="store_true", help="Log full prompts before model calls.")
    out.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    out.add_argument("--seed", type=int, default=42, help="Random seed.")

    # -- Model & backend --
    model = parser.add_argument_group("Model & backend")
    model.add_argument(
        "--backend",
        type=str,
        default="vllm_offline",
        choices=["vllm_offline", "auto"],
        help="Model backend: vllm_offline (in-process) or auto (OpenRouter).",
    )
    model.add_argument(
        "--generator-model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model name for generation.",
    )
    model.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for offline vLLM.",
    )
    model.add_argument(
        "--vllm-enable-expert-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable expert parallel for offline vLLM (use --no-vllm-enable-expert-parallel to disable).",
    )
    model.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force eager execution in offline vLLM (use --no-vllm-enforce-eager to disable).",
    )
    model.add_argument(
        "--vllm-language-model-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only load the language model, skipping multimodal components (use --no-vllm-language-model-only to disable).",
    )
    model.add_argument(
        "--vllm-enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable automatic prefix caching for shared prompt prefixes (use --no-vllm-enable-prefix-caching to disable).",
    )
    model.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use for vLLM (default 0.9). "
             "Lower values leave more headroom for MoE kernel workspace and reduce OOM risk.",
    )
    model.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Maximum model context length for vLLM. If not set, vLLM uses the model's default.",
    )
    model.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty for sampling (vLLM offline only, default 0.0).",
    )
    model.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for sampling (vLLM offline only, default 1.0).",
    )
    model.add_argument(
        "--filter-top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling for filter/judge stages only (vLLM offline only, default 1.0 = disabled).",
    )
    model.add_argument(
        "--filter-top-k",
        type=int,
        default=-1,
        help="Top-k sampling for filter/judge stages only (vLLM offline only, default -1 = disabled).",
    )
    model.add_argument(
        "--speculative-config",
        type=str,
        default=None,
        help='JSON string for vLLM speculative decoding config, e.g. \'{"method":"qwen3_next_mtp","num_speculative_tokens":2}\'.',
    )
    model.add_argument(
        "--vllm-kv-cache-dtype",
        type=str,
        default="auto",
        choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
        help="Data type for KV cache in vLLM. 'fp8' reduces memory usage, enabling higher throughput "
             "and longer contexts. Requires CUDA 11.8+ (default: auto).",
    )
    model.add_argument("--hf-cache-dir", type=str, default=None, help="Optional HF cache dir for offline vLLM downloads.")
    model.add_argument(
        "--reasoning-effort",
        type=str,
        default="high",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Optional Harmony reasoning effort for GPT-OSS models (offline vLLM only).",
    )
    model.add_argument(
        "--filter-reasoning-effort",
        type=str,
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for filter/judge stages. Defaults to --reasoning-effort if not set.",
    )
    model.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable thinking/reasoning for models that support it (use --no-enable-thinking to disable).",
    )
    model.add_argument("--max-tokens-per-stage", type=int, default=64000, help="Max tokens for each generation stage.")
    model.add_argument(
        "--diverse-generation-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation stages (profile, seeds, trajectories).",
    )
    model.add_argument(
        "--filter-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for filter/judge stages (quality filter, naive agent, sensibility check, leakage judge).",
    )

    # -- Resource paths --
    res = parser.add_argument_group("Resource paths")
    res.add_argument("--names-path", type=str, default=str(DEFAULT_NAMES_PATH), help="Path to name_frequency.jsonl.")
    res.add_argument(
        "--seed-options-path",
        type=str,
        default=str(DEFAULT_SEED_OPTIONS_PATH),
        help="Path to JSON file containing seed option lists.",
    )
    res.add_argument(
        "--prompts-dir",
        type=str,
        default=str(DEFAULT_PROMPTS_DIR),
        help="Directory containing stage prompt templates.",
    )
    res.add_argument(
        "--bootstrap-data-path",
        type=str,
        default=None,
        help="Optional existing dataset path to warm diversity counters.",
    )
    res.add_argument(
        "--toolkit-specs-path",
        type=str,
        default=str(DEFAULT_TOOLKIT_SPECS_PATH),
        help="Path to all_toolkits.json for parameter schema validation.",
    )

    # -- Pipeline parameters --
    pipe = parser.add_argument_group("Pipeline parameters")
    pipe.add_argument("--batch-size", type=int, default=8, help="Number of samples to process per batch. Set to 1 for sequential behavior.")
    pipe.add_argument(
        "--sensitive-memory-prob",
        type=float,
        default=0.5,
        help="Probability that a sample's memories contain sensitive information (0.0-1.0).",
    )
    pipe.add_argument("--min-tool-calls", type=int, default=3, help="Lower bound for per-sample minimum tool calls in executable_trajectory.")
    pipe.add_argument("--max-tool-calls", type=int, default=8, help="Upper bound for per-sample minimum tool calls in executable_trajectory.")
    pipe.add_argument("--max-trajectory-steps", type=int, default=32, help="Maximum allowed trajectory steps. Samples exceeding this are rejected. 0 = no limit.")
    pipe.add_argument("--max-trajectory-length", type=int, default=65536, help="Maximum allowed character length of executable_trajectory. Samples exceeding this are rejected. 0 = no limit.")
    pipe.add_argument("--min-toolkits", type=int, default=2, help="Lower bound for per-sample minimum toolkit count in seed generation.")
    pipe.add_argument("--max-toolkits", type=int, default=4, help="Upper bound for per-sample minimum toolkit count in seed generation.")
    pipe.add_argument(
        "--overgeneration-factor",
        type=int,
        default=4,
        help="Number of candidates to generate per sample in profile and scenario stages. "
             "The most diverse candidate is selected. Set to 1 to disable overgeneration.",
    )

    # -- Diversity --
    div = parser.add_argument_group("Diversity")
    div.add_argument("--name-frequency-alpha", type=float, default=0.35, help="Frequency smoothing exponent for name sampling.")
    div.add_argument("--name-repeat-penalty", type=float, default=2.0, help="Penalty factor for repeated first names.")
    # -- Filters --
    flt = parser.add_argument_group("Filters")
    flt.add_argument("--disable-quality-filter", action="store_true", help="Skip quality filter entirely.")
    flt.add_argument(
        "--disable-leakage-filter",
        action="store_true",
        help="Skip leakage filter (naive agent action generation + zero-shot leakage judgment).",
    )
    flt.add_argument(
        "--disable-link-passthrough-filter",
        action="store_true",
        help="Skip rule-based filter that rejects samples whose leakage is solely a forwarded pre-existing link.",
    )

    args = parser.parse_args()

    if args.min_tool_calls < 1:
        parser.error("--min-tool-calls must be >= 1.")
    if args.max_tool_calls < 1:
        parser.error("--max-tool-calls must be >= 1.")
    if args.min_tool_calls > args.max_tool_calls:
        parser.error("--min-tool-calls must be less than or equal to --max-tool-calls.")

    if args.min_toolkits < 1:
        parser.error("--min-toolkits must be >= 1.")
    if args.max_toolkits < 1:
        parser.error("--max-toolkits must be >= 1.")
    if args.min_toolkits > args.max_toolkits:
        parser.error("--min-toolkits must be less than or equal to --max-toolkits.")

    if args.overgeneration_factor < 1:
        parser.error("--overgeneration-factor must be >= 1.")

    return args


def main() -> None:
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    generator = ScenarioGenerator(args)
    try:
        generated, errors = generator.run()
        LOGGER.info("Generation finished. Total samples in output: %d", len(generated))
        if errors:
            LOGGER.warning("Encountered %d generation failures. See %s", len(errors), args.error_path)
    finally:
        model = getattr(generator, "model", None)
        if model is not None and hasattr(model, "close"):
            try:
                model.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass


if __name__ == "__main__":
    main()
