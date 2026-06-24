from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from utils.sampling_utils import validate_extra_sampling_kwargs


SUPPORTED_DTYPES = {"bfloat16", "float32"}
SUPPORTED_TRAINING_OBJECTIVES = {"policy_optimization"}
SUPPORTED_POLICY_SCORERS = {
    "privalign_rl_pairwise_judge",
    "privalign_pairwise_margin",
    "privalign_trained_genrm",
}
JUDGE_POLICY_SCORERS = frozenset(
    {
        "privalign_rl_pairwise_judge",
    }
)
TRAINED_GENRM_POLICY_SCORERS = frozenset(
    {"privalign_trained_genrm"}
)
DEFAULT_TRAINED_GENRM_PROMPT_TEMPLATES: dict[str, str] = {
    "privalign_trained_genrm": "privalign_genrm_pairwise",
}
ANNOTATION_CONDITIONED_TRAINED_GENRM_PROMPT_TEMPLATES: dict[str, str] = {
    "privalign_trained_genrm": "privalign_rl_pairwise_judge",
}


def resolve_trained_genrm_prompt_template(
    *,
    policy_scorer: str,
    override: str | None,
    annotation_conditioning: bool = False,
) -> str:
    """Return the rate-this template name for a trained-genrm policy scorer.

    Falls back to ``DEFAULT_TRAINED_GENRM_PROMPT_TEMPLATES`` when ``override``
    is ``None`` or empty. When ``annotation_conditioning`` is true the default
    is pulled from ``ANNOTATION_CONDITIONED_TRAINED_GENRM_PROMPT_TEMPLATES``
    instead (Privalign scorers only). An explicit ``override`` always wins.
    """
    if override and str(override).strip():
        return str(override).strip()
    if annotation_conditioning:
        annotated_default = ANNOTATION_CONDITIONED_TRAINED_GENRM_PROMPT_TEMPLATES.get(
            policy_scorer
        )
        if annotated_default is None:
            raise ValueError(
                "policy_trained_genrm_annotation_conditioning is only supported for Privalign "
                f"policy scorers; got policy_scorer={policy_scorer!r}."
            )
        return annotated_default
    default = DEFAULT_TRAINED_GENRM_PROMPT_TEMPLATES.get(policy_scorer)
    if default is None:
        raise ValueError(
            f"No default trained-genrm prompt template registered for policy_scorer={policy_scorer!r}."
        )
    return default
SUPPORTED_PENALTY_SHAPES = {"linear"}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _resolve_local_rank(requested_local_rank: int, *, rank: int) -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    if requested_local_rank >= 0:
        return requested_local_rank
    return rank


def normalize_policy_scorer_name(name: str) -> str:
    return name.strip().lower()


@dataclass
class TrainingConfig:
    """Configuration for the shared training/runtime stack."""

    # --- Required ---
    output_dir: str

    # --- DeepSpeed ---
    deepspeed: str = "configs/deepspeed/zero3.json"
    deepspeed_offload_optimizer: bool | None = None
    deepspeed_offload_param: bool | None = None

    # --- Training hyperparameters ---
    seed: int = 42
    learning_rate: float = 1e-6
    weight_decay: float = 0.001
    warmup_ratio: float = 0.05
    warmup_steps: int | None = None
    lr_scheduler_type: str = "constant_with_warmup"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    prompt_gradient_accumulation_steps: int | None = None
    global_prompt_batch_size: int | None = None
    max_grad_norm: float = 1.0
    bf16: bool = True
    attn_implementation: str | None = None
    dtype: str | None = "bfloat16"
    trust_remote_code: bool = False
    deepcompile: bool = False
    disable_dropout: bool = True
    gradient_checkpointing: bool = True
    log_phase_progress: bool = True
    # Run per-tensor `torch.isfinite` guards in the trainer hot path. Each guard forces a
    # D2H sync (~10+ per micro-batch), so leave off unless debugging a nan/inf issue.
    debug_finite_tensor_checks: bool = False
    # Insert distributed barriers around backward-phase debug logging. Useful for
    # diagnosing rank skew/hangs, but too expensive for normal training.
    debug_distributed_phase_barriers: bool = False
    training_objective: str = "policy_optimization"
    policy_scorer: str = "privalign_rl_pairwise_judge"
    policy_judge_model_name: str | None = None
    dev_eval_judge_model_name: str | None = None
    # Override the generation budget for the dev-eval LLM judge alone (separate
    # from the policy's max_completion_length). Long Privalign prompts plus
    # Qwen3-32B + thinking can saturate the default budget and emit nothing
    # parseable, producing invalid_judgments. None = inherit max_completion_length.
    dev_eval_judge_max_new_tokens: int | None = None
    policy_algorithm: str = "sapo"  # "sapo" (existing) or "vanilla" (importance-weighted REINFORCE, no clipping).
    policy_sapo_tau_pos: float = 1.0
    policy_sapo_tau_neg: float = 1.05
    policy_reward_kl_coef: float = 0.05
    policy_reward_clip_min: float = -1000.0
    policy_reward_clip_max: float = 1000.0
    # "peer" (default) = pairwise-within-group for every unordered pair
    # (C(G,2) calls per group, or 2*C(G,2) when policy_judge_dual_order=True).
    # "anchor" =
    # original anchor-style double-swap (rollout vs cached policy_reference_response).
    policy_judge_scoring_mode: str = "peer"
    # Whether pairwise policy scorers evaluate both response orderings for each
    # pair. Applies to LLM-judge and trained-GenRM policy scorers. Default False
    # halves in-training scorer calls; enable to recover double-swap
    # order-bias cancellation.
    policy_judge_dual_order: bool = False
    # Whether the in-training pairwise scorer model runs with Qwen-style
    # thinking on. Applies to LLM-judge and trained-GenRM policy scorers.
    # Default True preserves original behavior; turn off to keep scorer output
    # inside the token budget and avoid the "score inside <think> block /
    # truncated mid-think" failure mode.
    policy_judge_enable_thinking: bool = True
    # Optional override for scorer generation budget. LLM judges inherit
    # max_completion_length when unset; trained GenRM falls back to
    # policy_trained_genrm_max_new_tokens.
    policy_judge_max_new_tokens: int | None = None
    # Optional in-training policy-scorer sampling overrides. None preserves the
    # previous fallback to eval/judge-wide sampling settings.
    policy_judge_vllm_temperature: float | None = None
    policy_judge_vllm_top_p: float | None = None
    policy_judge_vllm_top_k: int | None = None
    policy_judge_vllm_presence_penalty: float | None = None
    policy_judge_vllm_repetition_penalty: float | None = None
    # Privalign judge prompt cap after thinking traces/special tokens have been
    # stripped from candidate/reference responses. 0 disables the cap.
    privalign_judge_max_response_words: int | None = 1000
    # Length penalty applied to the per-sequence reward to keep RL responses
    # comparable to the SFT-trained baselines. Default off (no penalty).
    policy_penalty_max_len: int | None = None        # words; trigger threshold
    policy_penalty_per_word: float = 0.0             # subtracted per word over max_len
    policy_penalty_max_value: float | None = 2.0      # cap on the total length penalty; None disables cap
    policy_penalty_shape: str = "linear"
    # Privalign-only relative undershortness penalty. For each prompt's rollout
    # group, compute floor_ratio * median(payload word counts) over parseable
    # JSON tool-call responses. Valid JSON responses below that floor receive a
    # capped penalty. Default max=0 disables the regularizer.
    policy_undershort_penalty_max: float = 0.0
    policy_undershort_floor_ratio: float = 0.5
    # Pairwise-margin scorer (Phase A of the trained-gen-RM baseline).
    # Gen-RM emits a single Score in [-2, +2] (eval pipeline scale). Reward:
    #   R = -policy_pairwise_margin_c1 * I_format
    #       + policy_pairwise_margin_c2 * (2 - |predicted_score - target_score|)
    #       + policy_pairwise_w_leak * leak_reward
    # where leak_reward sums the two per-response leak errors and ranges [-2, 0].
    policy_pairwise_margin_c1: float = 10.0
    policy_pairwise_margin_c2: float = 1.0
    # Optional Privalign pairwise GenRM auxiliary term. Active only when
    # ranking_demo carries gold_leak_response1/2 targets.
    policy_pairwise_w_leak: float = 1.0
    # Trained generative reward model (Phase B). Loaded into a temporary vLLM
    # engine via the existing judge lifecycle; outputs `Score: <int -2..2>`,
    # parsed as the rollout-vs-anchor signed preference margin.
    policy_trained_genrm_path: str | None = None
    policy_trained_genrm_max_new_tokens: int = 1024
    policy_trained_genrm_enable_thinking: bool = True
    # Override rate-this prompt template name. None ⇒ pick by policy_scorer
    # (DEFAULT_TRAINED_GENRM_PROMPT_TEMPLATES).
    policy_trained_genrm_prompt_template: str | None = None
    # Condition the trained gen-RM on dataset reference responses and
    # per-annotator leak/omit labels. When True the default template flips to
    # 'privalign_rl_pairwise_judge' and the rate-this render path fills
    # {reference_response_a}, {reference_response_b}, {annotator_block}.
    policy_trained_genrm_annotation_conditioning: bool = False
    # Composite cells: alpha in R = alpha * judge_score + (1 - alpha) * genrm_score.
    # Optional suffix appended to the last user turn of every training row.
    # Mirrors the SFT baselines' "Aim for about 100 words." injection so RL
    # and SFT see byte-identical prompts.
    user_prompt_suffix: str | None = None

    # --- Sequence lengths and generation ---
    max_prompt_length: int | None = 16384
    max_completion_length: int = 16384
    max_train_completion_length: int | None = None
    num_generations: int = 2
    train_vllm_temperature: float = 1.0
    eval_judge_vllm_temperature: float = 0.6
    eval_reference_vllm_temperature: float | None = None
    eval_reference_vllm_top_p: float = 0.95
    eval_reference_vllm_top_k: int | None = 20
    eval_reference_vllm_repetition_penalty: float | None = None
    eval_reference_vllm_presence_penalty: float | None = None
    dev_eval_reference_response_cache: str | None = None
    dev_eval_reference_response_cache_readonly: bool = False
    dev_eval_num_generations: int = 1
    dev_eval_loss: bool = False
    judge_vllm_top_p: float = 1.0
    judge_vllm_top_k: int | None = None
    judge_vllm_presence_penalty: float | None = None
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float = 1.0
    generation_kwargs: dict | None = None
    student_last_user_instruction: str | None = None
    qwen_enable_thinking: bool = True
    disable_student_thinking: bool = False

    # --- vLLM rollout engine ---
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.3
    vllm_enable_sleep_mode: bool = True
    vllm_enforce_eager: bool = False
    vllm_show_progress: bool = True

    # --- Reference model + lm-head chunking ---
    lm_head_chunk_size: int | None = 128
    ref_model_offload: bool = False

    # --- Checkpointing ---
    resume_from: str | None = None

    # --- Logging ---
    logging_steps: int = 1
    sample_log_steps: int = 1
    save_steps: int = 100
    dev_eval_steps: int = 0
    dev_eval_before_training: bool = False
    dev_eval_max_samples: int | None = None
    save_best_metric: str | None = None
    save_best_higher_is_better: bool = True
    report_to: Sequence[str] | str | None = None

    # --- Data loading ---
    dataloader_num_workers: int = 0

    # --- Derived (computed in __post_init__) ---
    local_rank: int = -1
    rank: int = field(init=False)
    world_size: int = field(init=False)
    local_prompt_batch_size: int = field(init=False)

    def __post_init__(self) -> None:
        self._resolve_derived_fields()
        self._validate_general_fields()
        self._validate_sequence_and_generation()
        self._validate_objective_constraints()
        self._validate_kl_chunking()

    # --- Derived field resolution ---

    def _resolve_derived_fields(self) -> None:
        self.output_dir = str(Path(self.output_dir))
        self.rank = _env_int("RANK", 0)
        self.world_size = _env_int("WORLD_SIZE", 1)
        self.local_rank = _resolve_local_rank(self.local_rank, rank=self.rank)
        self.policy_scorer = normalize_policy_scorer_name(self.policy_scorer)
        if self.prompt_gradient_accumulation_steps is None:
            self.prompt_gradient_accumulation_steps = self.gradient_accumulation_steps
        if self.global_prompt_batch_size is not None:
            if self.global_prompt_batch_size <= 0:
                raise ValueError("global_prompt_batch_size must be > 0 when set.")
            if self.global_prompt_batch_size % self.world_size != 0:
                raise ValueError(
                    "global_prompt_batch_size must be divisible by world_size so prompts shard evenly."
                )
            self.local_prompt_batch_size = self.global_prompt_batch_size // self.world_size
        else:
            self.local_prompt_batch_size = (
                self.per_device_train_batch_size * self.prompt_gradient_accumulation_steps
            )
        if self.dtype is not None and self.dtype not in SUPPORTED_DTYPES:
            raise ValueError("dtype must be one of: bfloat16, float32.")
        if self.dtype == "bfloat16":
            self.bf16 = True
        elif self.dtype == "float32":
            self.bf16 = False

    # --- General required fields and positive-integer constraints ---

    def _validate_general_fields(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir must be set.")
        if not self.deepspeed or not self.deepspeed.strip():
            raise ValueError("deepspeed must be set.")
        for field_name in (
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "prompt_gradient_accumulation_steps",
            "num_train_epochs",
            "num_generations",
            "max_completion_length",
            "vllm_tensor_parallel_size",
            "logging_steps",
            "save_steps",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be > 0.")
        if self.sample_log_steps < 0:
            raise ValueError("sample_log_steps must be >= 0.")
        if self.dev_eval_steps < 0:
            raise ValueError("dev_eval_steps must be >= 0.")
        if self.dev_eval_max_samples is not None and self.dev_eval_max_samples <= 0:
            raise ValueError("dev_eval_max_samples must be > 0 when set.")
        if self.lm_head_chunk_size is not None and self.lm_head_chunk_size <= 0:
            raise ValueError("lm_head_chunk_size must be > 0 when set.")
        if not 0.0 < self.vllm_gpu_memory_utilization <= 1.0:
            raise ValueError("vllm_gpu_memory_utilization must be in the range (0, 1].")

    # --- Sequence lengths and generation sampling ---

    def _validate_sequence_and_generation(self) -> None:
        if self.max_train_completion_length is not None and self.max_train_completion_length <= 0:
            raise ValueError("max_train_completion_length must be > 0 when set.")
        if self.train_vllm_temperature < 0.0:
            raise ValueError("train_vllm_temperature must be >= 0.")
        if self.eval_judge_vllm_temperature <= 0.0:
            raise ValueError("eval_judge_vllm_temperature must be > 0.")
        if self.policy_judge_vllm_temperature is not None and self.policy_judge_vllm_temperature <= 0.0:
            raise ValueError("policy_judge_vllm_temperature must be > 0 when provided.")
        if self.policy_judge_vllm_top_p is not None and not 0.0 < self.policy_judge_vllm_top_p <= 1.0:
            raise ValueError("policy_judge_vllm_top_p must be in the range (0, 1] when provided.")
        if self.policy_judge_vllm_top_k is not None and self.policy_judge_vllm_top_k <= 0:
            raise ValueError("policy_judge_vllm_top_k must be > 0 when set.")
        if (
            self.policy_judge_vllm_repetition_penalty is not None
            and self.policy_judge_vllm_repetition_penalty <= 0.0
        ):
            raise ValueError(
                "policy_judge_vllm_repetition_penalty must be > 0 when set."
            )
        if (
            self.privalign_judge_max_response_words is not None
            and self.privalign_judge_max_response_words < 0
        ):
            raise ValueError("privalign_judge_max_response_words must be >= 0 when set.")
        if (
            self.eval_reference_vllm_temperature is not None
            and self.eval_reference_vllm_temperature <= 0.0
        ):
            raise ValueError("eval_reference_vllm_temperature must be > 0 when provided.")
        if not 0.0 < self.eval_reference_vllm_top_p <= 1.0:
            raise ValueError("eval_reference_vllm_top_p must be in the range (0, 1].")
        if self.eval_reference_vllm_top_k is not None and self.eval_reference_vllm_top_k <= 0:
            raise ValueError("eval_reference_vllm_top_k must be > 0 when set.")
        if (
            self.eval_reference_vllm_repetition_penalty is not None
            and self.eval_reference_vllm_repetition_penalty <= 0.0
        ):
            raise ValueError(
                "eval_reference_vllm_repetition_penalty must be > 0 when set."
            )
        if not 0.0 < self.judge_vllm_top_p <= 1.0:
            raise ValueError("judge_vllm_top_p must be in the range (0, 1].")
        if self.judge_vllm_top_k is not None and self.judge_vllm_top_k <= 0:
            raise ValueError("judge_vllm_top_k must be > 0 when set.")
        validate_extra_sampling_kwargs(self.generation_kwargs)

    # --- Training objective and policy ---

    def _validate_objective_constraints(self) -> None:
        if self.training_objective not in SUPPORTED_TRAINING_OBJECTIVES:
            supported = ", ".join(sorted(SUPPORTED_TRAINING_OBJECTIVES))
            raise ValueError(f"training_objective must be one of: {supported}.")
        if self.training_objective == "policy_optimization":
            if self.policy_scorer not in SUPPORTED_POLICY_SCORERS:
                supported = ", ".join(sorted(SUPPORTED_POLICY_SCORERS))
                raise ValueError(f"policy_scorer must be one of: {supported}.")
            if (
                self.policy_scorer in JUDGE_POLICY_SCORERS
                and not self.policy_judge_model_name
            ):
                raise ValueError(
                    "policy_judge_model_name must be set when using an RL LLM-judge policy_scorer."
                )
            if (
                self.policy_scorer in TRAINED_GENRM_POLICY_SCORERS
                and not self.policy_trained_genrm_path
            ):
                raise ValueError(
                    f"policy_trained_genrm_path must be set when policy_scorer={self.policy_scorer!r}."
                )
            if (
                self.policy_trained_genrm_prompt_template is not None
                and not str(self.policy_trained_genrm_prompt_template).strip()
            ):
                raise ValueError(
                    "policy_trained_genrm_prompt_template must be a non-empty string when set."
                )
            if (
                self.policy_pairwise_margin_c1 < 0.0
                or self.policy_pairwise_margin_c2 < 0.0
                or self.policy_pairwise_w_leak < 0.0
            ):
                raise ValueError("policy_pairwise_margin_c1/c2 and policy_pairwise_w_leak must be >= 0.")
            if self.policy_trained_genrm_max_new_tokens <= 0:
                raise ValueError("policy_trained_genrm_max_new_tokens must be > 0.")
            if self.policy_algorithm not in {"sapo", "vanilla"}:
                raise ValueError(
                    f"policy_algorithm must be 'sapo' or 'vanilla'; got {self.policy_algorithm!r}."
                )
            if self.policy_sapo_tau_pos <= 0.0:
                raise ValueError("policy_sapo_tau_pos must be > 0.")
            if self.policy_sapo_tau_neg <= 0.0:
                raise ValueError("policy_sapo_tau_neg must be > 0.")
            if self.policy_reward_kl_coef < 0.0:
                raise ValueError("policy_reward_kl_coef must be >= 0.")
            if self.policy_penalty_max_len is not None and self.policy_penalty_max_len < 0:
                raise ValueError("policy_penalty_max_len must be >= 0 when set.")
            if self.policy_penalty_per_word < 0.0:
                raise ValueError("policy_penalty_per_word must be >= 0.")
            if self.policy_penalty_max_value is not None and self.policy_penalty_max_value < 0.0:
                raise ValueError("policy_penalty_max_value must be >= 0 when set.")
            if self.policy_penalty_shape not in SUPPORTED_PENALTY_SHAPES:
                raise ValueError("policy_penalty_shape must be 'linear'.")
            if self.policy_undershort_penalty_max < 0.0:
                raise ValueError("policy_undershort_penalty_max must be >= 0.")
            if self.policy_undershort_penalty_max > 0.0 and self.policy_undershort_floor_ratio <= 0.0:
                raise ValueError(
                    "policy_undershort_floor_ratio must be > 0 when undershort penalty is enabled."
                )
            if self.policy_reward_clip_min > self.policy_reward_clip_max:
                raise ValueError("policy_reward_clip_min must be <= policy_reward_clip_max.")
            if self.num_generations <= 1:
                raise ValueError(
                    "policy_optimization requires num_generations > 1 so each prompt has a "
                    "reward baseline group (REINFORCE++-baseline)."
                )
    # --- KL support ---

    def _validate_kl_chunking(self) -> None:
        if self.policy_reward_kl_coef > 0.0 and self.lm_head_chunk_size is None:
            raise ValueError(
                "policy_reward_kl_coef > 0 requires lm_head_chunk_size to be set so the "
                "full-vocab reference KL kernel can stream the lm-head pass."
            )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def to_dict(self) -> dict:
        return asdict(self)
