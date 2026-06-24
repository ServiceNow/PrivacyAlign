from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
from typing import Any

import torch

def resolve_deepspeed_config(config_path: str) -> str:
    resolved_path = Path(config_path.strip())
    if not resolved_path.is_absolute():
        resolved_path = Path(__file__).resolve().parent / resolved_path
    resolved_path = resolved_path.resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(f"DeepSpeed config not found: {resolved_path}")

    return str(resolved_path)


def normalize_report_to(value: str) -> list[str] | None:
    normalized = value.strip().lower()
    if normalized in {"", "none", "off", "disable"}:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_generation_kwargs(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"generation_kwargs must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("generation_kwargs must decode to a JSON object.")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preference-conditioned training runner")
    parser.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", type=str, default="preference", help="Dataset type")
    parser.add_argument("--dataset_name", type=str, default="privalign-dataset", help="Dataset name or path")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Optional HuggingFace dataset config/subset.",
    )
    parser.add_argument(
        "--train-max-samples",
        dest="train_max_samples",
        type=int,
        default=None,
        help="Optional cap on the number of training examples loaded before prompt-length filtering.",
    )
    parser.add_argument(
        "--student-last-user-instruction",
        dest="student_last_user_instruction",
        type=str,
        default=None,
        help=(
            "Optional instruction appended to the last user turn for student/reference rollouts."
        ),
    )
    parser.add_argument(
        "--deduplicate_preference_examples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate preference examples that produce identical training prompts before formatting.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="AdamW weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="LR warmup as fraction of total steps")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Absolute number of LR warmup steps. Overrides --warmup_ratio when set.",
    )
    parser.add_argument(
        "--training_objective",
        type=str,
        default="policy_optimization",
        choices=["policy_optimization"],
        help="Optimization objective used by the trainer loop. Only policy RL is supported.",
    )
    parser.add_argument(
        "--policy_scorer",
        type=str,
        default="privalign_rl_pairwise_judge",
        choices=[
            "privalign_rl_pairwise_judge",
            "privalign_pairwise_margin",
            "privalign_trained_genrm",
        ],
        help=(
            "Coordinator-side trajectory scorer used when --training_objective=policy_optimization. "
            "'privalign_rl_pairwise_judge' compares sampled completions within each prompt "
            "group, conditioning the judge on leak/omit annotations for the two reference responses. "
            "'privalign_pairwise_margin' trains a Privalign pairwise gen-RM: the model "
            "emits Response 1/2 leak labels plus `Score: <int -2..2>` for two embedded "
            "Privalign responses. Reward = -c1*format_violation + c2*margin_reward "
            "+ w_leak*leak_reward - length_penalty. The data builder emits both "
            "orderings of each pair so position bias washes out. "
            "'privalign_trained_genrm' renders the Privalign rate-this prompt "
            "(user_instruction + memories + executable_trajectory + 2 responses) from each "
            "example's judge_demo."
        ),
    )
    parser.add_argument(
        "--policy_judge_model_name",
        type=str,
        default=None,
        help=(
            "Model path used for LLM-judge scoring with RL pairwise judge policy scorers. "
            "Defaults to --model_name."
        ),
    )
    parser.add_argument(
        "--dev-eval-judge-model-name",
        type=str,
        default=None,
        help=(
            "Model path used for preference dev-eval judging. Reference-response generation still uses "
            "the frozen reference model. Defaults to --model_name."
        ),
    )
    parser.add_argument(
        "--dev-eval-judge-max-new-tokens",
        type=int,
        default=None,
        help=(
            "Override the generation budget for the dev-eval LLM judge alone "
            "(does not affect policy rollouts). Useful when a long-prompt + thinking judge "
            "model truncates before emitting a parseable verdict, inflating invalid_judgments. "
            "None = inherit --max_completion_length."
        ),
    )
    parser.add_argument(
        "--policy_algorithm",
        type=str,
        default="sapo",
        choices=["sapo", "vanilla"],
        help=(
            "Policy-gradient loss variant when --training_objective=policy_optimization. "
            "'sapo' (default): the SAPO soft-clipped surrogate. "
            "'vanilla': importance-weighted REINFORCE without clipping or soft gating "
            "(per-token loss = -(ratio * advantage)). Use 'vanilla' for a neutral RL baseline."
        ),
    )
    parser.add_argument(
        "--policy_sapo_tau_pos",
        type=float,
        default=1.0,
        help="SAPO temperature for positive-advantage tokens (ignored when --policy_algorithm=vanilla).",
    )
    parser.add_argument(
        "--policy_sapo_tau_neg",
        type=float,
        default=1.05,
        help="SAPO temperature for non-positive-advantage tokens.",
    )
    parser.add_argument(
        "--policy_reward_kl_coef",
        type=float,
        default=0.05,
        help="Optional coefficient for an explicit reference-KL loss term added to the policy objective.",
    )
    parser.add_argument(
        "--user_prompt_suffix",
        type=str,
        default=None,
        help=(
            "If set, appended to the last user turn of every loaded row. "
            "Mirrors the SFT baselines' 'Aim for about 100 words.' suffix injection so RL "
            "and SFT see byte-identical prompts."
        ),
    )
    parser.add_argument(
        "--policy_judge_scoring_mode",
        type=str,
        default="peer",
        choices=["anchor", "peer"],
        help=(
            "How the pairwise LLM judge scores rollouts. 'peer' (default): "
            "pairwise-within-group for every unordered pair "
            "(C(num_generations, 2) calls per group, or double that with "
            "--policy_judge_dual_order), per-rollout reward = mean signed margin "
            "over peers. 'anchor': each rollout judged vs the cached base anchor."
        ),
    )
    parser.add_argument(
        "--policy_judge_dual_order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether in-training pairwise policy scorers score both response orderings "
            "for each pair. Applies to LLM-judge and trained-GenRM policy scorers. "
            "Default False halves scorer calls; pass --policy_judge_dual_order to "
            "recover double-swap order-bias cancellation."
        ),
    )
    parser.add_argument(
        "--policy_judge_enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether the in-training pairwise scorer model runs with Qwen thinking on. "
            "Applies to LLM-judge and trained-GenRM policy scorers. Default True "
            "(original behavior). Pass --no-policy_judge_enable_thinking to disable "
            "thinking when traces eat the token budget before the Score: line."
        ),
    )
    parser.add_argument(
        "--policy_judge_max_new_tokens",
        type=int,
        default=None,
        help=(
            "Optional override for in-training scorer max_new_tokens. Applies to "
            "LLM-judge and trained-GenRM policy scorers. For trained GenRM, None "
            "falls back to --policy_trained_genrm_max_new_tokens."
        ),
    )
    parser.add_argument(
        "--policy-judge-vllm-temperature",
        dest="policy_judge_vllm_temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature for the in-training policy scorer model. Applies "
            "to LLM-judge and trained-GenRM policy scorers. When unset, falls back "
            "to --eval-judge-vllm-temperature."
        ),
    )
    parser.add_argument(
        "--policy-judge-vllm-top-p",
        dest="policy_judge_vllm_top_p",
        type=float,
        default=None,
        help=(
            "Top-p sampling for the in-training policy scorer model. Applies to "
            "LLM-judge and trained-GenRM policy scorers. When unset, falls back to "
            "--judge-vllm-top-p."
        ),
    )
    parser.add_argument(
        "--policy-judge-vllm-top-k",
        dest="policy_judge_vllm_top_k",
        type=int,
        default=None,
        help=(
            "Top-k sampling for the in-training policy scorer model. Applies to "
            "LLM-judge and trained-GenRM policy scorers. When unset, falls back to "
            "--judge-vllm-top-k."
        ),
    )
    parser.add_argument(
        "--policy-judge-vllm-presence-penalty",
        dest="policy_judge_vllm_presence_penalty",
        type=float,
        default=None,
        help=(
            "Presence penalty for the in-training policy LLM judge. "
            "When unset, falls back to --judge-vllm-presence-penalty."
        ),
    )
    parser.add_argument(
        "--policy-judge-vllm-repetition-penalty",
        dest="policy_judge_vllm_repetition_penalty",
        type=float,
        default=None,
        help=(
            "Repetition penalty for the in-training policy LLM judge. "
            "When unset, no repetition penalty is applied."
        ),
    )
    parser.add_argument(
        "--privalign_judge_max_response_words",
        type=int,
        default=1000,
        help=(
            "Maximum words from each stripped Privalign response included in pairwise "
            "judge prompts. Default 1000; pass 0 to disable the judge prompt cap."
        ),
    )
    parser.add_argument(
        "--policy_penalty_max_len",
        type=int,
        default=None,
        help=(
            "Length-penalty trigger threshold (in WORDS). When the completion exceeds "
            "this length, subtract --policy_penalty_per_word per extra word from the "
            "sequence-level reward before computing advantages. Default None disables "
            "the penalty."
        ),
    )
    parser.add_argument(
        "--policy_penalty_per_word",
        type=float,
        default=0.0,
        help="Penalty subtracted from the sequence-level reward per word above --policy_penalty_max_len.",
    )
    parser.add_argument(
        "--policy_penalty_max_value",
        type=float,
        default=2.0,
        help=(
            "Maximum total overlength penalty subtracted from a sequence reward. "
            "Default 2.0 keeps the length penalty on the same scale as the pairwise "
            "judge margin. Negative values are invalid; set this to a large value "
            "to approximate the old unbounded behavior."
        ),
    )
    parser.add_argument(
        "--policy_penalty_shape",
        type=str,
        default="linear",
        choices=["linear"],
        help="Penalty shape for overlong policy responses. Only linear penalties are supported.",
    )
    parser.add_argument(
        "--policy_undershort_penalty_max",
        type=float,
        default=0.0,
        help=(
            "Maximum Privalign relative undershortness penalty. Default 0 disables it. "
            "When enabled, valid JSON tool-call responses much shorter than valid peers "
            "in the same rollout group have up to this value subtracted from reward."
        ),
    )
    parser.add_argument(
        "--policy_undershort_floor_ratio",
        type=float,
        default=0.5,
        help=(
            "Privalign undershort floor as a fraction of the median payload word count "
            "among parseable JSON responses in the same prompt group. Default 0.5."
        ),
    )
    parser.add_argument(
        "--policy_pairwise_margin_c1",
        type=float,
        default=10.0,
        help=(
            "Format-violation coefficient C1 in the pairwise-margin reward "
            "R = -C1*I_format + C2*(2 - |predicted_score - target_score|) "
            "+ w_leak*leak_reward. Active when --policy_scorer=privalign_pairwise_margin. "
            "Default 10 makes malformed outputs decisively worse than valid wrong answers."
        ),
    )
    parser.add_argument(
        "--policy_pairwise_margin_c2",
        type=float,
        default=1.0,
        help=(
            "Signed-margin coefficient C2 in the pairwise-margin reward. Active when "
            "--policy_scorer=privalign_pairwise_margin. Default 1 leaves the parsed reward "
            "in [-2, +2] matching the eval pipeline score scale."
        ),
    )
    parser.add_argument(
        "--policy_pairwise_w_leak",
        type=float,
        default=1.0,
        help=(
            "Weight on the auxiliary per-response leak term for pairwise Privalign "
            "GenRM rows. The leak term sums two response errors and ranges [-2, 0]. "
            "Ignored when ranking_demo has no gold_leak_response1/2 targets."
        ),
    )
    parser.add_argument(
        "--policy_trained_genrm_path",
        type=str,
        default=None,
        help=(
            "Path to a trained generative reward model checkpoint (Phase A output). "
            "Required when --policy_scorer=privalign_trained_genrm. The checkpoint is loaded "
            "into a temporary vLLM engine each step and asked to emit `Score: <int -2..2>` "
            "for every policy rollout."
        ),
    )
    parser.add_argument(
        "--policy_trained_genrm_max_new_tokens",
        type=int,
        default=1024,
        help=(
            "Max new tokens for the trained gen-RM at policy-RL time. Default 1024 gives "
            "the gen-RM room for a thinking trace plus the final `Score: <int>` line."
        ),
    )
    parser.add_argument(
        "--policy_trained_genrm_enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether the trained gen-RM runs with Qwen thinking on at policy-RL time. "
            "Default True to match Phase A training. Pass --no-policy_trained_genrm_enable_thinking "
            "to disable thinking and run with a tighter token budget."
        ),
    )
    parser.add_argument(
        "--policy-trained-genrm-prompt-template",
        dest="policy_trained_genrm_prompt_template",
        type=str,
        default=None,
        help=(
            "Name of the rate-this prompt template the trained gen-RM uses at policy-RL time. "
            "Defaults to 'privalign_genrm_pairwise' for --policy_scorer=privalign_trained_genrm. Override "
            "to point at a custom template under prompts/<name>.txt."
        ),
    )
    parser.add_argument(
        "--policy-trained-genrm-annotation-conditioning",
        dest="policy_trained_genrm_annotation_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Condition the trained gen-RM on the dataset's reference responses and per-annotator "
            "leak/omit labels. When set, the default prompt template switches to "
            "'privalign_rl_pairwise_judge' (still overridable via "
            "--policy-trained-genrm-prompt-template) and the rate-this content is rendered with "
            "{reference_response_a}, {reference_response_b}, and {annotator_block} populated from "
            "example.judge_demo. Only meaningful for Privalign policy scorers."
        ),
    )
    parser.add_argument(
        "--policy_reward_clip_min",
        type=float,
        default=-1000.0,
        help="Minimum clip applied to raw sequence-level policy scores before baseline centering.",
    )
    parser.add_argument(
        "--policy_reward_clip_max",
        type=float,
        default=1000.0,
        help="Maximum clip applied to raw sequence-level policy scores before baseline centering.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="constant_with_warmup",
        help="LR scheduler type (e.g. constant_with_warmup, cosine, linear)",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Per-GPU training micro-batch size measured in generated sequences, not prompts.",
    )
    parser.add_argument(
        "--global_prompt_batch_size",
        type=int,
        default=32,
        help=(
            "Global prompt batch per optimizer step across all GPUs. "
            "With num_generations > 1, this expands to global_prompt_batch_size * num_generations "
            "generated sequences."
        ),
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=16384,
        help="Maximum prompt length; longer training examples are filtered out instead of truncated.",
    )
    parser.add_argument("--max_completion_length", type=int, default=16384, help="Maximum completion length")
    parser.add_argument(
        "--max_train_completion_length",
        type=int,
        default=None,
        help="Optional training-time cap for completion tokens after rollout generation.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=2,
        help="Number of completions to sample per prompt.",
    )
    parser.add_argument(
        "--train-vllm-temperature",
        dest="train_vllm_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for training vLLM rollout generation.",
    )
    parser.add_argument(
        "--eval-judge-vllm-temperature",
        dest="eval_judge_vllm_temperature",
        type=float,
        default=0.6,
        help=(
            "Sampling temperature for the LLM-judge scoring model. Also the fallback "
            "for eval-candidate generation when --eval-reference-vllm-temperature is unset."
        ),
    )
    parser.add_argument(
        "--eval-reference-vllm-temperature",
        dest="eval_reference_vllm_temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature for eval-candidate generation (student and reference responses). "
            "When unset, falls back to --eval-judge-vllm-temperature."
        ),
    )
    parser.add_argument(
        "--eval-reference-vllm-top-p",
        dest="eval_reference_vllm_top_p",
        type=float,
        default=0.95,
        help=(
            "Top-p sampling for eval-candidate generation (student and reference responses)."
        ),
    )
    parser.add_argument(
        "--eval-reference-vllm-top-k",
        dest="eval_reference_vllm_top_k",
        type=int,
        default=20,
        help=(
            "Top-k sampling for eval-candidate generation (student and reference responses)."
        ),
    )
    parser.add_argument(
        "--eval-reference-vllm-repetition-penalty",
        dest="eval_reference_vllm_repetition_penalty",
        type=float,
        default=None,
        help=(
            "Repetition penalty for eval-candidate generation (student and reference responses). "
            "When unset, no repetition penalty is applied."
        ),
    )
    parser.add_argument(
        "--eval-reference-vllm-presence-penalty",
        dest="eval_reference_vllm_presence_penalty",
        type=float,
        default=None,
        help=(
            "Presence penalty for eval-candidate generation (student and reference responses). "
            "When unset, no presence penalty is applied."
        ),
    )
    parser.add_argument(
        "--dev-eval-reference-response-cache",
        dest="dev_eval_reference_response_cache",
        type=str,
        default=None,
        help=(
            "Optional JSONL cache for frozen-reference dev-eval responses. "
            "If the file exists with matching eval examples/generations, those responses are reused; "
            "otherwise they are generated once and written to the path."
        ),
    )
    parser.add_argument(
        "--dev-eval-reference-response-cache-readonly",
        dest="dev_eval_reference_response_cache_readonly",
        action="store_true",
        default=False,
        help=(
            "Require --dev-eval-reference-response-cache to already exist with all expected "
            "rows. Missing or incomplete caches raise instead of regenerating references."
        ),
    )
    parser.add_argument(
        "--judge-vllm-top-p",
        dest="judge_vllm_top_p",
        type=float,
        default=1.0,
        help=(
            "Top-p sampling used for LLM-judge generation paths."
        ),
    )
    parser.add_argument(
        "--judge-vllm-top-k",
        dest="judge_vllm_top_k",
        type=int,
        default=None,
        help=(
            "Top-k sampling used for LLM-judge generation paths."
        ),
    )
    parser.add_argument(
        "--judge-vllm-presence-penalty",
        dest="judge_vllm_presence_penalty",
        type=float,
        default=None,
        help=(
            "Presence penalty used for LLM-judge generation paths. Disabled by default."
        ),
    )
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling for rollout generation.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling for rollout generation.")
    parser.add_argument("--min_p", type=float, default=None, help="Min-p sampling for rollout generation.")
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=None,
        help="Presence penalty applied during rollout generation.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty applied during rollout generation.",
    )
    parser.add_argument(
        "--generation_kwargs",
        type=parse_generation_kwargs,
        default=None,
        help="Extra rollout generation kwargs as a JSON object.",
    )
    parser.add_argument(
        "--lm_head_chunk_size",
        type=int,
        default=128,
        help=(
            "Number of completion positions to project through lm_head at once for full-vocabulary "
            "reference KL. Reduces peak VRAM by chunking token-wise vocab logits."
        ),
    )
    parser.add_argument(
        "--ref_model_offload",
        action="store_true",
        help="Keep the frozen reference model on CPU to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory, checkpoint name under output_dir, or 'latest'.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Model name")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Allow custom model/tokenizer/config code from the model repo for both HF and vLLM loads.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="Model/vLLM dtype override (bfloat16 by default; set float32 to disable bf16).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help=(
            "Attention implementation override "
            "(e.g. flash_attention_2, flash_attention_3, eager, "
            "kernels-community/vllm-flash-attn3)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every N optimizer steps")
    parser.add_argument(
        "--sample-log-steps",
        type=int,
        default=1,
        help=(
            "Log training sample texts (for example prompt and student_response) "
            "every N optimizer steps; set to 0 to disable."
        ),
    )
    parser.add_argument("--save_steps", type=int, default=100, help="Save every N optimizer steps")
    parser.add_argument(
        "--dev-eval-steps",
        type=int,
        default=0,
        help="Run preference dev validation evaluation every N optimizer steps; 0 disables it.",
    )
    parser.add_argument(
        "--dev-eval-before-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run preference dev validation LLM-judge evaluation once before the first optimizer step; "
            "use --no-dev-eval-before-training to disable."
        ),
    )
    parser.add_argument(
        "--dev-eval-max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of preference dev validation prompts used per dev eval.",
    )
    parser.add_argument(
        "--dev-eval-num-generations",
        type=int,
        default=1,
        help="Number of candidate responses to generate per prompt during dev eval (reduces noise by averaging).",
    )
    parser.add_argument(
        "--dev-eval-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute objective loss on dev eval data when supported.",
    )
    parser.add_argument(
        "--save-best-metric",
        dest="save_best_metric",
        type=str,
        default=None,
        help=(
            "When set, after each dev eval also write/overwrite a 'best' checkpoint "
            "if this metric improved over the prior best (e.g. 'privalign/student_mean_score'). "
            "When unset, only --save_steps checkpoints are kept."
        ),
    )
    parser.add_argument(
        "--save-best-higher-is-better",
        dest="save_best_higher_is_better",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Direction for --save-best-metric (default: higher is better).",
    )
    parser.add_argument("--report_to", type=str, default="wandb", help="Comma-separated logger targets or 'none'")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing; use --no-gradient_checkpointing to disable.",
    )
    parser.add_argument(
        "--deepcompile",
        action="store_true",
        default=False,
        help="Enable DeepSpeed engine.compile() when supported.",
    )
    parser.add_argument(
        "--log_phase_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable detailed phase-progress logging; use --no-log_phase_progress to disable.",
    )
    parser.add_argument(
        "--debug-distributed-phase-barriers",
        dest="debug_distributed_phase_barriers",
        action="store_true",
        default=False,
        help=(
            "Add distributed barriers around backward-phase debug logs. "
            "Useful for diagnosing rank skew or hangs, but expensive in normal training."
        ),
    )
    parser.add_argument(
        "--disable_dropout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable dropout in student and reference models; use --no-disable_dropout to keep it on.",
    )

    parser.add_argument(
        "--deepspeed",
        type=str,
        default="configs/deepspeed/zero3.json",
        help="DeepSpeed config JSON path.",
    )
    parser.add_argument(
        "--deepspeed_offload_optimizer",
        action="store_true",
        default=None,
        help="Override the runtime DeepSpeed config to offload student optimizer states to CPU.",
    )
    parser.add_argument(
        "--deepspeed_offload_param",
        action="store_true",
        default=None,
        help="Override the runtime DeepSpeed config to offload student parameters to CPU.",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for colocated vLLM rollout generation.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.3,
        help="Fraction of GPU memory reserved for colocated vLLM rollout generation.",
    )
    parser.add_argument(
        "--vllm_enable_sleep_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable vLLM sleep mode; use --no-vllm_enable_sleep_mode to disable.",
    )
    parser.add_argument(
        "--vllm_enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable vLLM CUDA graph capture/torch compile; defaults to vLLM's "
            "optimized graph path."
        ),
    )
    parser.add_argument(
        "--vllm_show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show vLLM generation progress; use --no-vllm_show_progress to hide it.",
    )
    parser.add_argument(
        "--qwen_enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Qwen thinking-mode chat formatting; use --no-qwen_enable_thinking to disable.",
    )
    parser.add_argument(
        "--disable_student_thinking",
        action="store_true",
        default=False,
        help=(
            "Disable student-only thinking-mode chat formatting and response parsing while keeping "
            "student thinking otherwise controlled by --qwen_enable_thinking."
        ),
    )
    return parser


def build_model_kwargs(
    attn_implementation: str | None = None,
    *,
    dtype: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if dtype == "bfloat16":
        kwargs["dtype"] = torch.bfloat16
    elif dtype == "float32":
        kwargs["dtype"] = torch.float32
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return kwargs


def initialize_distributed_runtime(world_size: int, local_rank: int = -1) -> None:
    if world_size <= 1 or torch.distributed.is_initialized():
        return

    if "LOCAL_RANK" in os.environ:
        resolved_local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:
        resolved_local_rank = int(os.environ["SLURM_LOCALID"])
    elif local_rank >= 0:
        resolved_local_rank = local_rank
    else:
        resolved_local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(resolved_local_rank)

    import deepspeed

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    timeout_seconds = os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SECONDS")
    if timeout_seconds is None:
        deepspeed.init_distributed(dist_backend=backend)
        return

    deepspeed.init_distributed(
        dist_backend=backend,
        timeout=timedelta(seconds=int(timeout_seconds)),
    )

_CLI_ONLY_ARGS = frozenset({
    "model_name",
    "dataset",
    "dataset_name",
    "dataset_config",
    "deduplicate_preference_examples",
    "train_max_samples",
})


def build_training_config(
    args,
    *,
    gradient_accumulation_steps: int,
):
    import dataclasses
    from training.config import TrainingConfig

    config_fields = {f.name for f in dataclasses.fields(TrainingConfig)}
    kwargs: dict[str, Any] = {}
    for arg_name, value in vars(args).items():
        if arg_name in _CLI_ONLY_ARGS:
            continue
        if arg_name in config_fields:
            kwargs[arg_name] = value

    kwargs["deepspeed"] = resolve_deepspeed_config(args.deepspeed)
    kwargs["report_to"] = normalize_report_to(args.report_to)
    kwargs["gradient_accumulation_steps"] = gradient_accumulation_steps
    kwargs["policy_judge_model_name"] = args.policy_judge_model_name or args.model_name
    kwargs["dev_eval_judge_model_name"] = args.dev_eval_judge_model_name or args.model_name

    return TrainingConfig(**kwargs)
