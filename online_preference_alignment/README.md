# Online Preference Alignment

Online RL stack for contextual privacy alignment of tool-using LLM agents, part of the [PrivacyAlign](../) research release (ServiceNow, Apache-2.0). It takes a base policy (e.g. `Qwen/Qwen3-4B`), samples agent responses on the [`ServiceNow/PrivacyAlign`](https://huggingface.co/datasets/ServiceNow/PrivacyAlign) dataset with vLLM, scores them with an annotation-conditioned reward source, and updates the policy with a group-relative policy-optimization objective (SAPO). Everything runs as a synchronous, colocated Ray pipeline. The goal is a policy whose final tool calls act on the user's request without leaking private context or omitting necessary detail.

## RL loop

Each optimizer step runs the full loop.

1. Rollout. A batch of prompts is sampled `num_generations` times per prompt with colocated vLLM engines (sleep-mode, weight-synced to the live policy).
2. Score. Every completion gets a sequence-level reward from the configured `policy_scorer` (annotation-conditioned LLM judge, trained gen-RM, or pairwise-margin reward used to train such a gen-RM).
3. Advantage. Rewards are clipped and centered within each prompt's generation group (group-relative baseline), so `num_generations > 1` is required.
4. Update. The policy is optimized with SAPO (a soft-clipped, asymmetric-temperature surrogate) using Dr. GRPO length normalization (token losses divided by the global action-token count, not per-sequence length), with an optional full-vocabulary reverse-KL term `KL(policy ‖ reference)` anchoring to the frozen base model.

The objective is computed against vLLM-generated `old_log_probs`, so the update is importance-weighted rather than strictly on-policy. Rollout/training overlap is disabled, so the driver only sees fresh synchronous rollouts. Training workers and vLLM engines are colocated in one Ray placement group and exchange weights over NCCL (or CUDA IPC when colocated).

### Data flow

```
ray_backend.coordinator.main()            (driver: CLI, dataset, Ray bootstrap, outer loop, logging)
        │
        │  load ServiceNow/PrivacyAlign → render naive_agent prompts → filter by max_prompt_length
        ▼
  for each prompt batch:
        │
        ├─► rollout_dispatch.build_training_payload
        │       ├─ vLLM engines (vllm_engine.py)  ──► sampled completions + per-token logprobs
        │       └─ experience/ pipeline:
        │              scorers.py   → sequence rewards (judge / gen-RM / pairwise-margin)
        │              builders.py  → group-baseline advantages → PolicyOptimizationBatch
        │
        ├─► rollout_dispatch.run_training_iteration
        │       └─ RayActorGroup of TrainingModelActor (DeepSpeed ZeRO-3)
        │              training/trainer.py → objectives/policy_optimization.py (SAPO + optional ref-KL)
        │
        └─► dev_eval.run_dev_eval  (every --dev-eval-steps)  → checkpoints (latest / best / final)
```

## Module layout

| Path | Responsibility |
|------|----------------|
| `main.py`, `cli.py` | Entry point and argument surface (`build_arg_parser`, `TrainingConfig` construction, distributed-runtime init). |
| `ray_backend/` | Ray training pipeline: `coordinator.py` (CLI/Ray bootstrap, outer loop), `rollout_dispatch.py` (prompts to vLLM, training iterations), `dev_eval.py` (dev-eval, reference caching), `vllm_*` (colocated engines, weight sync), `training_worker.py` (per-GPU actor), plus launcher/helpers. |
| `objectives/` | Pluggable objectives: SAPO (active), `vanilla` importance-weighted policy-gradient baseline, Dr. GRPO normalization, optional reference-KL. |
| `training/` | Worker-side trainer mixins (DeepSpeed runtime, prompt shaping, forward passes, loss/metrics, `full_vocab_kl.py` chunked reverse-KL kernel, checkpointing, `config.py` validation). |
| `judges/` | Prompt rendering, output parsing, and dual-order combination for annotation-conditioned pairwise and pointwise (leak/omit) LLM judges. |
| `experience/` | Coordinator-side conversion of trajectories into policy batches: `scorers.py` (reward scorers), `builders.py` (rewards to group-baseline advantages), `pipeline.py` (wiring). |
| `genrm/` | `build_privalign_genrm_training_data.py` builds gen-RM (Phase A) rows from PrivacyAlign (both orderings, signed `target_score` in `[-2, +2]`, optional soft leak labels). |
| `data_loaders/` | `preference.py` loads/renders PrivacyAlign (student-agent prompts, judge demos, deterministic train/val split, in-memory gen-RM rows, Hub download/fallback). |
| `sft/`, `prompts/`, `utils/`, `configs/deepspeed/`, `scripts/` | SFT utilities, text templates, helper utilities, ZeRO-3 configs, and SFT launch scripts. |

## Quickstart

Launch through the Ray coordinator (`main.py` forwards to it). This is the RL setup used to train the released models, with the annotation-conditioned pairwise judge.

```bash
python -m ray_backend.coordinator \
  --model_name Qwen/Qwen3-4B \
  --dataset preference --dataset_name privalign-dataset \
  --training_objective policy_optimization \
  --policy_scorer privalign_rl_pairwise_judge \
  --num_generations 4 \
  --global_prompt_batch_size 64 \
  --learning_rate 1e-6 \
  --lr_scheduler_type constant_with_warmup --warmup_steps 10 \
  --weight_decay 0.01 \
  --max_prompt_length 16384 --max_completion_length 16384 \
  --train-vllm-temperature 1.0 \
  --policy_sapo_tau_pos 1.0 --policy_sapo_tau_neg 1.05 \
  --policy_reward_kl_coef 0.002 --lm_head_chunk_size 1024 \
  --policy_undershort_penalty_max 4.0 --policy_undershort_floor_ratio 0.5 \
  --deepspeed configs/deepspeed/zero3.json --deepspeed_offload_optimizer \
  --dev-eval-num-generations 4 \
  --dev-eval-judge-model-name google/gemma-4-31b-it \
  --save-best-metric privalign/clean_rate --save-best-higher-is-better \
  --ray_num_train_workers 8 --vllm_tensor_parallel_size 8 \
  --output_dir outputs/rl-privalign
```

`--dataset_name privalign-dataset` resolves to `ServiceNow/PrivacyAlign` on the Hub by default (a local export path is accepted as fallback). The split is downloaded once and cached as JSONL under `~/.cache/privacyalign` (override with `PRIVALIGN_CACHE_DIR`). The dev split is a deterministic ~10% prompt-level holdout of `train`.

Standalone gen-RM data builder (Phase A of the trained-gen-RM baseline, writes JSONL instead of the in-memory path):

```bash
python -m genrm.build_privalign_genrm_training_data --output_dir outputs/genrm-privalign/training
```

## Training hyperparameters

| Hyperparameter | Value | Flag |
|----------------|-------|------|
| Base policy | Qwen3-4B, Qwen3-8B, Nemotron-3-Nano-4B | `--model_name` |
| Learning rate | sweep `{1e-6, 2e-6}` | `--learning_rate` |
| LR schedule | constant, 10-step warmup | `--lr_scheduler_type constant_with_warmup --warmup_steps 10` |
| Weight decay | 0.01 (AdamW) | `--weight_decay` |
| Precision | bf16 | `--dtype bfloat16` (default) |
| Distributed | DeepSpeed ZeRO-3, optimizer offload | `--deepspeed configs/deepspeed/zero3.json --deepspeed_offload_optimizer` |
| Rollout group size *K* | 4 | `--num_generations` |
| Global prompt batch size | 64 | `--global_prompt_batch_size` |
| Max prompt / completion length | 16,384 / 16,384 | `--max_prompt_length` / `--max_completion_length` |
| Rollout sampling temperature | 1.0 | `--train-vllm-temperature` |
| SAPO gating τ⁺ / τ⁻ | 1.0 / 1.05 | `--policy_sapo_tau_pos` / `--policy_sapo_tau_neg` |
| KL coefficient β | 0.002 | `--policy_reward_kl_coef` |
| Short-response penalty (max) | 4.0 | `--policy_undershort_penalty_max` |
| Short-response floor | 0.5× reference avg. words | `--policy_undershort_floor_ratio` |
| Dev split | deterministic ~10% prompt-level holdout | (automatic) |
| Dev generations per prompt | 4 | `--dev-eval-num-generations` |
| Dev eval judge | Gemma-4-31B-it | `--dev-eval-judge-model-name` |
| Checkpoint selection metric | dev clean rate | `--save-best-metric privalign/clean_rate --save-best-higher-is-better` |

Clean rate is the fraction of policy rollouts the annotation-conditioned leak/omit judge labels as neither leaking sensitive information nor omitting relevant non-sensitive information (`leaks=0` and `omits=0`). The dev-eval judge is a stronger, out-of-family model than the in-training reward source, which keeps checkpoint selection less noisy than the training reward.

## Reward source and objective options

| Flag | Choices / note | Meaning |
|------|----------------|---------|
| `--training_objective` | `policy_optimization` | Only policy RL is supported. |
| `--policy_algorithm` | `sapo` (default), `vanilla` | `sapo` = soft-clipped asymmetric-temperature surrogate; `vanilla` = importance-weighted policy gradient (`-(ratio·advantage)`), a neutral baseline. |
| `--policy_scorer` | `privalign_rl_pairwise_judge`, `privalign_pairwise_margin`, `privalign_trained_genrm` | In-group pairwise LLM judge conditioned on per-annotator leak/omit labels; pairwise-margin reward that trains a gen-RM; or a trained gen-RM checkpoint. |
| `--num_generations` | `4` (must be > 1) | Completions sampled per prompt; forms the reward baseline group. |
| `--policy_reward_kl_coef` | `0.002` (`0` disables) | Coefficient on `KL(policy ‖ reference)`. `> 0` makes the worker own a frozen reference model. |
| `--lm_head_chunk_size` | e.g. `1024` | Completion positions projected through `lm_head` per chunk for the full-vocab KL. Required whenever `--policy_reward_kl_coef > 0` (bounds lm-head pass memory). |
| `--policy_judge_scoring_mode` | `peer` (default), `anchor` | `peer` = pairwise within the generation group; `anchor` = vs a cached base anchor. |
| `--policy_judge_dual_order` | off by default | Score both response orderings to cancel position bias (doubles scorer calls). |
| `--policy_trained_genrm_path` | – | Required for `privalign_trained_genrm`; the gen-RM checkpoint loaded into a temporary vLLM engine each step. |
