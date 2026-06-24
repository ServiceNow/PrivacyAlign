# Data Generation Pipeline

A staged generator that produces private-sensitive samples for evaluating the privacy-awareness of LLM agents.

## Structure

- `generate_data.py` — main entry point for data generation
- `mine_samples.py` — mine pre-generated samples for privacy-handling differences across models
- `prune_mined_pairs.py` — postprocess-style pruning for mined pairs, with optional unanimous-sensibility filtering
- `filter_embedding_diversity.py` — standalone embedding diversity filter
- `filter_link_passthrough.py` — standalone forwarded-link leakage filter
- `postprocess.py` — post-processing (link filter, name dedup, diversity sampling, reporting)
- `combine_filter_postprocess.py` — combine + embedding filter + postprocess in one command
- `model_client.py` — model loading (vLLM offline)
- `pipeline/` — modular stage implementations (profile, seed/vignette, trajectory + memories, quality/leakage filters, diversity, identifiers, toolkit registry, output assembly, parsing helpers)

## Resource Files

- `resources/seed_options.json` — seed/profile option pools (sex, ethnicity, religion, citizenship, etc.)
- `resources/names/name_frequency.jsonl` — weighted first-name list for profile sampling
- `resources/assets/all_toolkits.json` — toolkit specs for parameter schema validation
- `resources/prompts/` — prompt templates for each generation and judge stage

## What it generates

Each output sample contains `name`, `seed` (with contextual-integrity fields such as `data_subject`, `data_sender`, `data_recipient`), `vignette`, `trajectory` (`user_instruction`, `toolkits`, `executable_trajectory`, `final_action`), and `generation_metadata` (profile attributes, `domains`, toolkit signature).

## Stages

1. **Profile generation** — sampled first name with constrained options for sex, ethnicity, religion. Overgeneration selects the most diverse candidate.
2. **Scenario generation** — combined seed + vignette in one pass: CI parameters, toolkits, final action, story, user instruction, sensitive info items. Overgeneration selects the most diverse candidate.
3. **Vignette quality check** — LLM judge filters bad scenarios before trajectory generation.
4. **Trajectory + memories generation** — executable trajectory and memory items grounded in the vignette.
5. **Quality filter** — optional LLM judge to reject low-quality samples.
6. **Leakage filter** — naive agent action, sensibility check, then zero-shot leakage judgment. Ensures the scenario actually triggers a privacy leak.
7. **Link passthrough filter** — rejects samples whose leakage is just forwarding a pre-existing URL (document link, wiki page, report URL). Detects URL overlap plus document-reference signals.

## Quick start

From `PrivacyAlign/synthetic_data_generation`:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4

python generate_data.py \
  --backend vllm_offline \
  --generator-model openai/gpt-oss-120b \
  --num-samples 1000000 \
  --output-path outputs/main_data_generated_gpt_oss_120b.json \
  --error-path outputs/generation_errors_gpt_oss_120b.jsonl \
  --include-metadata \
  --batch-size 256 \
  --vllm-tensor-parallel-size 8 \
  --vllm-gpu-memory-utilization 0.9 \
  --reasoning-effort high \
  --seed 1
```

Model-specific notes for the other generators used to build the dataset:

- **NVIDIA Nemotron** (dense): add `--vllm-language-model-only --filter-top-p 0.95`.
- **Qwen** (MoE): add `--vllm-language-model-only --presence-penalty 0.5 --filter-top-p 0.8 --filter-top-k 20 --no-enable-thinking`.

Use `--resume` with a fresh `--seed` to add more samples to an existing output file (re-running the same model with new seeds is how the larger datasets were built). For an 8-GPU node, add `--vllm-enable-expert-parallel` for MoE models (GPT-OSS, Qwen3.5).

`--batch-size` (default `8`) sets how many samples share batched LLM calls at every stage. Larger batches give better GPU utilization with vLLM offline; set `--batch-size 1` for sequential mode. Diversity counters update between batches but are stale within a batch, an acceptable soft constraint.

Run `python generate_data.py --help` for the full option list. The most relevant knobs:

| Flag | Default | Purpose |
|------|---------|---------|
| `--generator-model` | `openai/gpt-oss-120b` | model to generate with |
| `--num-samples` | `20` | new samples to generate |
| `--batch-size` | `8` | samples per batched LLM call |
| `--overgeneration-factor` | `4` | candidates per sample in profile/scenario stages (most diverse kept) |
| `--reasoning-effort` | `high` | Harmony reasoning effort for GPT-OSS stages |
| `--vllm-tensor-parallel-size` | `1` | GPU sharding |
| `--vllm-language-model-only` | off | skip multimodal weights to free KV cache (Llama-4, Qwen-3.5, Mistral-3) |
| `--sensitive-memory-prob` | `0.5` | chance a sample's memories contain sensitive info |
| `--disable-quality-filter` / `--disable-leakage-filter` / `--disable-link-passthrough-filter` | off | skip a filter stage |

## Post-generation filters

Recommended workflow after `generate_data.py`: combine the generated JSON files, run embedding diversity filtering, then run `postprocess.py`. `combine_filter_postprocess.py` runs all three end to end.

```bash
# 1) Combine discovered outputs.
python3 combine_filter_postprocess.py --stop-after-merge

# 2) Embedding diversity filtering.
python3 filter_embedding_diversity.py \
  --input-path outputs/combined.json \
  --output-path outputs/combined_diversity_filtered.json \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --embedding-device cuda \
  --max-similarity 0.95

# 3) Postprocessing.
python3 postprocess.py \
  --input-path outputs/combined_diversity_filtered.json \
  --output-path outputs/combined_postprocessed.json \
  --name-threshold 2.0 \
  --max-pct 25.0 --toolkit-max-pct 25.0 --domain-signature-max-pct 5.0 \
  --min-model-family-pct 29.0 \
  --self-scope-pct 29.0 --third-party-scope-pct 29.0 --multi-subject-scope-pct 29.0 \
  --exact-time-limit 1200
```

**Embedding diversity filter** (`filter_embedding_diversity.py`) removes near-semantic duplicates via cosine similarity using a local HuggingFace embedding model (no API cost). A sample is rejected if its max similarity to any accepted sample exceeds `--max-similarity` (default `0.95`, lower = stricter).

**Link passthrough filter** (`filter_link_passthrough.py`) is the standalone version of the in-pipeline filter. It rejects samples whose leakage is forwarding a pre-existing URL from the reference pool.

**Postprocess** (`postprocess.py`) runs in sequence: domain normalization, a healthcare-mismatch filter, the link-passthrough filter, name deduplication (overrepresented character names swapped for ethnicity- and gender-matched alternatives, in prose, emails, and handles), diversity sampling (caps any single category of the balance fields below `--max-pct`% via a SciPy MILP, jointly enforcing minimum subject-scope and model-family shares), model-family balancing, and a before/after reporting pass. Run it directly to rerun postprocessing on a combined or filtered file. Key flags: `--max-pct`, `--toolkit-max-pct`, `--domain-signature-max-pct`, the `--*-scope-pct` minimum shares, `--min-model-family-pct` / `--equalize-model-families`, `--name-threshold`, and `--exact-time-limit`. See `--help` for the rest.

## Mining samples (`mine_samples.py`)

Generates fresh responses for the GPT/NVIDIA/Qwen trio on an existing dataset, runs sensibility and comparative judging with each judge model, and keeps at most one pair per sample. The comparative judge output also records per-response leak/omit labels.

- A response survives to pair selection only if a majority of judges mark it sensible.
- A pair survives only if at least one response has enough leak support and the comparative judges agree on which response is better.
- Selection prefers pairs where both responses were unanimously sensible, then judge diversity, then pair-family diversity.

```bash
python mine_samples.py \
  --input-path outputs/combined_postprocessed.json \
  --output-path outputs/mined_pairs.json \
  --backend vllm_offline \
  --response-models openai/gpt-oss-120b nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 Qwen/Qwen3.5-397B-A17B-FP8 \
  --judge-models openai/gpt-oss-120b nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 Qwen/Qwen3.5-397B-A17B-FP8 \
  --batch-size 512 \
  --vllm-tensor-parallel-size 8 \
  --reasoning-effort high
```

`--judge-models` defaults to `--response-models`. It shares the vLLM/backend flags with `generate_data.py`; see `--help` for the full list.

The output JSON has four top-level arrays: `analyzed_pairs` (every evaluated pair), `selected_pairs` (one kept pair per sample, a subset of `analyzed_pairs`), `samples` (reconstructed samples with response/judge metadata), and `invalid_samples`. Each analyzed pair carries aggregate stats plus parsed per-judge verdicts, and a `metadata` block summarizes the run (counts of inputs, responses, sensible responses, analyzed pairs, and selected pairs).

## Pruning mined pairs (`prune_mined_pairs.py`)

Takes the mined output, optionally keeps only pairs judged sensible by every judge model, then applies the exact `postprocess.py` diversity solver to the corresponding source samples. Use this when you want mined pairs to respect the same diversity constraints as postprocess.

```bash
python3 prune_mined_pairs.py \
  --input-path outputs/mined_pairs.json \
  --output-path outputs/mined_pairs_pruned.json \
  --require-unanimous-sensible \
  --max-pct 20.0 --toolkit-max-pct 25.0 --domain-signature-max-pct 5.0 \
  --min-model-family-pct 28.0 \
  --self-scope-pct 33.0 --third-party-scope-pct 27.0 --multi-subject-scope-pct 30.0 \
  --exact-time-limit 1200
```

Same diversity constraints as the postprocess command above (minus `--name-threshold`, since this prunes pairs without rewriting sample text). `--require-unanimous-sensible` demands `sensible=True` from every judge for both responses, and `--target-count` returns the largest feasible subset up to that count. `--source-input-path` points to the original dataset if it cannot be resolved from `metadata.input_path`.

## Notes

- For offline vLLM, install `vllm` (and for GPT-OSS, also `openai-harmony`).
