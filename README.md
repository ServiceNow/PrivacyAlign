# PrivacyAlign: Contextual Privacy Alignment for LLM Agents

[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2606.21710)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-PrivacyAlign-yellow)](https://huggingface.co/datasets/ServiceNow/PrivacyAlign)
[![Project Page](https://img.shields.io/badge/project-page-blue)](https://privacyalign.github.io/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

<p align="center">
  <img src="assets/figure1_pAA_github.png" alt="Contextual privacy leakage by an LLM agent" width="60%">
</p>

<p align="center"><em>We study contextual privacy leakage and the alignment of LLM agents with human privacy norms. Poorly aligned agents risk leaking private details (in this case, financial) drawn from tool calls and persistent memory.</em></p>

AI agents acting on behalf of users constantly decide what is appropriate to share, with whom, and under what conditions. **PrivacyAlign** places human judgment at the center of agentic privacy alignment: a dataset of detailed human privacy annotations over agent responses in diverse scenarios where contemporary LLMs actually leak, used to ground both alignment training and evaluation in human privacy norms.

We propose **annotation-conditioned reward modeling**, which conditions reward judgments on same-prompt human annotations in context rather than an automated proxy. This makes automated evaluation far more reliable, and small open-weight agents trained with RL against this reward better protect sensitive information.

## Repository structure

We include code to

- **generate** scenarios
- **evaluate** how models handle scenarios given human annotations
- **align** a policy against human privacy norms

```
synthetic_data_generation/   ──►   ServiceNow/PrivacyAlign   ──►   privacy_evaluation/
   (build scenarios)              (HF dataset: 1,150 train /        (benchmark frontier
                                   200 test, human-annotated)        & trained models)
                                          │
                                          ▼
                          online_preference_alignment/
                       (align a policy to the human norms)
```

| Component | Key contents | What it does |
|-----------|--------------|--------------|
| [`synthetic_data_generation/`](synthetic_data_generation/README.md) | `generate_data.py`, `pipeline/` (profile, seed/vignette, trajectory, quality/leakage/diversity filters), `resources/prompts/`, `mine_samples.py`, `postprocess.py` | Staged generator for privacy-sensitive agent scenarios: **profile → scenario (seed + vignette) → trajectory + memories → quality / leakage / diversity filtering**. Outputs the candidate scenarios that become the dataset. |
| [`privacy_evaluation/`](privacy_evaluation/README.md) | `generate_responses.py`, `judge_responses.py`, `resources/prompts/` | Runs the benchmark: generate agent responses, then **judge each for leaks / omissions** with annotation-conditioned LLM judges. |
| [`online_preference_alignment/`](online_preference_alignment/README.md) | `ray_backend/`, `training/`, `objectives/` (SAPO), `judges/`, `genrm/`, `experience/`, `data_loaders/` | RL alignment stack: **vLLM rollouts → annotation-conditioned reward → group-relative SAPO policy optimization** with optional full-vocab reference-KL, orchestrated with Ray. |

Each component has its own README with full details.

## Dataset

PrivacyAlign is a human-annotated preference dataset for privacy-aligned
tool-use agents (1,150 train / 200 test scenario pairs). All scenarios are
**synthetic** — user names, emails, memories, and tool trajectories are
fictional personas, so no real user data is included. It is available on
Hugging Face:

```python
from datasets import load_dataset

ds = load_dataset("ServiceNow/PrivacyAlign")
print(ds["train"][0]["user_instruction"])
```

The evaluation and training code in this repo download the dataset from the Hub automatically (cached locally under `.cache/`); pass an explicit `--dataset-path` to use a local copy instead.

## Evaluation

```bash
# Evaluate models on the benchmark
export OPENROUTER_API_KEY=...
python privacy_evaluation/generate_responses.py --models <model-id>
python privacy_evaluation/judge_responses.py --help
```

We also provide the full RL alignment stack used to train policies against the annotation-conditioned reward, in [`online_preference_alignment/`](online_preference_alignment/README.md).

## Third-party dependencies

The third-party Python packages used by PrivacyAlign are listed in [`requirements.txt`](requirements.txt). These packages are installed separately and remain subject to their respective licenses and terms.

## Citation

```bibtex
@article{tamber2026privacyalign,
  title         = {PrivacyAlign: Contextual Privacy Alignment for LLM Agents},
  author        = {Manveer Singh Tamber and Abhay Puri and Marc-Etienne Brunet and
                   Perouz Taslakian and Jimmy Lin and Spandana Gella},
  year          = {2026},
  eprint        = {2606.21710},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.21710}
}
```


## License

Apache-2.0 — see [LICENSE](LICENSE).
