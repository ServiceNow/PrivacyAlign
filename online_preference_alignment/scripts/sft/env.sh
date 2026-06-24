#!/usr/bin/env bash
# Source this file (do not exec) at the top of each SFT script.
# Override the conda env per-script via SFT_CONDA_ENV.
set -euo pipefail

# Optionally activate a conda environment if SFT_CONDA_ENV is set and conda is
# available. Otherwise the currently-active environment is used as-is.
SFT_CONDA_ENV="${SFT_CONDA_ENV:-}"
if [[ -n "${SFT_CONDA_ENV}" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${SFT_CONDA_ENV}"
fi

# Hugging Face cache location. Defaults to the standard user cache; override
# HF_HOME (e.g. to a shared network FS) before sourcing this file.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

# Make project modules importable from project root.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"

echo "[sft env] python=$(which python) HF_HOME=${HF_HOME}"
