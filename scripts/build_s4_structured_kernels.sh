#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/zza/code/S4M_MissingForecasting"
CONDA_ROOT="/mnt/data/zza/anaconda3"
ENV_NAME="s4m_missing_forecasting"
KERNEL_ROOT="$ROOT/baselines/S4M/s4m/model/S4/s4/extensions/kernels"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$KERNEL_ROOT"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6;8.9"
export MAX_JOBS="1"
python setup.py build_ext --inplace

sha256sum structured_kernels*.so
