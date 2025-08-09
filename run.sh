#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=${VENV_DIR:-venv}
PY=${PYTHON:-python3}

if [ ! -d "$VENV_DIR" ]; then
  "$PY" -m venv "$VENV_DIR"
  "$VENV_DIR"/bin/pip install -q --upgrade pip wheel setuptools
fi

"$VENV_DIR"/bin/pip install -q numpy pybind11 torch

cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$("$VENV_DIR"/bin/python -m pybind11 --cmakedir)"
cmake --build build -j"$(nproc)"

export PYTHONPATH="$(pwd):build"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8

exec "$VENV_DIR"/bin/python -m hybridchess.cli
