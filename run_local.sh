#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "venv" ]; then python3 -m venv venv; fi
venv/bin/python -m pip install -q --upgrade pip wheel
venv/bin/python -m pip install -q numpy pybind11 psutil
venv/bin/python -m pip install -q --index-url https://download.pytorch.org/whl/cu121 torch

cmake -S cpp -B build \
  -Dpybind11_DIR="$(venv/bin/python -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

export PYTHONPATH="${PWD}:${PWD}/build"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:true"

exec venv/bin/python -m hybridchess.trainer "$@"
