#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "venv" ]; then python3 -m venv venv; fi
venv/bin/python -m pip install -q --upgrade pip wheel
venv/bin/python -m pip install -q numpy pybind11 psutil
venv/bin/python -m pip install -q --extra-index-url https://download.pytorch.org/whl/cu128 torch

cmake -S cpp -B build -Dpybind11_DIR="$(venv/bin/python -m pybind11 --cmakedir)" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

export PYTHONPATH="${PWD}:${PWD}/build"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec venv/bin/python -m hybridchess.trainer "$@"
