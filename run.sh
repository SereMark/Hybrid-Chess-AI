#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

venv/bin/python -m pip install -q --upgrade pip wheel
venv/bin/python -m pip install -q numpy pybind11 torch

cmake -S cpp -B build -Dpybind11_DIR="$(venv/bin/python -m pybind11 --cmakedir)" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

export PYTHONPATH="${PWD}:${PWD}/build"
exec venv/bin/python -m hybridchess.trainer
