#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

venv/bin/python -m pip install -q --upgrade pip numpy pybind11 torch

cmake -S cpp -B build -Dpybind11_DIR="$(venv/bin/python -m pybind11 --cmakedir)"
cmake --build build -j"$(nproc)"

PYTHONPATH="${PWD}:build" exec venv/bin/python -m hybridchess.trainer
