set -e
if [ ! -d venv ]; then
    python3 -m venv venv
    ./venv/bin/pip install torch numpy pybind11 psutil --quiet
fi
mkdir -p build && cd build
cmake -Dpybind11_DIR=$(../venv/bin/python -m pybind11 --cmakedir) .. && make -j$(nproc)
cd ..
PYTHONPATH=build ./venv/bin/python train.py
