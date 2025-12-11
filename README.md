# Hybrid Chess AI

AlphaZero-szerű, megerősítéses tanuláson (RL) alapuló sakkmotor C++ maggal (MCTS + sakkmotor, `pybind11` kötés) és PyTorch policy/value hálóval. Úgy lett tervezve, hogy egy RTX 3070-es laptopon is végigfusson a teljes end-to-end tanítási és értékelési folyamat.

---

## Követelmények / Tesztelt környezet

- Windows 11 x64
- Python 3.13
- Visual Studio 2022 Build Tools (Desktop C++)
- CMake ≥ 3.21
- NVIDIA driver + CUDA-kompatibilis GPU (RTX 30xx ajánlott)

---

## Gyors indítás

1. **Virtuális környezet és csomagok**:
   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   # PyTorch: ellenőrizd a saját beállításodnak megfelelő parancsot a https://pytorch.org/get-started/locally/ oldalon
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```

2. **C++ mag fordítása**:
   ```powershell
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
   cmake --build build --config Release
   ```

3. **Alap tanítási futás**:
   ```powershell
   $env:PYTHONPATH = "$PWD\src\python"
   python -m main -c configs\run1_baseline.yaml
   ```

A futások eredményei és checkpointjai a `runs/` mappában jelennek meg.

---

## Bővebb használat

- **Tanítás konfigurációval + ráfedéssel**:
  ```powershell
  $env:PYTHONPATH = "$PWD\src\python"
  python -m main -c configs\run1_baseline.yaml -o configs\my_overrides.yaml
  ```

---

## Mappák röviden

- `src/core/` - C++ sakkmotor, MCTS és `pybind11` kötés.
- `src/python/` - tanítási pipeline (önjátszás, replay buffer, háló, trainer, fő CLI).
- `configs/` - YAML konfigurációk különböző futásokhoz.
- `runs/` - automatikusan létrehozott futási könyvtárak (checkpointok, logok).
- `tools/` - benchmark és artifact generáló scriptek.

---

## Tesztelés és minőség

```powershell
$env:PYTHONPATH = "$PWD\src\python"
python -m pytest
python -m black src/python tests
python -m ruff check src/python tests
python -m mypy src/python
```