# Hybrid Chess AI

AlphaZero-szerű, megerősítéses tanuláson (RL) alapuló sakkmotor C++ maggal (MCTS + sakkmotor, `pybind11` kötés) és PyTorch policy/value hálóval. Úgy lett tervezve, hogy egy RTX 3070-es laptopon is végigfusson a teljes end-to-end tanítási és értékelési folyamat.

---

## Követelmények

- Windows 11 x64
- Python 3.13
- Visual Studio 2022 Build Tools (Desktop C++)
- CMake ≥ 3.21
- NVIDIA driver + CUDA-kompatibilis GPU

---

## Fő Python csomagok

A `requirements.txt` a következő kulcsfontosságú csomagokat tartalmazza:

- `numpy` - numerikus számítások, állapot-tenzorok, replay buffer
- `torch` - neurális háló (policy/value), tanítás és inferencia
- `pybind11` - a C++ sakkmag és az MCTS Python-kötése
- `python-chess` - FEN/PGN kezelés és kiegészítő sakkeszközök
- `pyyaml` - YAML alapú konfigurációk
- `matplotlib`, `seaborn`, `pandas` - benchmarkok, statisztikák, grafikonok
- `pytest` - egység- és integrációs tesztek
- `black`, `ruff`, `mypy` - formázás, lintelés, statikus típusellenőrzés

---

## Telepítés

Virtuális környezet létrehozása és a Python csomagok telepítése (PowerShell):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

C++ mag fordítása CMake-kel (Release build):

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

---

## Tanítás

Ajánlott kiinduló konfiguráció RTX 3070-es laptophoz: `configs/run1_baseline.yaml`. További szcenáriók a `configs/` mappában találhatók.

```powershell
$env:PYTHONPATH = "$PWD\src\python"
python -m main -c configs\run1_baseline.yaml
```

További konfigurációs fájlok "ráfedése":

```powershell
python -m main -c configs\run1_baseline.yaml -o configs\my_overrides.yaml
```

A futások eredményei és checkpointjai a `runs/` mappában jelennek meg.

---

## Tesztelés és minőség

```powershell
python -m pytest
python -m black src/python tests
python -m ruff check src/python tests
python -m mypy src/python
```
