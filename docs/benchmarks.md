# Benchmarks

The benchmark scripts helped me answer practical questions while building the project: whether moving chess logic into C++ was worthwhile, how much batching helps inference, and where a training iteration actually spends its time.

Selected results are stored in `benchmark_reports/` so they can be inspected without repeating hours of self-play or training. They are snapshots from the original development machine, not promises about performance on every system.

## Saved reports

| File | What it measures |
| --- | --- |
| `chess_cpp.csv` | C++ move generation compared with `python-chess` |
| `mcts_scaling.csv` | MCTS throughput at different evaluation batch sizes |
| `inference_suite.csv` | Network latency and throughput across devices, dtypes, and batch sizes |
| `system_bench.csv` | End-to-end self-play and training-step throughput |
| `cost_breakdown.csv` | Time spent in self-play, training, and arena evaluation |
| `ranking_results.json` | Tournament-style results for saved models and simple baselines |

## Preparing the benchmark environment

Install the optional plotting/data packages if they are not already present:

```powershell
python -m pip install -e ".[dev,benchmarks]"
```

Build the native extension:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

$env:PYTHONPATH = "$PWD\src\python"
```

For GPU measurements, it is worth checking that CUDA is actually available before trusting the output:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Running benchmarks

### Chess core

```powershell
python tools\benchmarks.py chess --output-dir benchmark_reports
```

This compares move-generation work against `python-chess`. Results depend on the selected positions, compiler optimisation, and CPU, so the relative trend is more useful than a single headline number.

### MCTS scaling

```powershell
python tools\benchmarks.py mcts --scaling --output-dir benchmark_reports
```

This explores how search throughput changes with evaluation batch size. Very small batches cross the Python/PyTorch boundary too often; very large batches can spend too long waiting to fill.

### Neural-network inference

```powershell
python tools\benchmarks.py inference --output-csv benchmark_reports\inference_suite.csv
```

The inference suite compares batch sizes, devices, and supported dtypes. GPU timings should be compared only after warm-up, because the first calls include setup costs.

### End-to-end system timing

```powershell
python tools\benchmarks.py system --mode all --output-csv benchmark_reports\system_bench.csv
```

This measures larger pieces of the pipeline rather than isolated functions. It is more representative of real use, but also more affected by background processes and configuration.

### Cost breakdown and ranking

These commands require completed runs under `runs/`:

```powershell
python tools\benchmarks.py cost `
  --run runs\<run-id> `
  --output benchmark_reports\cost_breakdown.csv

python tools\benchmarks.py ranking `
  --runs-dir runs `
  --output-dir benchmark_reports
```

Ranking results are especially noisy when only a small number of games are played. Colour balance, opening selection, search settings, and random seeds should stay fixed when comparing models.

## Reading the numbers carefully

- Always record the CPU, GPU, Python, PyTorch, CUDA, and build configuration.
- Use a Release build of `chesscore`; Debug builds are not meaningful for performance comparisons.
- Run enough repetitions to reduce one-off timing noise.
- Keep the tested positions and command arguments fixed when comparing code changes.
- Do not compare GPU numbers from different precision modes as if they were the same experiment.
- Treat small arena score differences as uncertain unless they are supported by enough games.

The reports are most useful as engineering evidence: they show which part became faster or slower under a controlled setup. They are not intended as a broad claim that this project is faster than every alternative.
