import os
import sys
import time
import yaml
import importlib
import subprocess
from enum import Enum

class PipelineStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"

PIPELINES = {
    '1': {
        'name': 'data',
        'description': "Process PGN files",
        'class_name': 'DataPipeline',
        'requirements': ['chess', 'numpy', 'pandas', 'h5py', 'wandb']
    },
    '2': {
        'name': 'supervised',
        'description': "Train with supervision",
        'class_name': 'SupervisedPipeline',
        'requirements': ['chess', 'torch', 'numpy', 'pandas', 'wandb']
    },
    '3': {
        'name': 'reinforcement',
        'description': "Self-play learning",
        'class_name': 'ReinforcementPipeline',
        'requirements': ['chess', 'torch', 'numpy', 'wandb']
    },
    '4': {
        'name': 'eval',
        'description': "Evaluate models",
        'class_name': 'EvalPipeline',
        'requirements': ['chess', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'wandb', 'torch']
    }
}

def print_header(title):   print(f"\n=== {title} ===\n")
def print_info(msg):       print(f"INFO: {msg}")
def print_success(msg):    print(f"SUCCESS: {msg}")
def print_error(msg):      print(f"ERROR: {msg}")

def setup_environment():
    project_dir = '/content/drive/MyDrive/chess_ai'

    if not os.path.exists(project_dir):
        os.makedirs(os.path.join(project_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'evaluation'), exist_ok=True)

    os.chdir(project_dir)

    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    if 'src.utils.config' in sys.modules:
        from src.utils.config import Config
        Config._instance = None

    return project_dir


def _ensure_python_package(pkg: str):
    try:
        importlib.import_module(pkg.replace('-', '_'))
        print(f"- Found: {pkg}")
    except ImportError:
        print_info(f"Installing {pkg} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
            check=True
        )
        print_success(f"Installed {pkg}")


def _ensure_stockfish_engine():
    common_paths = [
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        "stockfish",
        "./stockfish"
    ]

    for p in common_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            print_success(f"Found Stockfish engine at: {p}")
            return p

    print_info("Stockfish engine not found; installing with apt-get ...")
    try:
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "stockfish"], check=True)
    except subprocess.CalledProcessError as e:
        print_error(f"apt-get failed: {e}")
        return None

    if os.path.isfile("/usr/games/stockfish"):
        print_success("Installed Stockfish engine via apt-get")
        return "/usr/games/stockfish"

    print_error("Stockfish engine installation failed")
    return None


def install_dependencies(pipelines):
    print_header("Installing Dependencies")

    required = {pkg for p in pipelines for pkg in p['requirements']}
    for pkg in sorted(required):
        _ensure_python_package(pkg)

    if any(p['name'] == 'eval' for p in pipelines):
        print_header("Stockfish Setup")

        _ensure_python_package("stockfish")

        engine_path = _ensure_stockfish_engine()
        if engine_path:
            os.environ["STOCKFISH_PATH"] = engine_path
            print_success(f"STOCKFISH_PATH set to: {engine_path}")
        else:
            print_error("No usable Stockfish engine found. Evaluation pipeline may fail.")


class Configuration:
    def __init__(self, config_path, mode="test"):
        self.path = config_path
        self.mode = mode
        self.data = self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return {}
        with open(self.path, "r") as f:
            data = yaml.safe_load(f) or {}
        return data

    def get(self, key, default=None):
        if self.mode == "prod":
            prod_key = key.replace(".", ".prod.", 1)
            val = self._get(prod_key)
            if val is not None:
                return val
        return self._get(key, default)

    def _get(self, dotted_key, default=None):
        cur = self.data
        for part in dotted_key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur


def time_format(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m}m"


def run_pipeline(pipeline_info, cfg):
    module_path = f"src.pipelines.{pipeline_info['name']}"
    try:
        print_info(f"Loading module {module_path} …")
        mod = importlib.import_module(module_path)
        Pipeline = getattr(mod, pipeline_info["class_name"])

        print_header(f"RUNNING {pipeline_info['name'].upper()}")
        print(f"Description: {pipeline_info['description']}")

        t0 = time.time()
        success = Pipeline(cfg).run()
        duration = time.time() - t0

        return {
            "name": pipeline_info["name"],
            "status": PipelineStatus.SUCCESS if success else PipelineStatus.FAILED,
            "duration": duration,
        }

    except Exception as exc:
        duration = time.time() - t0 if "t0" in locals() else 0
        print_error(f"Pipeline error: {exc}")
        import traceback; traceback.print_exc()
        return {
            "name": pipeline_info["name"],
            "status": PipelineStatus.FAILED,
            "duration": duration,
            "error": str(exc),
        }


def main() -> int:
    setup_environment()

    print_header("Pipeline Selection")
    for idx, info in PIPELINES.items():
        print(f"  {idx}. {info['name']}: {info['description']}")
    choice = input("\nSelect pipelines (comma-separated numbers): ").strip()

    selected = [PIPELINES[i.strip()] for i in choice.split(",") if i.strip() in PIPELINES]
    if not selected:
        print_error("No valid pipelines selected. Exiting.")
        return 1

    names = ", ".join(p['name'] for p in selected)
    print_info(f"Running pipelines: {names}")

    mode = (input("Select mode (test/prod) [test]: ").strip().lower() or "test")
    install_dependencies(selected)

    cfg = Configuration("/content/drive/MyDrive/chess_ai/config.yml", mode)

    results, start = [], time.time()
    for i, p in enumerate(selected, 1):
        print(f"\n--- PIPELINE {i}/{len(selected)}: {p['name'].upper()} ---")
        r = run_pipeline(p, cfg)
        results.append(r)

        elapsed = time.time() - start
        remaining = elapsed / i * (len(selected) - i) if i else 0
        status = "SUCCESS" if r['status'] == PipelineStatus.SUCCESS else "FAILED"
        print(f"\n{p['name']} finished in {time_format(r['duration'])} – {status}")
        print(f"Total elapsed: {time_format(elapsed)} | ETA: {time_format(remaining)}")

    print_header("Summary")
    for r in results:
        st = "SUCCESS" if r['status'] == PipelineStatus.SUCCESS else "FAILED"
        print(f"  {r['name']}: {st} ({time_format(r['duration'])})")
    succ = sum(r['status'] == PipelineStatus.SUCCESS for r in results)
    print(f"\nTotal: {len(results)}  Successful: {succ}  Failed: {len(results)-succ}")
    print(f"Overall time: {time_format(time.time() - start)}")

    return 0 if succ == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())