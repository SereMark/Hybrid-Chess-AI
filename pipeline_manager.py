import os
import sys
import time
import importlib
import subprocess
from src.utils.config import Config

PIPELINES = {
    '1': {
        'name': 'data',
        'class_name': 'DataPipeline',
        'requirements': ['chess', 'numpy', 'pandas', 'h5py', 'wandb']
    },
    '2': {
        'name': 'supervised',
        'class_name': 'SupervisedPipeline',
        'requirements': ['chess', 'torch', 'numpy', 'pandas', 'wandb']
    },
    '3': {
        'name': 'reinforcement',
        'class_name': 'ReinforcementPipeline',
        'requirements': ['chess', 'torch', 'numpy', 'wandb']
    },
    '4': {
        'name': 'eval',
        'class_name': 'EvalPipeline',
        'requirements': ['chess', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'wandb', 'torch']
    }
}


if __name__ == "__main__":
    project_dir = '/content/drive/MyDrive/chess_ai'

    if not os.path.exists(project_dir):
        os.makedirs(os.path.join(project_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'evaluation'), exist_ok=True)

    os.chdir(project_dir)

    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    Config._instance = None

    print("Pipeline Selection")
    for idx, info in PIPELINES.items():
        print(f"  {idx}. {info['name']}")
    choice = input("\nSelect pipelines (comma-separated numbers): ").strip()

    selected = [PIPELINES[i.strip()] for i in choice.split(",") if i.strip() in PIPELINES]
    if not selected:
        print("No valid pipelines selected!")

    names = ", ".join(p['name'] for p in selected)
    print(f"Running pipelines: {names}")

    mode = (input("Select mode (test/prod) [test]: ").strip().lower() or "test")

    print("Installing Dependencies")

    required = {pkg for p in selected for pkg in p['requirements']}
    for pkg in sorted(required):
        try:
            importlib.import_module(pkg.replace('-', '_'))
            print(f"- Found: {pkg}")
        except ImportError:
            print(f"Installing {pkg} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--quiet"], check=True)
            print(f"Installed {pkg}")

    cfg = Config("/content/drive/MyDrive/chess_ai/config.yml", mode)

    results = []
    for i, p in enumerate(selected, 1):
        print(f"\n--- PIPELINE {i}/{len(selected)}: {p['name'].upper()} ---")
        module_path = f"src.pipelines.{p['name']}"
        try:
            print(f"Loading module {module_path} …")
            mod = importlib.import_module(module_path)
            Pipeline = getattr(mod, p["class_name"])

            print(f"RUNNING {p['name'].upper()}")

            t0 = time.time()
            success = Pipeline(cfg).run()
            duration = time.time() - t0

            r = {
                "name": p["name"],
                "status": "success" if success else "failed",
                "duration": duration,
            }

        except Exception as exc:
            duration = time.time() - t0 if "t0" in locals() else 0
            print(f"Pipeline error: {exc}")
            import traceback; traceback.print_exc()
            r = {
                "name": p["name"],
                "status": "failed",
                "duration": duration,
                "error": str(exc),
            }
        results.append(r)