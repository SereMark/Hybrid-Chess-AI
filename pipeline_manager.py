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

def print_header(title):
    print(f"\n=== {title} ===\n")

def print_info(message):
    print(f"INFO: {message}")

def print_success(message):
    print(f"SUCCESS: {message}")

def print_error(message):
    print(f"ERROR: {message}")

def setup_environment():
    project_dir = '/content/drive/MyDrive/chess_ai'
    
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
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

def install_dependencies(pipelines):
    print_header("Installing Dependencies")
    
    all_requirements = set()
    for pipeline in pipelines:
        all_requirements.update(pipeline['requirements'])
    
    missing_packages = []
    for package in all_requirements:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"- Found: {package}")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print_info(f"Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            try:
                print(f"Installing {package}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            except subprocess.CalledProcessError:
                print_error(f"Failed to install {package}")
    else:
        print_success("All dependencies already installed")
    
    for pipeline in pipelines:
        if pipeline['name'] == 'eval':
            try:
                result = subprocess.run(["which", "stockfish"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode != 0:
                    print_info("Installing Stockfish...")
                    subprocess.run(["apt-get", "update"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    subprocess.run(["apt-get", "install", "-y", "stockfish"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError:
                print_error("Failed to install Stockfish")
            break

class Configuration:
    def __init__(self, config_path, mode="test"):
        self.path = config_path
        self.mode = mode
        self.data = self._load()
    
    def _load(self):
        if not os.path.exists(self.path):
            return {}
        
        with open(self.path, 'r') as f:
            data = yaml.safe_load(f)
            return data if data else {}
    
    def get(self, key, default=None):
        keys = key.split('.')
        
        if self.mode == "prod":
            prod_keys = keys.copy()
            prod_keys.insert(-1, "prod")
            prod_key = '.'.join(prod_keys)
            
            prod_value = self._get_value(prod_key)
            if prod_value is not None:
                return prod_value
        
        return self._get_value(key, default)
    
    def _get_value(self, key, default=None):
        keys = key.split('.')
        current = self.data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current

def time_format(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

def run_pipeline(pipeline_info, config):
    module_path = f"src.pipelines.{pipeline_info['name']}"
    
    try:
        print_info(f"Loading pipeline module {pipeline_info['name']}...")
        module = importlib.import_module(module_path)
        pipeline_class = getattr(module, pipeline_info['class_name'])
        
        print_header(f"RUNNING {pipeline_info['name'].upper()}")
        print(f"Description: {pipeline_info['description']}")
        
        start_time = time.time()
        pipeline = pipeline_class(config)
        
        print_info("Starting pipeline execution...")
        success = pipeline.run()
        
        end_time = time.time()
        duration = end_time - start_time
        
        status = PipelineStatus.SUCCESS if success else PipelineStatus.FAILED
        
        return {
            'name': pipeline_info['name'],
            'status': status,
            'duration': duration
        }
    except Exception as e:
        end_time = time.time() if 'start_time' in locals() else time.time()
        duration = end_time - start_time if 'start_time' in locals() else 0
        
        print_error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'name': pipeline_info['name'],
            'status': PipelineStatus.FAILED,
            'duration': duration,
            'error': str(e)
        }

def main():
    setup_environment()
    
    print_header("Pipeline Selection")
    for idx, info in PIPELINES.items():
        print(f"  {idx}. {info['name']}: {info['description']}")
    
    pipeline_input = input("\nSelect pipelines to run (comma-separated numbers): ").strip()
    
    selected_pipelines = []
    if pipeline_input:
        for idx in pipeline_input.split(','):
            idx = idx.strip()
            if idx in PIPELINES:
                selected_pipelines.append(PIPELINES[idx])
    
    if not selected_pipelines:
        print_error("No valid pipelines selected. Exiting.")
        return 1
    
    pipeline_names = [p['name'] for p in selected_pipelines]
    print_info(f"Running pipelines: {', '.join(pipeline_names)}")
    
    mode = input("Select mode (test/prod) [test]: ").strip().lower() or "test"
    
    install_dependencies(selected_pipelines)
    
    config_path = os.path.join('/content/drive/MyDrive/chess_ai', 'config.yml')
    config = Configuration(config_path, mode)
    
    results = []
    execution_start = time.time()
    
    for i, pipeline_info in enumerate(selected_pipelines, 1):
        pipeline_start = time.time()
        print(f"\n--- PIPELINE {i}/{len(selected_pipelines)}: {pipeline_info['name'].upper()} ---")
        
        result = run_pipeline(pipeline_info, config)
        results.append(result)
        
        pipeline_time = time.time() - pipeline_start
        elapsed_total = time.time() - execution_start
        remaining_pipelines = len(selected_pipelines) - i
        estimated_remaining = elapsed_total / i * remaining_pipelines if i > 0 else 0
        
        status = "SUCCESS" if result['status'] == PipelineStatus.SUCCESS else "FAILED"
        print(f"\n{pipeline_info['name']} completed in {time_format(pipeline_time)} - Status: {status}")
        print(f"Elapsed: {time_format(elapsed_total)} | Remaining: {time_format(estimated_remaining)}")
    
    print_header("Summary")
    
    success_count = sum(1 for result in results if result['status'] == PipelineStatus.SUCCESS)
    
    print("Results:")
    for result in results:
        status = "SUCCESS" if result['status'] == PipelineStatus.SUCCESS else "FAILED"
        print(f"  {result['name']}: {status} ({time_format(result['duration'])})")
    
    print(f"\nTotal: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    print(f"Time: {time_format(time.time() - execution_start)}")
    
    return 0 if all(result['status'] == PipelineStatus.SUCCESS for result in results) else 1

if __name__ == "__main__":
    sys.exit(main())