import os
import sys
import time
import yaml
import shutil
import importlib
import subprocess
from enum import Enum
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field

class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

class HardwareType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    ANY = "any"

@dataclass
class HardwareRequirement:
    type: HardwareType
    high_ram: bool = False
    min_cores: Optional[int] = None
    gpu_type: Optional[str] = None
    description: str = ""
    
    def __str__(self) -> str:
        parts = []
        if self.type == HardwareType.GPU:
            parts.append(f"GPU")
            if self.gpu_type:
                parts.append(f"({self.gpu_type})")
        else:
            parts.append(f"CPU")
            if self.min_cores:
                parts.append(f"({self.min_cores}+ cores)")
        
        if self.high_ram:
            parts.append(f"with High RAM")
            
        return " ".join(parts)

@dataclass
class PipelineResult:
    name: str
    status: PipelineStatus
    duration: float
    error: Optional[str] = None
    
    @property
    def succeeded(self) -> bool:
        return self.status == PipelineStatus.SUCCESS

@dataclass
class PipelineConfig:
    name: str
    description: str
    dependencies: List[str]
    class_name: str
    requirements: List[str]
    hardware: HardwareRequirement = field(default_factory=lambda: HardwareRequirement(HardwareType.ANY))

PIPELINE_REQUIREMENTS = {
    'data': ['chess', 'numpy', 'pandas', 'h5py', 'wandb'],
    'supervised': ['chess', 'torch', 'numpy', 'pandas', 'wandb'],
    'reinforcement': ['chess', 'torch', 'numpy', 'wandb'],
    'eval': ['chess', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'wandb', 'torch']
}

PIPELINES = {
    'data': PipelineConfig(
        name='data', 
        description="Process PGN files into efficient training format",
        dependencies=[],
        class_name='DataPipeline',
        requirements=PIPELINE_REQUIREMENTS['data'],
        hardware=HardwareRequirement(
            type=HardwareType.CPU,
            high_ram=True,
            min_cores=4,
            description="CPU-based data processing (saves GPU credits) with high RAM for large datasets"
        )
    ),
    'supervised': PipelineConfig(
        name='supervised', 
        description="Train model on human games with supervised learning",
        dependencies=['data'],
        class_name='SupervisedPipeline',
        requirements=PIPELINE_REQUIREMENTS['supervised'],
        hardware=HardwareRequirement(
            type=HardwareType.GPU,
            high_ram=True,
            gpu_type="A100",
            description="Heavy GPU training with large batch sizes"
        )
    ),
    'reinforcement': PipelineConfig(
        name='reinforcement', 
        description="Improve model through self-play reinforcement learning",
        dependencies=['supervised'],
        class_name='ReinforcementPipeline',
        requirements=PIPELINE_REQUIREMENTS['reinforcement'],
        hardware=HardwareRequirement(
            type=HardwareType.GPU,
            high_ram=True,
            gpu_type="A100",
            description="Intensive GPU training with parallel self-play"
        )
    ),
    'eval': PipelineConfig(
        name='eval', 
        description="Evaluate models and benchmark against Stockfish",
        dependencies=['data'],
        class_name='EvalPipeline',
        requirements=PIPELINE_REQUIREMENTS['eval'],
        hardware=HardwareRequirement(
            type=HardwareType.GPU,
            high_ram=True,
            gpu_type="A100",
            description="GPU-accelerated model evaluation and benchmarking"
        )
    )
}

PIPELINE_ORDER = ['data', 'supervised', 'reinforcement', 'eval']

def print_header(title):
    print(f"\n=== {title} ===\n")

def print_info(message):
    print(f"INFO: {message}")

def print_success(message):
    print(f"SUCCESS: {message}")

def print_error(message):
    print(f"ERROR: {message}")

def print_warning(message):
    print(f"WARNING: {message}")

def print_step(step_num, message, total_steps=None):
    if total_steps:
        print(f"\n--- Step {step_num}/{total_steps}: {message} ---")
    else:
        print(f"\n--- Step {step_num}: {message} ---")

def get_input(prompt, default=None):
    if default:
        prompt_text = f"{prompt} [{default}]: "
    else:
        prompt_text = f"{prompt}: "
    
    response = input(prompt_text).strip()
    
    if not response and default:
        return default
    return response

def confirm(message, default="y"):
    response = get_input(f"{message} (y/n)", default)
    return response.lower() in ("y", "yes")

def setup_environment():
    print_info("Setting up environment...")
    try:
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
            
        try:
            import torch
            original_load = torch.load
            
            def safe_torch_load(f, map_location=None, pickle_module=None, **kwargs):
                if map_location is None:
                    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
                try:
                    return original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
                except RuntimeError:
                    print(f"Warning: Detected loading issue, trying with CPU mapping...")
                    return original_load(f, map_location='cpu', pickle_module=pickle_module, **kwargs)
                    
            torch.load = safe_torch_load
            
        except Exception as e:
            print(f"Failed to install CUDA patch: {e}")
    except Exception as e:
        print(f"Error in setup: {e}")

    print_success(f"Environment ready at: {project_dir}")
    return project_dir

def detect_completed_pipelines():
    completed = {}
    
    completed['data'] = os.path.exists('/content/drive/MyDrive/chess_ai/data/train_indices.npy')
    completed['supervised'] = os.path.exists('/content/drive/MyDrive/chess_ai/models/supervised_model.pth')
    completed['reinforcement'] = os.path.exists('/content/drive/MyDrive/chess_ai/models/reinforcement_model.pth')
    completed['eval'] = os.path.exists('/content/drive/MyDrive/chess_ai/evaluation/evaluation_report.txt')
    
    return completed

def check_hardware():
    hardware_info = {
        "gpu": False,
        "high_ram_enabled": False,
        "cpu_cores": os.cpu_count() or 1,
        "drive_space_gb": None
    }
    
    try:
        try:
            import torch
            hardware_info["gpu"] = torch.cuda.is_available()
            if hardware_info["gpu"]:
                hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
                hardware_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass
            
        try:
            import psutil
            total_ram = psutil.virtual_memory().total / (1024**3)
            hardware_info["high_ram_enabled"] = total_ram > 20.0
            hardware_info["ram_gb"] = total_ram
        except:
            pass
            
        try:
            import psutil
            hardware_info["drive_space_gb"] = psutil.disk_usage('/content/drive').free / (1024**3) if os.path.exists('/content/drive') else None
        except:
            pass
    except:
        pass
        
    return hardware_info

def display_hardware_info(hardware):
    print_header("Hardware Configuration")
    
    if hardware["gpu"]:
        print(f"Device: GPU - {hardware.get('gpu_name', 'Unknown')}")
        if hardware.get("gpu_memory"):
            print(f"GPU memory: {hardware['gpu_memory']:.1f}GB")
    else:
        print("Device: CPU")
    
    print(f"High RAM: {'Enabled' if hardware.get('high_ram_enabled', False) else 'Disabled'}")
    
    if hardware.get("ram_gb"):
        print(f"Total RAM: {hardware.get('ram_gb', 0):.1f}GB")
    
    print(f"CPU Cores: {hardware.get('cpu_cores', 0)}")
    
    if hardware.get("drive_space_gb"):
        space = hardware.get("drive_space_gb")
        print(f"Drive Space: {space:.1f}GB free")
        
        if space < 5:
            print_warning("Low disk space warning!")

def check_dependencies(pipelines):
    print_header("Dependency Check")
    
    all_requirements = set()
    for pipeline in pipelines:
        all_requirements.update(PIPELINES[pipeline].requirements)
    
    missing_packages = []
    print_info("Scanning installed packages...")
    for package in all_requirements:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"- Package '{package}' is installed")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print_warning(f"Found {len(missing_packages)} missing dependencies: {', '.join(missing_packages)}")
        if confirm("Install missing dependencies now?"):
            for i, package in enumerate(missing_packages):
                try:
                    print(f"Installing {package}...")
                    _ = subprocess.run(
                        [sys.executable, "-m", "pip", "install", package],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    print_success(f"Installed {package}")
                except subprocess.CalledProcessError as e:
                    print_error(f"Failed to install {package}: {e.stderr}")
    else:
        print_success("All required dependencies are already installed")

    if "eval" in pipelines:
        print_header("Stockfish Check")
        if not shutil.which("stockfish"):
            print_info("Stockfish not found on PATH, installing via apt...")
            try:
                print("Updating apt cache...")
                subprocess.run(
                    ["apt-get", "update"],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                print("Installing Stockfish...")
                subprocess.run(
                    ["apt-get", "install", "-y", "stockfish"],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                if shutil.which("stockfish"):
                    print_success("Stockfish installed and available on PATH")
                else:
                    print_error("Stockfish installation completed but binary not found on PATH")
            except subprocess.CalledProcessError as e:
                print_error(f"Failed to install Stockfish: {e}")
        else:
            print_success("Stockfish binary already present")

class Configuration:
    def __init__(self, config_path, mode="test"):
        self.path = config_path
        self.mode = mode
        self.data = self._load()
    
    def _load(self):
        if not os.path.exists(self.path):
            print_warning(f"Config file not found: {self.path}")
            return {}
        
        with open(self.path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key, default=None):
        keys = key.split('.')
        
        if len(keys) > 1 and keys[0] in self.data:
            section = self.data[keys[0]]
            if isinstance(section, dict):
                if self.mode == "prod" and "prod" in section and keys[1] in section["prod"]:
                    return section["prod"][keys[1]]
                
                if keys[1] in section:
                    if len(keys) == 2:
                        return section[keys[1]]
                    return self._get_nested(section, keys[1:], default)
        
        return self._get_nested(self.data, keys, default)
    
    def _get_nested(self, section, keys, default):
        current = section
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

def load_configuration(config_path, mode):
    print_info(f"Loading {mode} configuration from {config_path}...")
    config = Configuration(config_path, mode)
    print_success(f"Configuration loaded: {config_path} ({mode} mode)")
    return config

def resolve_dependencies(selected_pipelines, completed):
    required = set(selected_pipelines)
    skipped_deps = []
    
    for pipeline in selected_pipelines:
        for dependency in PIPELINES[pipeline].dependencies:
            if dependency not in required:
                if completed.get(dependency, False):
                    skipped_deps.append(dependency)
                else:
                    required.add(dependency)
    
    ordered = [p for p in PIPELINE_ORDER if p in required]
    return ordered, skipped_deps

def display_pipelines(completed_pipelines=None, hardware_info=None):
    print_header("Pipeline Selection")
    
    if completed_pipelines is None:
        completed_pipelines = detect_completed_pipelines()
    
    print_info("Available pipelines:")
    
    for i, name in enumerate(PIPELINE_ORDER, 1):
        pipeline_config = PIPELINES[name]
        is_completed = completed_pipelines.get(name, False)
        
        status = "[✓]" if is_completed else "[ ]"
        depends = ", ".join(pipeline_config.dependencies) if pipeline_config.dependencies else "None"
        
        print(f"  {i}. {status} {name}: {pipeline_config.description} (Dependencies: {depends})")
        
        hw_req = pipeline_config.hardware
        hw_type = "GPU" if hw_req.type == HardwareType.GPU else "CPU"
        
        print(f"     Hardware: {hw_type}{' (High RAM)' if hw_req.high_ram else ''}")
        
        if hardware_info:
            hw_match = True
            if (hw_req.type == HardwareType.GPU and not hardware_info.get("gpu", False)) or \
               (hw_req.type == HardwareType.CPU and hardware_info.get("cpu_cores", 0) < hw_req.min_cores) or \
               (hw_req.high_ram and not hardware_info.get("high_ram_enabled", False)):
                hw_match = False
                
            if not hw_match:
                print(f"     WARNING: Current hardware may not be optimal for this pipeline")
                
                if name == "data" and hardware_info.get("gpu", False):
                    print(f"     TIP: Switch to CPU-only runtime to save GPU credits for this pipeline")

def show_execution_plan(pipelines, skipped, hardware_info=None):
    print_header("Execution Plan")
    
    print(f"Execution flow: {' -> '.join(pipelines)}")
    
    if skipped:
        print(f"Skipped Dependencies: {', '.join(skipped)} (using existing output)")
    
    if hardware_info:
        for pipeline in pipelines:
            hw_req = PIPELINES[pipeline].hardware
            if hw_req.type == HardwareType.GPU and not hardware_info.get("gpu", False):
                print_warning(f"Pipeline '{pipeline}' requires GPU but running on CPU")
            elif hw_req.high_ram and not hardware_info.get("high_ram_enabled", False):
                print_warning(f"Pipeline '{pipeline}' requires High RAM but it appears to be disabled")

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

def run_pipeline(pipeline_name, config, hardware_info):
    module_path = f"src.pipelines.{pipeline_name}"
    
    try:
        print_info(f"Loading pipeline module {pipeline_name}...")
        module = importlib.import_module(module_path)
        pipeline_class = getattr(module, PIPELINES[pipeline_name].class_name)
        
        print_header(f"{pipeline_name.upper()} PIPELINE")
        
        desc = PIPELINES[pipeline_name].description
        deps = ", ".join(PIPELINES[pipeline_name].dependencies) if PIPELINES[pipeline_name].dependencies else "None"
        
        print(f"Description: {desc}")
        print(f"Dependencies: {deps}")
        
        hw_req = PIPELINES[pipeline_name].hardware
        print(f"Hardware: {hw_req}")
        
        hardware_match = True
        if hw_req.type == HardwareType.GPU and not hardware_info.get("gpu", False):
            hardware_match = False
            print_warning("GPU required but running on CPU")
            print_info("Change to: Runtime → Change runtime type → Hardware accelerator → GPU → A100")
        elif hw_req.type == HardwareType.CPU and hardware_info.get("gpu", True):
            if pipeline_name == "data":
                print_info("Data processing runs efficiently on CPU (saves GPU credits)")
                print_info("Consider: Runtime → Change runtime type → Hardware accelerator → None")
        
        if hw_req.high_ram and not hardware_info.get("high_ram_enabled", False):
            hardware_match = False
            print_warning("High RAM toggle required but appears to be disabled")
            print_info("Enable: Runtime → Change runtime type → Hardware accelerator → High-RAM")
            
        start_time = time.time()
        pipeline = pipeline_class(config)
        
        print_info("Starting pipeline execution...")
        
        if not hardware_match:
            print_warning("Performance will be significantly impacted by hardware limitations")
        elif pipeline_name == "data" and hardware_info.get("gpu", False):
            print_info("Data pipeline is running on GPU but would be more efficient on CPU to save GPU credits")
            
        success = pipeline.run()
        
        end_time = time.time()
        duration = end_time - start_time
        
        status = PipelineStatus.SUCCESS if success else PipelineStatus.FAILED
        
        return PipelineResult(
            name=pipeline_name,
            status=status,
            duration=duration
        )
    except Exception as e:
        end_time = time.time() if 'start_time' in locals() else time.time()
        duration = end_time - start_time if 'start_time' in locals() else 0
        
        print_error(f"Pipeline error: {e}")
        
        return PipelineResult(
            name=pipeline_name,
            status=PipelineStatus.FAILED,
            duration=duration,
            error=str(e)
        )

def print_summary(results, hardware_info=None):
    print_header("Execution Summary")
    
    total_time = sum(result.duration for result in results)
    success_count = sum(1 for result in results if result.succeeded)
    
    print("Pipeline Results:")
    for result in results:
        status = "SUCCESS" if result.succeeded else "FAILED"
        print(f"  {result.name}: {status} ({time_format(result.duration)})")
    
    print(f"\nTotal pipelines: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    print(f"Total execution time: {time_format(total_time)}")
    
    if hardware_info:
        device_type = "GPU" if hardware_info.get("gpu", False) else "CPU"
        device_info = f"Hardware: {device_type}"
        if device_type == "GPU" and hardware_info.get("gpu_name"):
            device_info += f" ({hardware_info.get('gpu_name')})"
        print(device_info)
    
    if all(result.succeeded for result in results):
        print_success("\nAll pipelines completed successfully!")
    else:
        failed = [result.name for result in results if not result.succeeded]
        print_error(f"\nSome pipelines failed: {', '.join(failed)}")
        print_info("Check logs for more details")

def main():
    try:
        print("\nCHESS AI PIPELINE MANAGER")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        project_dir = setup_environment()
        
        hardware = check_hardware()
        display_hardware_info(hardware)
        
        print_step(1, "Select Running Mode", 3)
        
        mode = get_input("Select running mode (test/prod)", "test")
        
        if mode == "prod":
            print_warning("Production mode will use full datasets and longer training times.")
            print_info("Make sure you have enough resources and time for completion.")
        
        config_path = os.path.join(project_dir, 'config.yml')
        config = load_configuration(config_path, mode)
        
        completed = detect_completed_pipelines()
        display_pipelines(completed, hardware)
        
        completed_list = [p for p, v in completed.items() if v]
        if completed_list:
            print_info(f"Previously completed: {', '.join(completed_list)}")
        else:
            print_info("No previously completed pipelines")
            
        pipeline_names = PIPELINE_ORDER.copy()
        
        pipeline_input = get_input("Select pipelines to run (comma-separated numbers, or press Enter for all)")
        
        if not pipeline_input:
            selected_pipeline_names = pipeline_names
            print_success(f"Selected all pipelines: {', '.join(selected_pipeline_names)}")
        else:
            try:
                selected_indices = [int(idx.strip()) for idx in pipeline_input.split(',')]
                selected_pipeline_names = []
                
                for idx in selected_indices:
                    if 1 <= idx <= len(pipeline_names):
                        selected_pipeline_names.append(pipeline_names[idx-1])
                    else:
                        print_error(f"Invalid pipeline number: {idx}. Ignoring.")
                
                if not selected_pipeline_names:
                    print_error("No valid pipelines selected. Exiting.")
                    return 1
                    
                print_success(f"Selected pipelines: {', '.join(selected_pipeline_names)}")
            except ValueError:
                print_error("Invalid input format. Please enter numbers separated by commas.")
                return 1
        
        ordered_pipelines, skipped_deps = resolve_dependencies(selected_pipeline_names, completed)
        
        print_step(2, "Verify Dependencies", 3)
        check_dependencies(ordered_pipelines)
        
        show_execution_plan(ordered_pipelines, skipped_deps, hardware)
        
        print_step(3, "Execute Pipelines", 3)
        
        if not confirm("Proceed with execution?"):
            print_warning("Operation cancelled by user.")
            return 0
        
        results = []
        execution_start = time.time()
        
        for i, pipeline in enumerate(ordered_pipelines, 1):
            pipeline_start = time.time()
            print(f"\n--- PIPELINE {i}/{len(ordered_pipelines)}: {pipeline.upper()} ---")
            
            result = run_pipeline(pipeline, config, hardware)
            results.append(result)
            
            pipeline_time = time.time() - pipeline_start
            elapsed_total = time.time() - execution_start
            remaining_pipelines = len(ordered_pipelines) - (i)
            estimated_remaining = elapsed_total / i * remaining_pipelines if i > 0 else 0
            
            status = "SUCCESS" if result.succeeded else "FAILED"
            print(f"\n{pipeline} completed in {time_format(pipeline_time)} - Status: {status}")
            print(f"Elapsed: {time_format(elapsed_total)} | Estimated remaining: {time_format(estimated_remaining)}")
            
            if not result.succeeded:
                dependents = [p for p in ordered_pipelines[i:] 
                              if pipeline in PIPELINES[p].dependencies]
                if dependents:
                    print_warning(f"This failure may affect later pipelines: {', '.join(dependents)}")
                    if not confirm("Continue with execution?"):
                        print_error("Execution aborted by user after pipeline failure.")
                        break
            
            if i < len(ordered_pipelines):
                next_pipeline = ordered_pipelines[i]
                next_hw_req = PIPELINES[next_pipeline].hardware
                
                if pipeline == "data" and next_hw_req.type == HardwareType.GPU and not hardware.get("gpu", False):
                    print_warning(f"Next pipeline '{next_pipeline}' requires GPU but you're currently on CPU.")
                    print_info("Recommended action: Runtime → Change runtime type → Hardware accelerator → A100")
                elif next_hw_req.type == HardwareType.CPU and hardware.get("gpu", False):
                    print_info(f"Next pipeline '{next_pipeline}' runs efficiently on CPU.")
                    print_info("To save GPU credits: Runtime → Change runtime type → Hardware accelerator → CPU")
        
        print_summary(results, hardware)
        
        if all(result.succeeded for result in results):
            if "eval" in [r.name for r in results if r.succeeded]:
                print_info("Check the evaluation results in the Google Drive folder:")
                print(f"/content/drive/MyDrive/chess_ai/evaluation")
        
        return 0 if all(result.succeeded for result in results) else 1
        
    except KeyboardInterrupt:
        print_warning("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        print_error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())