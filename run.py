#!/usr/bin/env python3

import importlib
import time
import sys
import os
import yaml
import subprocess
from typing import Any, List, Dict, Optional, Tuple
from datetime import datetime
import threading
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

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

PIPELINE_REQUIREMENTS = {
    'data': ['chess', 'numpy', 'pandas', 'h5py', 'wandb'],
    'hyperopt': ['optuna', 'numpy', 'pandas', 'wandb'],
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
        requirements=PIPELINE_REQUIREMENTS['data']
    ),
    'hyperopt': PipelineConfig(
        name='hyperopt', 
        description="Find optimal hyperparameters for the model",
        dependencies=['data'],
        class_name='HyperoptPipeline',
        requirements=PIPELINE_REQUIREMENTS['hyperopt']
    ),
    'supervised': PipelineConfig(
        name='supervised', 
        description="Train model on human games with supervised learning",
        dependencies=['data'],
        class_name='SupervisedPipeline',
        requirements=PIPELINE_REQUIREMENTS['supervised']
    ),
    'reinforcement': PipelineConfig(
        name='reinforcement', 
        description="Improve model through self-play reinforcement learning",
        dependencies=['supervised'],
        class_name='ReinforcementPipeline',
        requirements=PIPELINE_REQUIREMENTS['reinforcement']
    ),
    'eval': PipelineConfig(
        name='eval', 
        description="Evaluate models and benchmark against Stockfish",
        dependencies=['data'],
        class_name='EvalPipeline',
        requirements=PIPELINE_REQUIREMENTS['eval']
    )
}

PIPELINE_ORDER = ['data', 'hyperopt', 'supervised', 'reinforcement', 'eval']

WORKFLOWS = {
    "full": {
        "name": "Full Training Pipeline",
        "description": "Complete workflow from data to evaluation",
        "pipelines": ['data', 'supervised', 'reinforcement', 'eval'],
        "icon": "🔄"
    },
    "train": {
        "name": "Training Only",
        "description": "Train models without evaluation",
        "pipelines": ['data', 'supervised', 'reinforcement'],
        "icon": "🧠"
    },
    "quick": {
        "name": "Quick SL Training",
        "description": "Train supervised model only",
        "pipelines": ['data', 'supervised'],
        "icon": "⚡"
    },
    "evaluate": {
        "name": "Model Evaluation",
        "description": "Evaluate existing models",
        "pipelines": ['data', 'eval'],
        "icon": "📊"
    },
    "optimize": {
        "name": "Hyperparameter Optimization",
        "description": "Find optimal model parameters",
        "pipelines": ['data', 'hyperopt'],
        "icon": "🔍"
    }
}

class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLACK = "\033[30m"
    RED = "\033[38;5;203m"
    GREEN = "\033[38;5;107m"
    YELLOW = "\033[38;5;221m"
    BLUE = "\033[38;5;75m"
    MAGENTA = "\033[38;5;176m"
    CYAN = "\033[38;5;116m"
    WHITE = "\033[38;5;255m"
    GRAY = "\033[38;5;246m"
    
    BG_BLACK = "\033[40m"
    BG_RED = "\033[48;5;203m"
    BG_GREEN = "\033[48;5;107m"
    BG_YELLOW = "\033[48;5;221m"
    BG_BLUE = "\033[48;5;75m"
    BG_MAGENTA = "\033[48;5;176m"
    BG_CYAN = "\033[48;5;116m"
    BG_WHITE = "\033[48;5;255m"
    BG_GRAY = "\033[48;5;240m"
    
    SUCCESS = GREEN
    ERROR = RED
    WARNING = YELLOW
    INFO = BLUE
    MUTED = GRAY
    
    SYMBOL_SUCCESS = "✓"
    SYMBOL_ERROR = "✗"
    SYMBOL_WARNING = "⚠"
    SYMBOL_INFO = "ℹ"
    SYMBOL_ARROW = "→"
    SYMBOL_BULLET = "•"
    SYMBOL_PLAY = "▶"
    SYMBOL_STAR = "★"
    SYMBOL_GEAR = "⚙"
    SYMBOL_CIRCLE = "○"
    SYMBOL_CHECKED = "✓"
    SYMBOL_SKIP = "↷"
    
    @staticmethod
    def apply(text, *styles):
        result = ""
        for style in styles:
            result += style
        result += text + Style.RESET
        return result

class Console:
    @staticmethod
    def clear():
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def banner(title, subtitle=None):
        width = 80
        Console.clear()
        
        print(f"\n{Style.apply('━' * width, Style.BLUE, Style.BOLD)}")
        print(f"{Style.apply('┃', Style.BLUE, Style.BOLD)} {Style.apply(title.center(width-4), Style.BLUE, Style.BOLD)} {Style.apply('┃', Style.BLUE, Style.BOLD)}")
        if subtitle:
            print(f"{Style.apply('┃', Style.BLUE, Style.BOLD)} {Style.apply(subtitle.center(width-4), Style.BLUE)} {Style.apply('┃', Style.BLUE, Style.BOLD)}")
        print(f"{Style.apply('━' * width, Style.BLUE, Style.BOLD)}\n")
    
    @staticmethod
    def header(title):
        print(f"\n{Style.apply('┏━━ ', Style.BLUE, Style.BOLD)}{Style.apply(title, Style.BLUE, Style.BOLD)}{Style.apply(' ━━', Style.BLUE, Style.BOLD)}")
        
    @staticmethod
    def footer():
        print(f"{Style.apply('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', Style.BLUE)}\n")
    
    @staticmethod
    def info(message):
        print(f"{Style.apply('ℹ', Style.INFO)} {message}")
    
    @staticmethod
    def success(message):
        print(f"{Style.apply('✓', Style.SUCCESS, Style.BOLD)} {message}")
    
    @staticmethod
    def error(message):
        print(f"{Style.apply('✗', Style.ERROR, Style.BOLD)} {message}")
    
    @staticmethod
    def warning(message):
        print(f"{Style.apply('⚠', Style.WARNING, Style.BOLD)} {message}")
    
    @staticmethod
    def step(step_num, message):
        print(f"\n{Style.apply(f'Step {step_num}:', Style.BLUE, Style.BOLD)} {message}")
    
    @staticmethod
    def feature(message):
        print(f"  {Style.apply('★', Style.CYAN)} {message}")
    
    @staticmethod
    def input(prompt, default=None):
        if default:
            prompt_text = f"{prompt} [{default}]: "
        else:
            prompt_text = f"{prompt}: "
        
        sys.stdout.write(f"{Style.apply('➤', Style.GREEN)} {Style.apply(prompt_text, Style.BOLD)}")
        sys.stdout.flush()
        response = input().strip()
        
        if not response and default:
            return default
        return response
    
    @staticmethod
    def select(options, title="Select an option", default=None):
        Console.header(title)
        
        for i, (key, option) in enumerate(options.items(), 1):
            default_mark = f" {Style.apply('(default)', Style.YELLOW)}" if key == default else ""
            icon = option.get("icon", "•")
            name = option.get("name", key)
            desc = option.get("description", "")
            
            print(f"  {Style.apply(str(i), Style.CYAN)} {Style.apply(icon, Style.CYAN)} "
                  f"{Style.apply(name, Style.BOLD)} - {desc}{default_mark}")
        
        while True:
            choice = Console.input("Enter your choice", default)
            
            try:
                if choice.isdigit() and 1 <= int(choice) <= len(options):
                    return list(options.keys())[int(choice) - 1]
                
                if choice in options:
                    return choice
                
                Console.error(f"Invalid selection. Please enter a number between 1-{len(options)} or a valid option name.")
            except Exception:
                Console.error("Invalid selection. Please try again.")
    
    @staticmethod
    def confirm(message, default="y"):
        response = Console.input(f"{message} (y/n)", default)
        return response.lower() in ("y", "yes")

class Spinner:
    def __init__(self, message="Processing"):
        self.message = message
        self.running = False
        self.spinner_thread = None
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.delay = 0.1
        
    def spin(self):
        i = 0
        while self.running:
            frame = self.frames[i % len(self.frames)]
            sys.stdout.write(f"\r{Style.apply(frame, Style.CYAN)} {self.message}")
            sys.stdout.flush()
            time.sleep(self.delay)
            i += 1
            
    def start(self):
        self.running = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
        
    def stop(self):
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()

class ProgressBar:
    def __init__(self, total, prefix="", size=30):
        self.total = total
        self.prefix = prefix
        self.size = size
        self.start_time = time.time()
        self.count = 0

    def update(self, count=1):
        self.count += count
        filled_length = int(self.size * self.count / self.total)
        bar = f"{Style.apply('█' * filled_length, Style.GREEN)}{Style.apply('░' * (self.size - filled_length), Style.GRAY)}"
        elapsed_time = time.time() - self.start_time
        percent = 100 * self.count / self.total
        
        if self.total == 0:
            eta = 0
        elif self.count == 0:
            eta = 0  
        else:
            eta = elapsed_time * (self.total - self.count) / self.count
        
        eta_str = f"{eta:.1f}s remaining" if eta > 0 else "completed"
        
        sys.stdout.write(f"\r{self.prefix} |{bar}| {percent:.1f}% ({self.count}/{self.total}) [{elapsed_time:.1f}s, {eta_str}]")
        sys.stdout.flush()
        
        if self.count >= self.total:
            sys.stdout.write('\n')

@contextmanager
def spinner_context(message):
    spinner = Spinner(message)
    spinner.start()
    try:
        yield spinner
    finally:
        spinner.stop()

def setup_environment():
    with spinner_context("Setting up environment"):
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
                        map_location = 'cpu'
                    try:
                        return original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
                    except RuntimeError as e:
                        if "torch.storage.UntypedStorage (tagged with xla:0)" in str(e):
                            print("Detected TPU storage issue, trying with CPU mapping...")
                            return original_load(f, map_location='cpu', pickle_module=pickle_module, **kwargs)
                        raise
                        
                torch.load = safe_torch_load
                
            except Exception as e:
                print(f"Failed to install TPU patch: {e}")
        except Exception as e:
            print(f"Error in setup: {e}")

    Console.success(f"Environment ready at: {project_dir}")
    return project_dir

def detect_completed_pipelines():
    completed = {}
    
    completed['data'] = os.path.exists('/content/drive/MyDrive/chess_ai/data/train_indices.npy')
    completed['supervised'] = os.path.exists('/content/drive/MyDrive/chess_ai/models/supervised_model.pth')
    completed['reinforcement'] = os.path.exists('/content/drive/MyDrive/chess_ai/models/reinforcement_model.pth')
    completed['hyperopt'] = os.path.exists('/content/drive/MyDrive/chess_ai/hyperopt_results/best_trial.txt') 
    completed['eval'] = os.path.exists('/content/drive/MyDrive/chess_ai/evaluation/evaluation_report.txt')
    
    return completed

def check_hardware():
    hardware_info = {
        "gpu": False,
        "tpu": False,
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": None,
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
            import torch_xla.core.xla_model as xm
            hardware_info["tpu"] = True
        except:
            pass
            
        try:
            import psutil
            hardware_info["ram_gb"] = psutil.virtual_memory().total / (1024**3)
            hardware_info["drive_space_gb"] = psutil.disk_usage('/content/drive').free / (1024**3) if os.path.exists('/content/drive') else None
        except:
            pass
    except:
        pass
        
    return hardware_info

def check_dependencies(pipelines):
    Console.header("Checking Dependencies")
    
    all_requirements = set()
    for pipeline in pipelines:
        all_requirements.update(PIPELINES[pipeline].requirements)
    
    missing_packages = []
    with spinner_context("Scanning installed packages"):
        for package in all_requirements:
            try:
                importlib.import_module(package.replace('-', '_'))
                Console.info(f"Package '{package}' is already installed")
            except ImportError:
                missing_packages.append(package)
    
    if missing_packages:
        Console.warning(f"Found {len(missing_packages)} missing dependencies: {', '.join(missing_packages)}")
        if Console.confirm("Install missing dependencies now?"):
            progress = ProgressBar(len(missing_packages), prefix=f"{Style.apply('Installing packages:', Style.BOLD)}", size=30)
            
            for i, package in enumerate(missing_packages):
                try:
                    with spinner_context(f"Installing {package}"):
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", package],
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                    Console.success(f"Installed {package}")
                except subprocess.CalledProcessError as e:
                    Console.error(f"Failed to install {package}: {e.stderr}")
                
                progress.update(1)
    else:
        Console.success("All required dependencies are already installed")

class Configuration:
    def __init__(self, config_path, mode="test"):
        self.path = config_path
        self.mode = mode
        self.data = self._load()
    
    def _load(self):
        if not os.path.exists(self.path):
            Console.warning(f"Config file not found: {self.path}")
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
    with spinner_context(f"Loading {mode} configuration"):
        config = Configuration(config_path, mode)
    
    Console.success(f"Configuration loaded: {config_path} ({mode} mode)")
    return config

def resolve_dependencies(selected_pipelines, completed):
    required = set(selected_pipelines)
    skipped_deps = []
    
    pending_deps = []
    for pipeline in selected_pipelines:
        if pipeline in PIPELINES:
            for dependency in PIPELINES[pipeline].dependencies:
                if dependency not in required:
                    if completed.get(dependency, False):
                        skipped_deps.append(dependency)
                    else:
                        pending_deps.append(dependency)
                        required.add(dependency)
    
    while pending_deps:
        current_dep = pending_deps.pop(0)
        for sub_dep in PIPELINES[current_dep].dependencies:
            if sub_dep not in required:
                if completed.get(sub_dep, False):
                    skipped_deps.append(sub_dep)
                else:
                    pending_deps.append(sub_dep)
                    required.add(sub_dep)
    
    ordered = [p for p in PIPELINE_ORDER if p in required]
    return ordered, skipped_deps

def display_pipelines(pipelines):
    console_width = 80
    col_width = console_width // 2 - 4
    
    Console.header("Pipeline Status")
    
    completed = detect_completed_pipelines()
    
    rows = []
    for i in range(0, len(PIPELINE_ORDER), 2):
        row = []
        for j in range(2):
            idx = i + j
            if idx < len(PIPELINE_ORDER):
                name = PIPELINE_ORDER[idx]
                status_icon = Style.apply("✓", Style.SUCCESS, Style.BOLD) if completed.get(name, False) else Style.apply("○", Style.MUTED)
                name_str = Style.apply(name, Style.BOLD)
                desc = PIPELINES[name].description
                row.append(f"{status_icon} {name_str}: {desc}")
            else:
                row.append("")
        rows.append(row)
    
    for row in rows:
        left = row[0].ljust(col_width)
        right = row[1] if len(row) > 1 else ""
        print(f"  {left}  {right}")
    
    print()

def show_execution_plan(pipelines, skipped):
    Console.header("Execution Plan")
    
    flow = ""
    for i, pipeline in enumerate(pipelines):
        if i > 0:
            flow += f" {Style.apply('→', Style.BLUE)} "
        flow += Style.apply(pipeline, Style.CYAN, Style.BOLD)
    
    print(f"  {flow}")
    
    if skipped:
        print(f"\n{Style.apply('Skipped dependencies:', Style.BOLD)}")
        for dep in skipped:
            print(f"  {Style.apply('↷', Style.MUTED)} {Style.apply(dep, Style.MUTED)} (using existing output)")
    
    print()

def run_pipeline(pipeline_name, config):
    module_path = f"src.pipelines.{pipeline_name}"
    
    try:
        with spinner_context(f"Loading pipeline module {pipeline_name}"):
            module = importlib.import_module(module_path)
            pipeline_class = getattr(module, PIPELINES[pipeline_name].class_name)
        
        Console.header(f"Running {pipeline_name.upper()} Pipeline")
        print(f"{Style.apply('⚙', Style.CYAN)} {PIPELINES[pipeline_name].description}")
        
        start_time = time.time()
        pipeline = pipeline_class(config)
        
        print(f"\n{Style.apply('▶', Style.BOLD)} Starting pipeline execution...")
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
        
        Console.error(f"Pipeline error: {e}")
        return PipelineResult(
            name=pipeline_name,
            status=PipelineStatus.FAILED,
            duration=duration,
            error=str(e)
        )

def print_summary(results):
    Console.header("Execution Summary")
    
    total_time = sum(result.duration for result in results)
    success_count = sum(1 for result in results if result.succeeded)
    
    max_name_length = max(len(result.name) for result in results)
    
    print(f"{Style.apply('Pipeline'.ljust(max_name_length + 2), Style.BOLD)}"
          f"{Style.apply('Status'.ljust(20), Style.BOLD)}"
          f"{Style.apply('Duration', Style.BOLD)}")
    print(f"{Style.apply('━' * (max_name_length + 35), Style.CYAN)}")
    
    for result in results:
        name = result.name.ljust(max_name_length + 2)
        
        if result.status == PipelineStatus.SUCCESS:
            status = f"{Style.apply('✓ SUCCESS', Style.SUCCESS, Style.BOLD)}".ljust(20)
        else:
            status = f"{Style.apply('✗ FAILED', Style.ERROR, Style.BOLD)}".ljust(20)
            
        duration = f"{result.duration:.2f}s"
        print(f"{Style.apply(name, Style.BOLD)}{status}{duration}")
    
    print(f"{Style.apply('━' * (max_name_length + 35), Style.CYAN)}")
    print(f"Total: {len(results)} | "
          f"Successful: {Style.apply(str(success_count), Style.SUCCESS, Style.BOLD)} | "
          f"Failed: {Style.apply(str(len(results) - success_count), Style.ERROR, Style.BOLD)}")
    print(f"Total execution time: {Style.apply(f'{total_time:.2f}s', Style.YELLOW, Style.BOLD)}")
    
    if success_count == len(results):
        Console.success("\nAll pipelines completed successfully!")
    else:
        failed = [result.name for result in results if not result.succeeded]
        Console.error(f"\nSome pipelines failed: {', '.join(failed)}")
        Console.info("Check logs for more details.")

def main():
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        Console.banner("CHESS AI PIPELINE MANAGER", f"Date: {now}")
        
        project_dir = setup_environment()
        
        hardware = check_hardware()
        device_type = "TPU" if hardware["tpu"] else "GPU" if hardware["gpu"] else "CPU"
        Console.info(f"Running on: {Style.apply(device_type, Style.BOLD, Style.YELLOW)}")
        
        if device_type == "GPU" and hardware.get("gpu_name"):
            Console.info(f"GPU: {hardware['gpu_name']}")
        
        if hardware.get("drive_space_gb"):
            if hardware["drive_space_gb"] < 5:
                Console.warning(f"Low disk space: {hardware['drive_space_gb']:.1f}GB free")
            else:
                Console.info(f"Drive space: {hardware['drive_space_gb']:.1f}GB free")
        
        Console.step(1, "Select running mode")
        mode_options = {
            "test": {
                "name": "Test Mode",
                "description": "Faster with smaller datasets, good for testing",
                "icon": "🔬"
            },
            "prod": {
                "name": "Production Mode",
                "description": "Full training with large datasets, longer runtime",
                "icon": "🚀"
            }
        }
        mode = Console.select(mode_options, "Select running mode", "test")
        
        if mode == "prod":
            Console.warning("Production mode will use full datasets and longer training times.")
            Console.info("Make sure you have enough resources and time for completion.")
        
        config_path = os.path.join(project_dir, 'config.yml')
        config = load_configuration(config_path, mode)
        
        Console.step(2, "Select workflow")
        workflow_id = Console.select(WORKFLOWS, "Select workflow", "full")
        selected_pipelines = WORKFLOWS[workflow_id]["pipelines"]
        
        Console.info(f"Selected: {Style.apply(WORKFLOWS[workflow_id]['name'], Style.BOLD)}")
        
        display_pipelines(selected_pipelines)
        
        completed = detect_completed_pipelines()
        ordered_pipelines, skipped_deps = resolve_dependencies(selected_pipelines, completed)
        
        check_dependencies(ordered_pipelines)
        
        show_execution_plan(ordered_pipelines, skipped_deps)
        
        Console.step(3, "Execute pipelines")
        
        if not Console.confirm("Proceed with execution?"):
            Console.warning("Operation cancelled by user.")
            return 0
        
        results = []
        progress = ProgressBar(len(ordered_pipelines), 
                            prefix=f"{Style.apply('Overall Progress:', Style.BOLD)}", 
                            size=30)
        
        execution_start = time.time()
        
        for i, pipeline in enumerate(ordered_pipelines, 1):
            pipeline_start = time.time()
            print(f"\n{Style.apply('═' * 60, Style.CYAN)}")
            print(f"{Style.apply(f'Pipeline {i}/{len(ordered_pipelines)}:', Style.BOLD)} "
                  f"{Style.apply(pipeline.upper(), Style.CYAN, Style.BOLD)}")
            print(f"{Style.apply('═' * 60, Style.CYAN)}")
            
            result = run_pipeline(pipeline, config)
            results.append(result)
            
            status_symbol = Style.SYMBOL_SUCCESS if result.succeeded else Style.SYMBOL_ERROR
            status_style = Style.SUCCESS if result.succeeded else Style.ERROR
            
            pipeline_time = time.time() - pipeline_start
            elapsed_total = time.time() - execution_start
            
            print(f"\n{Style.apply(status_symbol, status_style, Style.BOLD)} "
                  f"{pipeline} completed in {pipeline_time:.2f}s")
            print(f"{Style.apply(f'Total elapsed time: {elapsed_total:.2f}s', Style.MUTED)}")
            
            if not result.succeeded:
                dependents = [p for p in ordered_pipelines[i:] 
                             if pipeline in PIPELINES[p].dependencies]
                if dependents:
                    Console.warning(f"This failure may affect later pipelines: {', '.join(dependents)}")
                    if not Console.confirm("Continue with execution?"):
                        Console.error("Execution aborted by user after pipeline failure.")
                        break
            
            progress.update(1)
        
        print_summary(results)
        
        if all(result.succeeded for result in results):
            if "eval" in [r.name for r in results if r.succeeded]:
                Console.info("\nCheck the evaluation results in the Google Drive folder: /content/drive/MyDrive/chess_ai/evaluation")
        
        return 0 if all(result.succeeded for result in results) else 1
        
    except KeyboardInterrupt:
        Console.warning("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        Console.error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())