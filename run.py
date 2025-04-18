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
    'eval': ['chess', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'wandb'],
    'benchmark': ['chess', 'numpy', 'pandas', 'wandb']
}

PIPELINES = {
    'data': PipelineConfig(
        name='data', 
        description="Processes raw PGN files into efficient training dataset",
        dependencies=[],
        class_name='DataPipeline',
        requirements=PIPELINE_REQUIREMENTS['data']
    ),
    'hyperopt': PipelineConfig(
        name='hyperopt', 
        description="Finds optimal hyperparameters for the model",
        dependencies=['data'],
        class_name='HyperoptPipeline',
        requirements=PIPELINE_REQUIREMENTS['hyperopt']
    ),
    'supervised': PipelineConfig(
        name='supervised', 
        description="Trains the model on human games using supervised learning",
        dependencies=['data'],
        class_name='SupervisedPipeline',
        requirements=PIPELINE_REQUIREMENTS['supervised']
    ),
    'reinforcement': PipelineConfig(
        name='reinforcement', 
        description="Improves model through self-play reinforcement learning",
        dependencies=['supervised'],
        class_name='ReinforcementPipeline',
        requirements=PIPELINE_REQUIREMENTS['reinforcement']
    ),
    'eval': PipelineConfig(
        name='eval', 
        description="Evaluates model performance against benchmarks",
        dependencies=['data'],
        class_name='EvalPipeline',
        requirements=PIPELINE_REQUIREMENTS['eval']
    ),
    'benchmark': PipelineConfig(
        name='benchmark', 
        description="Compares model performance against other engines",
        dependencies=['supervised'],
        class_name='BenchmarkPipeline',
        requirements=PIPELINE_REQUIREMENTS['benchmark']
    )
}

PIPELINE_ORDER = ['data', 'hyperopt', 'supervised', 'reinforcement', 'eval', 'benchmark']

class Style:
    RESET = "\033[0m"
    
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    BLACK = "\033[30m"
    RED = "\033[38;5;167m"
    GREEN = "\033[38;5;107m"
    YELLOW = "\033[38;5;221m"
    BLUE = "\033[38;5;110m"
    MAGENTA = "\033[38;5;176m"
    CYAN = "\033[38;5;116m"
    WHITE = "\033[38;5;255m"
    GRAY = "\033[38;5;246m"
    
    BG_BLACK = "\033[40m"
    BG_RED = "\033[48;5;167m"
    BG_GREEN = "\033[48;5;107m"
    BG_YELLOW = "\033[48;5;221m"
    BG_BLUE = "\033[48;5;25m"
    BG_MAGENTA = "\033[48;5;176m"
    BG_CYAN = "\033[48;5;116m"
    BG_WHITE = "\033[48;5;255m"
    
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

class Console:
    @staticmethod
    def clear() -> None:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def header(title: str, subtitle: str = None) -> None:
        width = 80
        padding = " " * 2
        
        print(f"{Style.BG_BLUE}{Style.WHITE}{Style.BOLD}{' ' * width}{Style.RESET}")
        print(f"{Style.BG_BLUE}{Style.WHITE}{Style.BOLD}{title.center(width)}{Style.RESET}")
        if subtitle:
            print(f"{Style.BG_BLUE}{Style.WHITE}{padding}{subtitle}{' ' * (width - len(subtitle) - len(padding))}{Style.RESET}")
        print(f"{Style.BG_BLUE}{Style.WHITE}{Style.BOLD}{' ' * width}{Style.RESET}")
    
    @staticmethod
    def section(title: str) -> None:
        print(f"\n{Style.BOLD}{Style.BLUE}{title}{Style.RESET}")
        print(f"{Style.CYAN}{'─' * 60}{Style.RESET}")
    
    @staticmethod
    def info(message: str) -> None:
        print(f"{Style.INFO}{Style.SYMBOL_INFO} {message}{Style.RESET}")
    
    @staticmethod
    def success(message: str) -> None:
        print(f"{Style.SUCCESS}{Style.SYMBOL_SUCCESS} {message}{Style.RESET}")
    
    @staticmethod
    def error(message: str) -> None:
        print(f"{Style.ERROR}{Style.SYMBOL_ERROR} {message}{Style.RESET}")
    
    @staticmethod
    def warning(message: str) -> None:
        print(f"{Style.WARNING}{Style.SYMBOL_WARNING} {message}{Style.RESET}")
    
    @staticmethod
    def step(message: str) -> None:
        print(f"{Style.CYAN}{Style.SYMBOL_ARROW} {Style.BOLD}{message}{Style.RESET}")
    
    @staticmethod
    def feature(message: str) -> None:
        print(f"  {Style.CYAN}{Style.SYMBOL_STAR} {message}{Style.RESET}")
    
    @staticmethod
    def input(prompt: str, default: str = None) -> str:
        if default:
            prompt_text = f"{prompt} [{default}]: "
        else:
            prompt_text = f"{prompt}: "
        
        sys.stdout.write(f"{Style.GREEN}⮞ {Style.BOLD}{prompt_text}{Style.RESET}")
        sys.stdout.flush()
        response = input().strip()
        
        if not response and default:
            return default
        return response
    
    @staticmethod
    def menu(title: str, options: List[Tuple[str, str]], default: str = None) -> str:
        Console.section(title)
        
        for i, (key, description) in enumerate(options, 1):
            default_mark = f" {Style.YELLOW}(default){Style.RESET}" if key == default else ""
            print(f"  {Style.CYAN}{i}{Style.RESET}. {Style.BOLD}{key}{Style.RESET} - {description}{default_mark}")
        
        while True:
            choice = Console.input("Enter the choice (number or name)", default)
            
            try:
                if choice.isdigit() and 1 <= int(choice) <= len(options):
                    return options[int(choice) - 1][0]
                
                for key, _ in options:
                    if choice.lower() == key.lower():
                        return key
                
                Console.error(f"Invalid choice. Please select a number between 1-{len(options)} or enter a valid name.")
            except Exception:
                Console.error("Invalid choice. Please try again.")
    
    @staticmethod
    def multi_select(title: str, options: List[Tuple[str, str]], default: List[str] = None) -> List[str]:
        Console.section(title)
        
        if default is None:
            default = []
        
        print(f"{Style.MUTED}(Enter numbers separated by comma, 'all' for all options, or 'none' for none){Style.RESET}")
        for i, (key, description) in enumerate(options, 1):
            default_mark = f" {Style.YELLOW}*{Style.RESET}" if key in default else ""
            print(f"  {Style.CYAN}{i}{Style.RESET}. {Style.BOLD}{key}{Style.RESET} - {description}{default_mark}")
        
        while True:
            choice = Console.input("Enter the choices", "all" if default == ["all"] else ",".join(default) if default else None)
            
            if choice.lower() == "all":
                return [key for key, _ in options]
            
            if choice.lower() == "none":
                return []
            
            try:
                selected = []
                for part in choice.split(","):
                    part = part.strip()
                    if part.isdigit() and 1 <= int(part) <= len(options):
                        selected.append(options[int(part) - 1][0])
                    else:
                        for key, _ in options:
                            if part.lower() == key.lower():
                                selected.append(key)
                                break
                        else:
                            Console.error(f"Invalid choice: {part}")
                            break
                else:
                    return selected
            except Exception:
                Console.error("Invalid format. Please try again.")

class Spinner:
    def __init__(self, message: str = "Processing"):
        self.message = message
        self.running = False
        self.spinner_thread = None
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.delay = 0.1
        
    def spin(self) -> None:
        i = 0
        while self.running:
            frame = self.frames[i % len(self.frames)]
            sys.stdout.write(f"\r{Style.CYAN}{frame}{Style.RESET} {self.message}")
            sys.stdout.flush()
            time.sleep(self.delay)
            i += 1
            
    def start(self) -> None:
        self.running = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
        
    def stop(self) -> None:
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()

class ProgressBar:
    def __init__(self, total: int, prefix: str = "", size: int = 30):
        self.total = total
        self.prefix = prefix
        self.size = size
        self.start_time = time.time()
        self.count = 0

    def update(self, count: int = 1) -> None:
        self.count += count
        filled_length = int(self.size * self.count / self.total)
        bar = f"{Style.GREEN}{'█' * filled_length}{Style.RESET}{Style.GRAY}{'░' * (self.size - filled_length)}{Style.RESET}"
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
def spinner_context(message: str):
    spinner = Spinner(message)
    spinner.start()
    try:
        yield spinner
    finally:
        spinner.stop()

def setup_colab_environment() -> str:
    with spinner_context("Setting up Colab environment"):
        project_dir = '/content/drive/MyDrive/chess_ai'
        
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
            os.makedirs(os.path.join(project_dir, 'data'), exist_ok=True)
            os.makedirs(os.path.join(project_dir, 'models'), exist_ok=True)
            os.makedirs(os.path.join(project_dir, 'logs'), exist_ok=True)
            
        os.chdir(project_dir)
        
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
        
        if 'src.utils.config' in sys.modules:
            from src.utils.config import Config
            Config._instance = None

    Console.success(f"Environment ready at: {project_dir}")
    return project_dir

def detect_completed_pipelines() -> Dict[str, bool]:
    completed = {}
    
    completed['data'] = os.path.exists('/content/drive/MyDrive/chess_ai/data/train_indices.npy')
    completed['supervised'] = os.path.exists('/content/drive/MyDrive/chess_ai/models/supervised_model.pth')
    completed['reinforcement'] = os.path.exists('/content/drive/MyDrive/chess_ai/models/reinforcement_model.pth')
    completed['hyperopt'] = os.path.exists('/content/drive/MyDrive/chess_ai/todo.json') # TODO
    completed['eval'] = os.path.exists('/content/drive/MyDrive/chess_ai/evaluation/todo.json') # TODO
    completed['benchmark'] = os.path.exists('/content/drive/MyDrive/chess_ai/benchmark/todo.txt') # TODO
    
    return completed

def check_and_install_dependencies(pipelines: List[str]) -> None:
    Console.section("Checking Dependencies")
    
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
        choice = Console.input("Install missing dependencies now?", "y")
        
        if choice.lower() == 'y':
            progress = ProgressBar(len(missing_packages), prefix=f"{Style.BOLD}Installing packages:{Style.RESET}", size=30)
            
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
    def __init__(self, config_path: str, mode: str = "test"):
        self.path = config_path
        self.mode = mode
        self.data = self._load()
        self.overrides = {}
    
    def _load(self) -> Dict:
        if not os.path.exists(self.path):
            Console.warning(f"Config file not found: {self.path}")
            return {}
        
        with open(self.path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        if key in self.overrides:
            return self.overrides[key]
            
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
    
    def _get_nested(self, section: Dict, keys: List[str], default: Any) -> Any:
        current = section
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current
        
    def set_override(self, key: str, value: Any) -> None:
        self.overrides[key] = value

def load_configuration(config_path: str, mode: str) -> Configuration:
    with spinner_context(f"Loading configuration ({mode} mode)"):
        config = Configuration(config_path, mode)
    
    Console.success(f"Configuration loaded: {config_path}")
    return config

def resolve_dependencies(selected_pipelines: List[str], completed: Dict[str, bool], 
                                force_rerun_deps: bool = False) -> Tuple[List[str], List[str]]:
    required = set(selected_pipelines)
    skipped_deps = []
    
    pending_deps = []
    for pipeline in selected_pipelines:
        if pipeline in PIPELINES:
            for dependency in PIPELINES[pipeline].dependencies:
                if dependency not in required:
                    if not force_rerun_deps and completed.get(dependency, False):
                        skipped_deps.append(dependency)
                    else:
                        pending_deps.append(dependency)
                        required.add(dependency)
    
    while pending_deps:
        current_dep = pending_deps.pop(0)
        for sub_dep in PIPELINES[current_dep].dependencies:
            if sub_dep not in required:
                if not force_rerun_deps and completed.get(sub_dep, False):
                    skipped_deps.append(sub_dep)
                else:
                    pending_deps.append(sub_dep)
                    required.add(sub_dep)
    
    ordered = [p for p in PIPELINE_ORDER if p in required]
    return ordered, skipped_deps

def select_pipelines() -> Tuple[List[str], bool]:
    completed = detect_completed_pipelines()
    
    Console.section("Pipeline Status")
    for name in PIPELINE_ORDER:
        status = f"{Style.SUCCESS}{Style.SYMBOL_CHECKED} Completed{Style.RESET}" if completed.get(name, False) else f"{Style.MUTED}{Style.SYMBOL_CIRCLE} Not run{Style.RESET}"
        print(f"  {Style.BOLD}{name.ljust(14)}{Style.RESET} - {status}")
    print()
    
    pipeline_options = []
    for name, config in PIPELINES.items():
        status_icon = f"{Style.SUCCESS}{Style.SYMBOL_CHECKED}{Style.RESET} " if completed.get(name, False) else ""
        pipeline_options.append((name, f"{status_icon}{config.description}"))
    
    preset_options = [
        ("all", "Run all pipelines (full training workflow)"),
        ("training", "Run data, supervised, and reinforcement pipelines"),
        ("evaluation", "Run data, eval, and benchmark pipelines"),
        ("custom", "Select specific pipelines to run"),
        ("continue", "Continue from where you left off (run incomplete pipelines)")
    ]
    
    default_choice = "all"
    if completed.get('data', False) and not completed.get('supervised', False):
        default_choice = "training"
    elif completed.get('supervised', False) and not completed.get('eval', False):
        default_choice = "evaluation"
    elif any(completed.values()) and not all(completed.values()):
        default_choice = "continue"
    
    choice = Console.menu("Select pipeline workflow", preset_options, default_choice)
    
    selected = []
    if choice == "all":
        selected = list(PIPELINES.keys())
    elif choice == "training":
        selected = ["data", "supervised", "reinforcement"]
    elif choice == "evaluation":
        selected = ["data", "eval", "benchmark"]
    elif choice == "continue":
        selected = [name for name, is_completed in completed.items() if not is_completed]
        if not selected:
            Console.success("All pipelines have already been completed!")
            selected = Console.multi_select("Select pipelines to re-run", pipeline_options)
    elif choice == "custom":
        selected = Console.multi_select("Select pipelines to run", pipeline_options)
    
    force_rerun_deps = False
    has_dependencies = any(len(PIPELINES[p].dependencies) > 0 for p in selected if p in PIPELINES)
    
    if has_dependencies:
        resolved, skipped = resolve_dependencies(selected, completed)
        additional_deps = [dep for dep in resolved if dep not in selected]
        
        if additional_deps:
            Console.section("Dependency Management")
            print(f"{Style.INFO}Selected pipelines require these dependencies:{Style.RESET}")
            for dep in additional_deps:
                dep_status = f"{Style.SUCCESS}{Style.SYMBOL_CHECKED} Already completed{Style.RESET}" if completed.get(dep, False) else f"{Style.MUTED}{Style.SYMBOL_CIRCLE} Not yet run{Style.RESET}"
                print(f"  {Style.CYAN}{Style.SYMBOL_BULLET}{Style.RESET} {dep} - {dep_status}")
            
            if any(completed.get(dep, False) for dep in additional_deps):
                options = [
                    ("skip", "Skip already completed dependencies (use existing outputs)"),
                    ("rerun", "Re-run all dependencies (regenerate all data)")
                ]
                dep_choice = Console.menu("How would you like to handle dependencies?", options, "skip")
                force_rerun_deps = dep_choice == "rerun"
    
    return selected, force_rerun_deps

def show_advanced_pipeline_options(pipeline_name: str, config: Configuration) -> Dict:
    Console.section(f"Configure {pipeline_name.upper()} Pipeline Options")
    options = {}
    
    if pipeline_name == "data":
        max_games = Console.input("Maximum games to process", str(config.get('data.max_games', 10000)))
        min_elo = Console.input("Minimum player ELO rating", str(config.get('data.min_elo', 2000)))
        try:
            options['data.max_games'] = int(max_games)
            options['data.min_elo'] = int(min_elo)
        except ValueError:
            Console.warning("Invalid numeric input, using default values")
        
    elif pipeline_name == "supervised":
        epochs = Console.input("Training epochs", str(config.get('supervised.epochs', 10)))
        batch = Console.input("Batch size", str(config.get('data.batch', 128)))
        lr = Console.input("Learning rate", str(config.get('supervised.lr', 0.001)))
        try:
            options['supervised.epochs'] = int(epochs)
            options['data.batch'] = int(batch)
            options['supervised.lr'] = float(lr)
        except ValueError:
            Console.warning("Invalid numeric input, using default values")
        
    elif pipeline_name == "reinforcement":
        iters = Console.input("Self-play iterations", str(config.get('reinforcement.iters', 10)))
        games = Console.input("Games per iteration", str(config.get('reinforcement.games_per_iter', 100)))
        sims = Console.input("MCTS simulations per move", str(config.get('reinforcement.sims_per_move', 100)))
        try:
            options['reinforcement.iters'] = int(iters)
            options['reinforcement.games_per_iter'] = int(games)
            options['reinforcement.sims_per_move'] = int(sims)
        except ValueError:
            Console.warning("Invalid numeric input, using default values")
        
    elif pipeline_name == "benchmark":
        games = Console.input("Number of benchmark games", str(config.get('benchmark.games', 10)))
        elo = Console.input("Stockfish ELO rating", str(config.get('benchmark.stockfish_elo', 1500)))
        try:
            options['benchmark.games'] = int(games)
            options['benchmark.stockfish_elo'] = int(elo)
        except ValueError:
            Console.warning("Invalid numeric input, using default values")
    
    return options

def run_pipeline(pipeline_name: str, config: Configuration) -> PipelineResult:
    Console.section(f"Running {pipeline_name.upper()} Pipeline")
    print(f"{Style.CYAN}{Style.SYMBOL_GEAR}{Style.RESET} {PIPELINES[pipeline_name].description}")
    
    try:
        pipeline_class = load_pipeline(pipeline_name)
        
        start_time = time.time()
        pipeline = pipeline_class(config)
        
        print(f"\n{Style.BOLD}{Style.SYMBOL_PLAY} Starting pipeline execution...{Style.RESET}")
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

def load_pipeline(name: str) -> Any:
    module_path = f"src.pipelines.{name}"
    
    with spinner_context(f"Loading pipeline module: {name}"):
        try:
            module = importlib.import_module(module_path)
            pipeline_class = getattr(module, PIPELINES[name].class_name)
            return pipeline_class
        except Exception as e:
            Console.error(f"Failed to load pipeline {name}: {e}")
            raise

def show_execution_plan(pipelines: List[str], skipped: List[str], config_overrides: Dict[str, Dict] = None):
    Console.section("Pipeline Execution Plan")
    
    print(f"{Style.BOLD}Execution order:{Style.RESET}")
    
    flow = ""
    for i, pipeline in enumerate(pipelines):
        if i > 0:
            flow += f" {Style.BLUE}{Style.SYMBOL_ARROW}{Style.RESET} "
        flow += f"{Style.CYAN}{pipeline}{Style.RESET}"
    
    print(f"  {flow}")
    
    if skipped:
        print(f"\n{Style.BOLD}Skipped dependencies:{Style.RESET}")
        for dep in skipped:
            print(f"  {Style.MUTED}{Style.SYMBOL_SKIP} {dep} (using existing output){Style.RESET}")
    
    print(f"\n{Style.BOLD}Pipeline details:{Style.RESET}")
    for i, pipeline in enumerate(pipelines, 1):
        dependencies = PIPELINES[pipeline].dependencies
        deps_str = f"{Style.MUTED}(depends on: {', '.join(dependencies)}){Style.RESET}" if dependencies else ""
        
        config_str = ""
        if config_overrides and pipeline in config_overrides and config_overrides[pipeline]:
            config_str = f" {Style.YELLOW}[custom config]{Style.RESET}"
            
        print(f"  {Style.CYAN}{i}.{Style.RESET} {Style.BOLD}{pipeline}{Style.RESET}{config_str} - "
              f"{PIPELINES[pipeline].description} {deps_str}")
        
        if config_overrides and pipeline in config_overrides and config_overrides[pipeline]:
            for key, value in config_overrides[pipeline].items():
                print(f"     {Style.MUTED}- {key.split('.')[-1]}: {value}{Style.RESET}")

def monitor_system_resources():
    try:
        import psutil
        import GPUtil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        disk = psutil.disk_usage('/content/drive')
        disk_percent = disk.percent
        disk_free_gb = disk.free / (1024 ** 3)
        
        gpu_info = ""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_info = f" | GPU: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil:.1%})"
        except:
            pass
        
        Console.section("System Resources")
        print(f"CPU: {cpu_percent}% | Memory: {memory_percent}% | Disk: {disk_percent}% ({disk_free_gb:.1f}GB free){gpu_info}")
        
        if memory_percent > 90:
            Console.warning("Memory usage is very high. Pipeline performance may be impacted.")
        if disk_percent > 90:
            Console.warning(f"Disk space is low. Only {disk_free_gb:.1f}GB free on Google Drive.")
        if cpu_percent > 90:
            Console.warning("CPU usage is very high. Pipeline processing may be slower.")
    except:
        pass

def print_pipeline_summary(results: List[PipelineResult]) -> None:
    Console.section("Pipeline Execution Summary")
    
    total_time = sum(result.duration for result in results)
    success_count = sum(1 for result in results if result.succeeded)
    
    max_name_length = max(len(result.name) for result in results)
    
    print(f"{Style.BOLD}{'Pipeline'.ljust(max_name_length + 2)}{'Status'.ljust(20)}{'Duration'.ljust(15)}{Style.RESET}")
    print(f"{Style.CYAN}{'─' * (max_name_length + 35)}{Style.RESET}")
    
    for result in results:
        name = result.name.ljust(max_name_length + 2)
        
        if result.status == PipelineStatus.SUCCESS:
            status = f"{Style.SUCCESS}{Style.SYMBOL_SUCCESS} SUCCESS{Style.RESET}".ljust(20)
        else:
            status = f"{Style.ERROR}{Style.SYMBOL_ERROR} FAILED{Style.RESET}".ljust(20)
            
        duration = f"{result.duration:.2f}s".ljust(15)
        print(f"{Style.BOLD}{name}{Style.RESET}{status}{duration}")
    
    print(f"{Style.CYAN}{'─' * (max_name_length + 35)}{Style.RESET}")
    print(f"Total pipelines: {len(results)} | "
          f"Successful: {Style.SUCCESS}{success_count}{Style.RESET} | "
          f"Failed: {Style.ERROR}{len(results) - success_count}{Style.RESET}")
    print(f"Total execution time: {Style.YELLOW}{total_time:.2f}s{Style.RESET}")
    
    if success_count == len(results):
        Console.success("All pipelines completed successfully!")
    else:
        failed = [result.name for result in results if not result.succeeded]
        Console.error(f"Some pipelines failed: {', '.join(failed)}")
        Console.info("Check logs for more details.")

def select_mode() -> str:
    mode_options = [
        ("test", "Test mode - faster with smaller datasets, good for testing workflows"),
        ("prod", "Production mode - full training with large datasets, longer runtime")
    ]
    
    mode = Console.menu("Select running mode", mode_options, "test")
    
    if mode == "prod":
        Console.warning("Production mode selected. This will use full datasets and longer training times.")
        Console.info("Make sure there is enough available resources and time for completion.")
    
    return mode

def main() -> int:
    try:
        project_dir = setup_colab_environment()
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        Console.clear()
        Console.header("CHESS AI PIPELINE MANAGER", f"Date: {now} | Environment: Google Colab")
        
        Console.feature("Data preparation and processing")
        Console.feature("Hyperparameter optimization")
        Console.feature("Supervised learning from human games")
        Console.feature("Reinforcement learning via self-play")
        Console.feature("Model evaluation and benchmarking")
        
        print()
        monitor_system_resources()
        
        mode = select_mode()
        
        config_path = os.path.join(project_dir, 'config.yml')
        config = load_configuration(config_path, mode)
        
        selected_pipelines, force_rerun = select_pipelines() 
        
        if not selected_pipelines:
            Console.error("No pipelines selected. Exiting.")
            return 0
        
        completed = detect_completed_pipelines()
        ordered_pipelines, skipped_deps = resolve_dependencies(
            selected_pipelines, completed, force_rerun
        )
        
        check_and_install_dependencies(ordered_pipelines)
        
        config_overrides = {}
        customize = Console.input("Would you like to customize pipeline parameters?", "n")
        if customize.lower() == 'y':
            for pipeline in ordered_pipelines:
                customized = show_advanced_pipeline_options(pipeline, config)
                if customized:
                    config_overrides[pipeline] = customized
                    for key, value in customized.items():
                        config.set_override(key, value)
        
        show_execution_plan(ordered_pipelines, skipped_deps, config_overrides)
        
        confirm = Console.input("Proceed with execution?", "y")
        if confirm.lower() != 'y':
            Console.warning("Operation cancelled by user.")
            return 0
        
        results = []
        progress = ProgressBar(len(ordered_pipelines), prefix=f"{Style.BOLD}Overall Progress:{Style.RESET}", size=30)
        
        execution_start = time.time()
        
        for i, pipeline in enumerate(ordered_pipelines, 1):
            pipeline_start = time.time()
            print(f"\n{Style.CYAN}{'═' * 60}{Style.RESET}")
            print(f"{Style.BOLD}Pipeline {i}/{len(ordered_pipelines)}: {Style.CYAN}{pipeline.upper()}{Style.RESET}")
            print(f"{Style.CYAN}{'═' * 60}{Style.RESET}")
            
            result = run_pipeline(pipeline, config)
            results.append(result)
            
            status_symbol = Style.SYMBOL_SUCCESS if result.succeeded else Style.SYMBOL_ERROR
            status_style = Style.SUCCESS if result.succeeded else Style.ERROR
            
            pipeline_time = time.time() - pipeline_start
            elapsed_total = time.time() - execution_start
            
            print(f"\n{status_style}{status_symbol} {pipeline} completed in {pipeline_time:.2f}s{Style.RESET}")
            print(f"{Style.MUTED}Total elapsed time: {elapsed_total:.2f}s{Style.RESET}")
            
            if not result.succeeded:
                dependents = [p for p in ordered_pipelines[i:] 
                             if pipeline in PIPELINES[p].dependencies]
                if dependents:
                    Console.warning(f"This failure may affect later pipelines: {', '.join(dependents)}")
                    choice = Console.input("Continue with execution?", "y")
                    if choice.lower() != 'y':
                        Console.error("Execution aborted by user after pipeline failure.")
                        break
            
            progress.update(1)
        
        print_pipeline_summary(results)
        
        if all(result.succeeded for result in results):
            Console.success("\nAll pipelines executed successfully!")
            Console.info("The Chess AI model is now ready for use.")
            
            if "benchmark" in [r.name for r in results if r.succeeded]:
                Console.info("Check the benchmark results in the Google Drive folder.")
        else:
            failed = [r.name for r in results if not r.succeeded]
            Console.info("You can re-run failed pipelines later with: 'custom' selection option.")
        
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