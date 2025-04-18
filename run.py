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
from google.colab import drive

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
        dependencies=['supervised', 'data'],
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
        dependencies=['supervised', 'data'],
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
        if not os.path.exists('/content/drive/MyDrive'):
            Console.info("Mounting Google Drive...")
            drive.mount('/content/drive')
        
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
        
    def _load(self) -> Dict:
        if not os.path.exists(self.path):
            Console.warning(f"Config file not found: {self.path}")
            return {}
        
        with open(self.path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
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

def load_configuration(config_path: str, mode: str) -> Configuration:
    with spinner_context(f"Loading configuration ({mode} mode)"):
        config = Configuration(config_path, mode)
    
    Console.success(f"Configuration loaded: {config_path}")
    return config

def resolve_dependencies(selected_pipelines: List[str]) -> List[str]:
    with spinner_context("Resolving pipeline dependencies"):
        required = set(selected_pipelines)
        
        added = True
        while added:
            added = False
            for pipeline in list(required):
                if pipeline in PIPELINES:
                    for dependency in PIPELINES[pipeline].dependencies:
                        if dependency not in required:
                            required.add(dependency)
                            added = True
        
        ordered = [p for p in PIPELINE_ORDER if p in required]
    
    added_deps = [p for p in ordered if p not in selected_pipelines]
    if added_deps:
        Console.info("Added required dependencies:")
        for dep in added_deps:
            needed_by = [p for p in selected_pipelines if dep in PIPELINES[p].dependencies]
            print(f"  {Style.CYAN}{Style.SYMBOL_BULLET}{Style.RESET} {dep} "
                  f"{Style.MUTED}(needed by: {', '.join(needed_by)}){Style.RESET}")
    
    return ordered

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

def show_execution_plan(pipelines: List[str]) -> None:
    Console.section("Pipeline Execution Plan")
    
    print(f"{Style.BOLD}Execution order:{Style.RESET}")
    
    flow = ""
    for i, pipeline in enumerate(pipelines):
        if i > 0:
            flow += f" {Style.BLUE}{Style.SYMBOL_ARROW}{Style.RESET} "
        flow += f"{Style.CYAN}{pipeline}{Style.RESET}"
    
    print(f"  {flow}")
    print()
    
    print(f"{Style.BOLD}Pipeline details:{Style.RESET}")
    for i, pipeline in enumerate(pipelines, 1):
        dependencies = PIPELINES[pipeline].dependencies
        deps_str = f"{Style.MUTED}(depends on: {', '.join(dependencies)}){Style.RESET}" if dependencies else ""
        print(f"  {Style.CYAN}{i}.{Style.RESET} {Style.BOLD}{pipeline}{Style.RESET} - "
              f"{PIPELINES[pipeline].description} {deps_str}")
    
def select_pipelines() -> List[str]:
    pipeline_options = [(name, config.description) for name, config in PIPELINES.items()]
    
    options_with_descriptions = [
        ("all", "Run all pipelines (full training workflow)"),
        ("training", "Run data, supervised, and reinforcement pipelines"),
        ("evaluation", "Run data, eval, and benchmark pipelines"),
        ("custom", "Select specific pipelines to run")
    ]
    
    choice = Console.menu("Select pipeline workflow", options_with_descriptions, "all")
    
    if choice == "all":
        return list(PIPELINES.keys())
    
    if choice == "training":
        return ["data", "supervised", "reinforcement"]
    
    if choice == "evaluation":
        return ["data", "eval", "benchmark"]
    
    if choice == "custom":
        return Console.multi_select("Select pipelines to run", pipeline_options, ["data"])
    
    return []

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
        mode = select_mode()
        
        config_path = os.path.join(project_dir, 'config.yml')
        config = load_configuration(config_path, mode)
        
        selected_pipelines = select_pipelines() 
        
        if not selected_pipelines:
            Console.error("No pipelines selected. Exiting.")
            return 0
        
        ordered_pipelines = resolve_dependencies(selected_pipelines)
        
        check_and_install_dependencies(ordered_pipelines)
        
        show_execution_plan(ordered_pipelines)
        
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
            
            progress.update(1)
        
        print_pipeline_summary(results)
        
        if all(result.succeeded for result in results):
            Console.success("\nAll pipelines executed successfully!")
            Console.info("The Chess AI model is now ready for use.")
            
            if "benchmark" in [r.name for r in results if r.succeeded]:
                Console.info("Check the benchmark results in the Google Drive folder.")
        
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