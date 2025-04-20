#!/usr/bin/env python3

import importlib
import time
import sys
import os
import yaml
import subprocess
import shutil
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
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

CHESS_LOGO = r"""
   ██████╗██╗  ██╗███████╗███████╗███████╗     █████╗ ██╗
  ██╔════╝██║  ██║██╔════╝██╔════╝██╔════╝    ██╔══██╗██║
  ██║     ███████║█████╗  ███████╗███████╗    ███████║██║
  ██║     ██╔══██║██╔══╝  ╚════██║╚════██║    ██╔══██║██║
  ╚██████╗██║  ██║███████╗███████║███████║    ██║  ██║██║
   ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝    ╚═╝  ╚═╝╚═╝
"""

class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    HIDDEN = "\033[8m"
    
    BLACK = "\033[30m"
    RED = "\033[38;5;203m"
    GREEN = "\033[38;5;107m"
    YELLOW = "\033[38;5;221m"
    BLUE = "\033[38;5;75m"
    MAGENTA = "\033[38;5;176m"
    CYAN = "\033[38;5;116m"
    WHITE = "\033[38;5;255m"
    GRAY = "\033[38;5;246m"
    ORANGE = "\033[38;5;209m"
    PURPLE = "\033[38;5;141m"
    LIME = "\033[38;5;119m"
    PINK = "\033[38;5;213m"
    TEAL = "\033[38;5;80m"
    GOLD = "\033[38;5;220m"
    
    BG_BLACK = "\033[40m"
    BG_RED = "\033[48;5;203m"
    BG_GREEN = "\033[48;5;107m"
    BG_YELLOW = "\033[48;5;221m"
    BG_BLUE = "\033[48;5;75m"
    BG_MAGENTA = "\033[48;5;176m"
    BG_CYAN = "\033[48;5;116m"
    BG_WHITE = "\033[48;5;255m"
    BG_GRAY = "\033[48;5;240m"
    BG_ORANGE = "\033[48;5;209m"
    BG_PURPLE = "\033[48;5;141m"
    
    SUCCESS = GREEN
    ERROR = RED
    WARNING = YELLOW
    INFO = BLUE
    MUTED = GRAY
    HIGHLIGHT = CYAN
    ACCENT = ORANGE
    
    SYMBOL_SUCCESS = "✓"
    SYMBOL_ERROR = "✗"
    SYMBOL_WARNING = "⚠"
    SYMBOL_INFO = "ℹ"
    SYMBOL_ARROW = "→"
    SYMBOL_BULLET = "•"
    SYMBOL_PLAY = "▶"
    SYMBOL_PAUSE = "⏸"
    SYMBOL_STAR = "★"
    SYMBOL_GEAR = "⚙"
    SYMBOL_CIRCLE = "○"
    SYMBOL_CHECKED = "✅"
    SYMBOL_SKIP = "↷"
    SYMBOL_CLOCK = "🕒"
    SYMBOL_HOURGLASS = "⏳"
    SYMBOL_ROCKET = "🚀"
    SYMBOL_LIGHT = "💡"
    SYMBOL_DISK = "💾"
    SYMBOL_FLAG = "🏁"
    SYMBOL_ZAP = "⚡"
    
    @staticmethod
    def apply(text, *styles):
        result = ""
        for style in styles:
            result += style
        result += text + Style.RESET
        return result
    
    @staticmethod
    def status_style(status):
        if status == PipelineStatus.SUCCESS:
            return Style.SUCCESS, Style.SYMBOL_SUCCESS
        elif status == PipelineStatus.FAILED:
            return Style.ERROR, Style.SYMBOL_ERROR
        elif status == PipelineStatus.RUNNING:
            return Style.BLUE, Style.SYMBOL_PLAY
        elif status == PipelineStatus.PENDING:
            return Style.MUTED, Style.SYMBOL_CIRCLE
        elif status == PipelineStatus.SKIPPED:
            return Style.MUTED, Style.SYMBOL_SKIP
        return Style.MUTED, Style.SYMBOL_CIRCLE

class UI:
    @staticmethod
    def clear():
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def get_terminal_size():
        try:
            columns, rows = shutil.get_terminal_size()
            return max(columns, 80), max(rows, 24)
        except:
            return 80, 24
    
    @staticmethod
    def logo():
        term_width, _ = UI.get_terminal_size()
        UI.clear()
        
        for line in CHESS_LOGO.split('\n'):
            padding = (term_width - len(line)) // 2
            if line.strip():
                print(" " * padding + Style.apply(line, Style.BLUE, Style.BOLD))
            else:
                print()
    
    @staticmethod
    def banner(title, subtitle=None):
        term_width, _ = UI.get_terminal_size()
        width = min(term_width - 4, 100)
        
        UI.clear()
        print(f"\n{Style.apply('╭' + '─' * (width - 2) + '╮', Style.BLUE)}")
        print(f"{Style.apply('│', Style.BLUE)} {Style.apply(title.center(width-4), Style.BLUE, Style.BOLD)} {Style.apply('│', Style.BLUE)}")
        if subtitle:
            print(f"{Style.apply('│', Style.BLUE)} {Style.apply(subtitle.center(width-4), Style.TEAL)} {Style.apply('│', Style.BLUE)}")
        print(f"{Style.apply('╰' + '─' * (width - 2) + '╯', Style.BLUE)}\n")
    
    @staticmethod
    def header(title, width=None):
        if width is None:
            term_width, _ = UI.get_terminal_size()
            width = min(term_width - 4, 100)
        
        print(f"\n{Style.apply('┏━', Style.BLUE, Style.BOLD)}{Style.apply('┫', Style.CYAN, Style.BOLD)} "
              f"{Style.apply(title, Style.BLUE, Style.BOLD)} "
              f"{Style.apply('┣', Style.CYAN, Style.BOLD)}{Style.apply('━' * (width - len(title) - 10), Style.BLUE, Style.BOLD)}")
    
    @staticmethod
    def subheader(title):
        print(f"\n{Style.apply('┌─', Style.TEAL)} {Style.apply(title, Style.TEAL, Style.BOLD)}")
    
    @staticmethod
    def divider(width=None):
        if width is None:
            term_width, _ = UI.get_terminal_size()
            width = min(term_width - 4, 100)
        print(f"{Style.apply('┄' * width, Style.GRAY)}")
    
    @staticmethod
    def footer(width=None):
        if width is None:
            term_width, _ = UI.get_terminal_size()
            width = min(term_width - 4, 100)
        print(f"\n{Style.apply('┗' + '━' * (width - 2) + '┛', Style.BLUE)}")
    
    @staticmethod
    def panel(title, content, width=None, style=Style.BLUE):
        if width is None:
            term_width, _ = UI.get_terminal_size()
            width = min(term_width - 4, 100)
        
        title_display = f" {title} " if title else ""
        top_border = "┌" + "─" * 2 + title_display + "─" * (width - len(title_display) - 3) + "┐"
        bottom_border = "└" + "─" * (width - 2) + "┘"
        
        print(f"{Style.apply(top_border, style)}")
        
        for line in content.split('\n'):
            padding = width - len(line) - 2
            print(f"{Style.apply('│', style)} {line}{' ' * padding}{Style.apply('│', style)}")
        
        print(f"{Style.apply(bottom_border, style)}")
    
    @staticmethod
    def info(message, indent=0):
        print(f"{' ' * indent}{Style.apply(Style.SYMBOL_INFO, Style.INFO)} {message}")
    
    @staticmethod
    def success(message, indent=0):
        print(f"{' ' * indent}{Style.apply(Style.SYMBOL_SUCCESS, Style.SUCCESS, Style.BOLD)} {message}")
    
    @staticmethod
    def error(message, indent=0):
        print(f"{' ' * indent}{Style.apply(Style.SYMBOL_ERROR, Style.ERROR, Style.BOLD)} {message}")
    
    @staticmethod
    def warning(message, indent=0):
        print(f"{' ' * indent}{Style.apply(Style.SYMBOL_WARNING, Style.WARNING, Style.BOLD)} {message}")
    
    @staticmethod
    def action(message, indent=0):
        print(f"{' ' * indent}{Style.apply(Style.SYMBOL_ARROW, Style.CYAN, Style.BOLD)} {Style.apply(message, Style.CYAN)}")
    
    @staticmethod
    def step(step_num, message, total_steps=None):
        if total_steps:
            step_display = f"Step {step_num}/{total_steps}"
        else:
            step_display = f"Step {step_num}"
            
        print(f"\n{Style.apply('╭─', Style.BLUE, Style.BOLD)} "
              f"{Style.apply(step_display, Style.LIME, Style.BOLD)}: "
              f"{Style.apply(message, Style.BLUE, Style.BOLD)}")
        print(f"{Style.apply('╰' + '─' * (len(step_display) + len(message) + 5), Style.BLUE)}")
    
    @staticmethod
    def feature(message, indent=2):
        print(f"{' ' * indent}{Style.apply(Style.SYMBOL_STAR, Style.ACCENT)} {message}")
    
    @staticmethod
    def note(message, indent=2):
        print(f"{' ' * indent}{Style.apply('Note:', Style.ITALIC, Style.CYAN)} {message}")
    
    @staticmethod
    def input(prompt, default=None):
        if default:
            prompt_text = f"{prompt} [{default}]: "
        else:
            prompt_text = f"{prompt}: "
        
        sys.stdout.write(f"{Style.apply('→', Style.GREEN)} {Style.apply(prompt_text, Style.BOLD)}")
        sys.stdout.flush()
        response = input().strip()
        
        if not response and default:
            return default
        return response
    
    @staticmethod
    def confirm(message, default="y"):
        response = UI.input(f"{message} (y/n)", default)
        return response.lower() in ("y", "yes")
    
    @staticmethod
    def multi_select(options, prompt, defaults=None, min_selections=0):
        if defaults is None:
            defaults = []
            
        UI.info(prompt)
        print(f"\n{Style.apply('Use the following commands:', Style.MUTED)}")
        print(f"  {Style.apply('number', Style.BOLD)}: Toggle selection")
        print(f"  {Style.apply('a', Style.BOLD)}: Select all")
        print(f"  {Style.apply('n', Style.BOLD)}: Select none")
        print(f"  {Style.apply('done/enter', Style.BOLD)}: Confirm selection\n")
        
        selected = set(defaults)
        valid_indices = list(range(1, len(options) + 1))
        
        while True:
            for i, name in enumerate(options, 1):
                is_selected = i in selected
                status_symbol = Style.SYMBOL_CHECKED if is_selected else Style.SYMBOL_CIRCLE
                status_color = Style.SUCCESS if is_selected else Style.MUTED
                name_style = Style.BOLD if is_selected else ""
                print(f"  {Style.apply(str(i), Style.CYAN, Style.BOLD)} "
                      f"{Style.apply(status_symbol, status_color)} "
                      f"{Style.apply(name, name_style)}")
            
            selection = UI.input("\nSelect option (or 'done' to confirm)", "done" if selected else "")
            
            if selection.lower() in ("", "done"):
                if len(selected) >= min_selections:
                    break
                else:
                    UI.warning(f"Please select at least {min_selections} option(s)")
                    continue
            elif selection.lower() == "a":
                selected = set(valid_indices)
            elif selection.lower() == "n":
                selected = set()
            else:
                try:
                    idx = int(selection)
                    if idx in valid_indices:
                        if idx in selected:
                            selected.remove(idx)
                        else:
                            selected.add(idx)
                    else:
                        UI.error(f"Invalid option: {idx}")
                except ValueError:
                    UI.error(f"Invalid input: {selection}")
            
            print("\033[F" * (len(options) + 1))
        
        return [options[i-1] for i in sorted(selected)]
    
    @staticmethod
    def select(options, prompt, default=None):
        UI.info(prompt)
        for key, data in options.items():
            icon = data.get("icon", "")
            name = data.get("name", key)
            desc = data.get("description", "")
            print(f"  {icon} {Style.apply(name, Style.BOLD)}: {desc}")
        
        selected = UI.input("Select option", default)
        return selected if selected in options else default
    
    @staticmethod
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

class Spinner:
    def __init__(self, message="Processing", style=Style.CYAN):
        self.message = message
        self.style = style
        self.running = False
        self.spinner_thread = None
        self.frames = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        self.delay = 0.1
        
    def spin(self):
        i = 0
        while self.running:
            frame = self.frames[i % len(self.frames)]
            timestamp = Style.apply(f"[{datetime.now().strftime('%H:%M:%S')}]", Style.MUTED)
            sys.stdout.write(f"\r{timestamp} {Style.apply(frame, self.style)} {self.message}")
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
        sys.stdout.write("\r" + " " * (len(self.message) + 30) + "\r")
        sys.stdout.flush()
    
    def update_message(self, message):
        self.message = message

class ProgressBar:
    def __init__(self, total, prefix="", size=30, style=Style.CYAN):
        self.total = max(1, total)
        self.prefix = prefix
        self.size = size
        self.style = style
        self.start_time = time.time()
        self.count = 0
        self.last_update_time = 0
        self.min_update_interval = 0.1

    def update(self, count=1, message=None):
        current_time = time.time()
        self.count += count
        
        if current_time - self.last_update_time < self.min_update_interval and self.count < self.total:
            return
            
        self.last_update_time = current_time
        self.render(message)
        
        if self.count >= self.total:
            print()
    
    def render(self, message=None):
        filled_length = int(self.size * min(self.count, self.total) / self.total)
        bar = f"{Style.apply('█' * filled_length, Style.GREEN)}{Style.apply('░' * (self.size - filled_length), Style.GRAY)}"
        elapsed_time = time.time() - self.start_time
        percent = 100 * min(self.count, self.total) / self.total
        
        if self.count > 0:
            eta = elapsed_time * (self.total - self.count) / self.count
            eta_str = UI.time_format(eta)
        else:
            eta_str = "calculating..."
        
        elapsed_str = UI.time_format(elapsed_time)
        
        message_str = f" | {message}" if message else ""
        
        progress_text = (f"\r{self.prefix} |{bar}| {percent:.1f}% ({self.count}/{self.total}) "
                        f"[{elapsed_str} elapsed, {eta_str} remaining]{message_str}")
        
        sys.stdout.write(progress_text)
        sys.stdout.flush()

@contextmanager
def spinner_context(message, style=Style.CYAN):
    spinner = Spinner(message, style)
    spinner.start()
    try:
        yield spinner
    finally:
        spinner.stop()

def setup_environment():
    with spinner_context("Setting up environment", Style.BLUE):
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
                        print(f"{Style.apply('⚠', Style.WARNING)} Detected loading issue, trying with CPU mapping...")
                        return original_load(f, map_location='cpu', pickle_module=pickle_module, **kwargs)
                        
                torch.load = safe_torch_load
                
            except Exception as e:
                print(f"Failed to install CUDA patch: {e}")
        except Exception as e:
            print(f"Error in setup: {e}")

    UI.success(f"Environment ready at: {project_dir}")
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
            import psutil
            hardware_info["ram_gb"] = psutil.virtual_memory().total / (1024**3)
            hardware_info["drive_space_gb"] = psutil.disk_usage('/content/drive').free / (1024**3) if os.path.exists('/content/drive') else None
        except:
            pass
    except:
        pass
        
    return hardware_info

def display_hardware_info(hardware):
    device_type = "GPU" if hardware["gpu"] else "CPU"
    device_color = Style.LIME if hardware["gpu"] else Style.YELLOW
    
    content = []
    
    device_info = f"{Style.apply(Style.SYMBOL_ZAP, Style.ORANGE)} Running on: {Style.apply(device_type, device_color, Style.BOLD)}"
    if device_type == "GPU" and hardware.get("gpu_name"):
        device_info += f" - {Style.apply(hardware['gpu_name'], Style.CYAN)}"
        if hardware.get("gpu_memory"):
            device_info += f" ({hardware['gpu_memory']:.1f}GB)"
    content.append(device_info)
    
    cpu_cores = hardware.get("cpu_cores", 0)
    if cpu_cores:
        content.append(f"{Style.apply('⚙', Style.BLUE)} CPU Cores: {Style.apply(str(cpu_cores), Style.BLUE, Style.BOLD)}")
    
    ram_gb = hardware.get("ram_gb")
    if ram_gb:
        content.append(f"{Style.apply('⚡', Style.PURPLE)} RAM: {Style.apply(f'{ram_gb:.1f}GB', Style.PURPLE, Style.BOLD)}")
    
    drive_space = hardware.get("drive_space_gb")
    if drive_space:
        space_color = Style.RED if drive_space < 5 else Style.GREEN
        content.append(f"{Style.apply('💾', space_color)} Drive Space: {Style.apply(f'{drive_space:.1f}GB free', space_color, Style.BOLD)}")
        
        if drive_space < 5:
            content.append(f"{Style.apply('⚠', Style.WARNING, Style.BOLD)} {Style.apply('Low disk space warning!', Style.WARNING)}")
    
    UI.panel("System Resources", "\n".join(content), style=Style.TEAL)

def check_dependencies(pipelines):
    UI.header("Dependency Check")
    
    all_requirements = set()
    for pipeline in pipelines:
        all_requirements.update(PIPELINES[pipeline].requirements)
    
    missing_packages = []
    with spinner_context("Scanning installed packages", Style.BLUE):
        for package in all_requirements:
            try:
                importlib.import_module(package.replace('-', '_'))
                UI.info(f"Package '{Style.apply(package, Style.BOLD)}' is installed", indent=2)
            except ImportError:
                missing_packages.append(package)
    
    if missing_packages:
        UI.warning(f"Found {len(missing_packages)} missing dependencies: {', '.join(Style.apply(p, Style.BOLD) for p in missing_packages)}")
        if UI.confirm("Install missing dependencies now?"):
            progress = ProgressBar(len(missing_packages), 
                                  prefix=f"{Style.apply('Installing packages:', Style.BOLD)}", 
                                  size=30,
                                  style=Style.BLUE)
            
            for i, package in enumerate(missing_packages):
                try:
                    with spinner_context(f"Installing {Style.apply(package, Style.BOLD)}", Style.CYAN):
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", package],
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                    UI.success(f"Installed {Style.apply(package, Style.BOLD)}")
                except subprocess.CalledProcessError as e:
                    UI.error(f"Failed to install {Style.apply(package, Style.BOLD)}: {e.stderr}")
                
                progress.update(1)
    else:
        UI.success("All required dependencies are already installed")

class Configuration:
    def __init__(self, config_path, mode="test"):
        self.path = config_path
        self.mode = mode
        self.data = self._load()
    
    def _load(self):
        if not os.path.exists(self.path):
            UI.warning(f"Config file not found: {self.path}")
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
    with spinner_context(f"Loading {Style.apply(mode, Style.BOLD)} configuration", Style.CYAN):
        config = Configuration(config_path, mode)
    
    UI.success(f"Configuration loaded: {Style.apply(config_path, Style.BOLD)} ({Style.apply(mode, Style.CYAN)} mode)")
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

def display_pipelines(completed_pipelines=None):
    UI.header("Pipeline Selection")
    
    if completed_pipelines is None:
        completed_pipelines = detect_completed_pipelines()
    
    max_name_len = max(len(p) for p in PIPELINE_ORDER)
    
    UI.info("Available pipelines:")
    print()
    
    for i, name in enumerate(PIPELINE_ORDER, 1):
        is_completed = completed_pipelines.get(name, False)
        status_icon = Style.apply(Style.SYMBOL_CHECKED, Style.SUCCESS, Style.BOLD) if is_completed else Style.apply(Style.SYMBOL_CIRCLE, Style.MUTED)
        name_str = Style.apply(name.ljust(max_name_len), Style.BOLD)
        desc = PIPELINES[name].description
        depends = ", ".join(PIPELINES[name].dependencies) if PIPELINES[name].dependencies else "None"
        deps_display = f" {Style.apply('(Deps:', Style.MUTED)} {Style.apply(depends, Style.CYAN)}{Style.apply(')', Style.MUTED)}"
        
        print(f"  {Style.apply(str(i), Style.CYAN, Style.BOLD)} {status_icon} {name_str}: {desc}{deps_display}")
    
    print()

def show_execution_plan(pipelines, skipped):
    UI.header("Execution Plan")
    
    flow = ""
    for i, pipeline in enumerate(pipelines):
        if i > 0:
            flow += f" {Style.apply('→', Style.BLUE)} "
        flow += Style.apply(pipeline, Style.CYAN, Style.BOLD)
    
    print(f"  {Style.apply(Style.SYMBOL_PLAY, Style.GREEN)} Execution flow: {flow}")
    
    if skipped:
        UI.subheader("Skipped Dependencies")
        for dep in skipped:
            print(f"  {Style.apply(Style.SYMBOL_SKIP, Style.MUTED)} {Style.apply(dep, Style.MUTED)} (using existing output)")
    
    print()

def run_pipeline(pipeline_name, config, progress=None):
    module_path = f"src.pipelines.{pipeline_name}"
    
    try:
        with spinner_context(f"Loading pipeline module {Style.apply(pipeline_name, Style.BOLD)}", Style.TEAL):
            module = importlib.import_module(module_path)
            pipeline_class = getattr(module, PIPELINES[pipeline_name].class_name)
        
        width, _ = UI.get_terminal_size()
        width = min(width - 4, 100)
        
        print(f"\n{Style.apply('╔' + '═' * (width - 2) + '╗', Style.CYAN, Style.BOLD)}")
        print(f"{Style.apply('║', Style.CYAN, Style.BOLD)} "
              f"{Style.apply(pipeline_name.upper(), Style.BLUE, Style.BOLD)} PIPELINE"
              f"{' ' * (width - len(pipeline_name) - 12)}"
              f"{Style.apply('║', Style.CYAN, Style.BOLD)}")
        print(f"{Style.apply('╠' + '═' * (width - 2) + '╣', Style.CYAN)}")
        
        desc_lines = []
        desc = PIPELINES[pipeline_name].description
        deps = ", ".join(PIPELINES[pipeline_name].dependencies) if PIPELINES[pipeline_name].dependencies else "None"
        
        desc_lines.append(f"{Style.apply(Style.SYMBOL_GEAR, Style.GOLD)} Description: {desc}")
        desc_lines.append(f"{Style.apply(Style.SYMBOL_ARROW, Style.LIME)} Dependencies: {deps}")
        
        for line in desc_lines:
            padding = width - len(line.replace(Style.RESET, "")) - 10
            print(f"{Style.apply('║', Style.CYAN)} {line}{' ' * padding}{Style.apply('║', Style.CYAN)}")
            
        print(f"{Style.apply('╚' + '═' * (width - 2) + '╝', Style.CYAN)}")
        
        start_time = time.time()
        pipeline = pipeline_class(config)
        
        print(f"\n{Style.apply(Style.SYMBOL_PLAY, Style.CYAN, Style.BOLD)} {Style.apply('Starting pipeline execution...', Style.CYAN)}")
        
        if progress:
            progress.update(message=f"Running {Style.apply(pipeline_name, Style.BOLD)}")
            
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
        
        UI.error(f"Pipeline error: {e}")
        UI.panel("Error Details", str(e), style=Style.ERROR)
        
        return PipelineResult(
            name=pipeline_name,
            status=PipelineStatus.FAILED,
            duration=duration,
            error=str(e)
        )

def print_summary(results):
    UI.header("Execution Summary")
    
    term_width, _ = UI.get_terminal_size()
    width = min(term_width - 4, 100)
    
    total_time = sum(result.duration for result in results)
    success_count = sum(1 for result in results if result.succeeded)
    
    max_name_length = max(len(result.name) for result in results)
    table_width = width - 4
    
    pipeline_width = max(max_name_length + 2, 15)
    status_width = 18
    duration_width = table_width - pipeline_width - status_width - 6
    
    print(f"\n{Style.apply('┌' + '─' * (table_width - 2) + '┐', Style.BLUE)}")
    
    header = (f"{Style.apply('│', Style.BLUE)} "
              f"{Style.apply('Pipeline'.center(pipeline_width), Style.BOLD)} "
              f"{Style.apply('│', Style.BLUE)} "
              f"{Style.apply('Status'.center(status_width), Style.BOLD)} "
              f"{Style.apply('│', Style.BLUE)} "
              f"{Style.apply('Duration'.center(duration_width), Style.BOLD)} "
              f"{Style.apply('│', Style.BLUE)}")
    print(header)
    
    print(f"{Style.apply('├' + '─' * (pipeline_width) + '┼' + '─' * (status_width) + '┼' + '─' * (duration_width) + '┤', Style.BLUE)}")
    
    for result in results:
        name = result.name.ljust(pipeline_width - 1)
        
        status_style, status_symbol = Style.status_style(result.status) 
        if result.status == PipelineStatus.SUCCESS:
            status_text = "SUCCESS"
        else:
            status_text = "FAILED"
            
        status = f"{status_symbol} {status_text}".ljust(status_width - 1)
        
        duration_text = UI.time_format(result.duration).ljust(duration_width - 1)
        
        row = (f"{Style.apply('│', Style.BLUE)} "
               f"{Style.apply(name, Style.BOLD)} "
               f"{Style.apply('│', Style.BLUE)} "
               f"{Style.apply(status, status_style, Style.BOLD)} "
               f"{Style.apply('│', Style.BLUE)} "
               f"{duration_text} "
               f"{Style.apply('│', Style.BLUE)}")
        print(row)
    
    print(f"{Style.apply('└' + '─' * (pipeline_width) + '┴' + '─' * (status_width) + '┴' + '─' * (duration_width) + '┘', Style.BLUE)}")
    
    stats_content = (
        f"{Style.apply(Style.SYMBOL_FLAG, Style.TEAL)} Total pipelines: {Style.apply(str(len(results)), Style.BOLD)}\n"
        f"{Style.apply(Style.SYMBOL_SUCCESS, Style.SUCCESS)} Successful: {Style.apply(str(success_count), Style.SUCCESS, Style.BOLD)}\n"
        f"{Style.apply(Style.SYMBOL_ERROR, Style.ERROR)} Failed: {Style.apply(str(len(results) - success_count), Style.ERROR, Style.BOLD)}\n"
        f"{Style.apply(Style.SYMBOL_CLOCK, Style.GOLD)} Total execution time: {Style.apply(UI.time_format(total_time), Style.GOLD, Style.BOLD)}"
    )
    
    UI.panel("Stats", stats_content, style=Style.TEAL)
    
    if success_count == len(results):
        UI.success("\nAll pipelines completed successfully!")
    else:
        failed = [result.name for result in results if not result.succeeded]
        UI.error(f"\nSome pipelines failed: {', '.join(Style.apply(p, Style.BOLD) for p in failed)}")
        UI.info("Check logs for more details")

def main():
    try:
        UI.logo()
        time.sleep(1)
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        UI.banner("CHESS AI PIPELINE MANAGER", f"Date: {now}")
        
        project_dir = setup_environment()
        
        hardware = check_hardware()
        display_hardware_info(hardware)
        
        total_steps = 3
        UI.step(1, "Select Running Mode", total_steps)
        
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
        mode = UI.select(mode_options, "Select running mode", "test")
        
        if mode == "prod":
            UI.panel("Production Mode", (
                f"{Style.apply(Style.SYMBOL_WARNING, Style.WARNING)} "
                f"{Style.apply('Production mode will use full datasets and longer training times.', Style.WARNING)}\n"
                f"{Style.apply(Style.SYMBOL_INFO, Style.INFO)} "
                f"Make sure you have enough resources and time for completion."
            ), style=Style.ORANGE)
        
        config_path = os.path.join(project_dir, 'config.yml')
        config = load_configuration(config_path, mode)
        
        completed = detect_completed_pipelines()
        display_pipelines(completed)
        
        UI.info(f"Previously completed: {', '.join(p for p, v in completed.items() if v)}")
        pipeline_names = PIPELINE_ORDER.copy()
        
        pipeline_input = UI.input("Select pipelines to run (comma-separated numbers, or press Enter for all)")
        
        if not pipeline_input:
            selected_pipeline_names = pipeline_names
            UI.success(f"Selected all pipelines: {', '.join(Style.apply(p, Style.BOLD) for p in selected_pipeline_names)}")
        else:
            try:
                selected_indices = [int(idx.strip()) for idx in pipeline_input.split(',')]
                selected_pipeline_names = []
                
                for idx in selected_indices:
                    if 1 <= idx <= len(pipeline_names):
                        selected_pipeline_names.append(pipeline_names[idx-1])
                    else:
                        UI.error(f"Invalid pipeline number: {idx}. Ignoring.")
                
                if not selected_pipeline_names:
                    UI.error("No valid pipelines selected. Exiting.")
                    return 1
                    
                UI.success(f"Selected pipelines: {', '.join(Style.apply(p, Style.BOLD) for p in selected_pipeline_names)}")
            except ValueError:
                UI.error("Invalid input format. Please enter numbers separated by commas.")
                return 1
        
        ordered_pipelines, skipped_deps = resolve_dependencies(selected_pipeline_names, completed)
        
        UI.step(2, "Verify Dependencies", total_steps)
        check_dependencies(ordered_pipelines)
        
        show_execution_plan(ordered_pipelines, skipped_deps)
        
        UI.step(3, "Execute Pipelines", total_steps)
        
        if not UI.confirm("Proceed with execution?"):
            UI.warning("Operation cancelled by user.")
            return 0
        
        results = []
        overall_progress = ProgressBar(
            len(ordered_pipelines), 
            prefix=f"{Style.apply('Overall Progress:', Style.BLUE, Style.BOLD)}", 
            size=30,
            style=Style.LIME
        )
        
        execution_start = time.time()
        
        for i, pipeline in enumerate(ordered_pipelines, 1):
            pipeline_start = time.time()
            execution_line = [
                Style.apply('┏', Style.ORANGE, Style.BOLD),
                Style.apply('━' * 20, Style.ORANGE),
                Style.apply('┫', Style.ORANGE, Style.BOLD),
                Style.apply(f' PIPELINE {i}/{len(ordered_pipelines)}: ', Style.WHITE, Style.BOLD),
                Style.apply(pipeline.upper(), Style.ORANGE, Style.BOLD),
                Style.apply('┣', Style.ORANGE, Style.BOLD),
                Style.apply('━' * 20, Style.ORANGE),
                Style.apply('┓', Style.ORANGE, Style.BOLD)
            ]
            print("\n" + "".join(execution_line))
            
            result = run_pipeline(pipeline, config, overall_progress)
            results.append(result)
            
            status_style, status_symbol = Style.status_style(result.status)
            pipeline_time = time.time() - pipeline_start
            elapsed_total = time.time() - execution_start
            remaining_pipelines = len(ordered_pipelines) - (i)
            estimated_remaining = elapsed_total / i * remaining_pipelines if i > 0 else 0
            
            completion_info = [
                f"\n{Style.apply(status_symbol, status_style, Style.BOLD)} ",
                f"{Style.apply(pipeline, Style.BOLD)} completed in ",
                f"{Style.apply(UI.time_format(pipeline_time), status_style, Style.BOLD)}"
            ]
            print("".join(completion_info))
            
            time_info = [
                Style.apply(Style.SYMBOL_CLOCK, Style.GOLD), " ",
                Style.apply(f"Elapsed: {UI.time_format(elapsed_total)}", Style.GOLD), " | ",
                Style.apply(f"Estimated remaining: {UI.time_format(estimated_remaining)}", Style.GOLD)
            ]
            print("".join(time_info))
            
            if not result.succeeded:
                dependents = [p for p in ordered_pipelines[i:] 
                              if pipeline in PIPELINES[p].dependencies]
                if dependents:
                    UI.warning(f"This failure may affect later pipelines: {', '.join(Style.apply(d, Style.BOLD) for d in dependents)}")
                    if not UI.confirm("Continue with execution?"):
                        UI.error("Execution aborted by user after pipeline failure.")
                        break
            
            overall_progress.update(1)
        
        print_summary(results)
        
        if all(result.succeeded for result in results):
            if "eval" in [r.name for r in results if r.succeeded]:
                UI.panel("Evaluation Results", 
                    f"{Style.apply(Style.SYMBOL_DISK, Style.CYAN)} "
                    f"Check the evaluation results in the Google Drive folder:\n"
                    f"{Style.apply('/content/drive/MyDrive/chess_ai/evaluation', Style.PURPLE, Style.BOLD)}",
                    style=Style.CYAN)
        
        return 0 if all(result.succeeded for result in results) else 1
        
    except KeyboardInterrupt:
        UI.warning("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        UI.error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())