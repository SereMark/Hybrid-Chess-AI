#!/usr/bin/env python3

import importlib
import time
import sys
import os
import yaml
import subprocess
import shutil
from typing import List, Optional
from datetime import datetime
import threading
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

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
            parts.append(f"with High RAM toggle")
            
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
    'hyperopt': ['chess','optuna', 'numpy', 'pandas', 'wandb'],
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
    'hyperopt': PipelineConfig(
        name='hyperopt', 
        description="Find optimal hyperparameters for the model",
        dependencies=['data'],
        class_name='HyperoptPipeline',
        requirements=PIPELINE_REQUIREMENTS['hyperopt'],
        hardware=HardwareRequirement(
            type=HardwareType.GPU,
            high_ram=True,
            gpu_type="A100",
            description="GPU-accelerated hyperparameter search with high memory usage"
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
    SYMBOL_CPU = "🖥️"
    SYMBOL_GPU = "🔋"
    SYMBOL_RAM = "🧠"
    
    @staticmethod
    def apply(text, *styles):
        result = ""
        for style in styles:
            result += style
        result += str(text) + Style.RESET
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
    
    @staticmethod
    def hardware_style(hw_type):
        if hw_type == HardwareType.GPU:
            return Style.PURPLE, Style.SYMBOL_GPU
        elif hw_type == HardwareType.CPU:
            return Style.BLUE, Style.SYMBOL_CPU
        return Style.TEAL, Style.SYMBOL_GEAR

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
        
        content_lines = content.split('\n')
        max_content_width = max([len(line.replace('\033[', '').split('m')[-1]) for line in content_lines]) if content_lines else 0
        width = max(width, max_content_width + 6)
        
        title_display = f" {title} " if title else ""
        title_len = len(title_display.replace('\033[', '').split('m')[-1]) if '\033[' in title_display else len(title_display)
        top_border = "┌" + "─" * 2 + title_display + "─" * (width - title_len - 3) + "┐"
        bottom_border = "└" + "─" * (width - 2) + "┘"
        
        print(f"{Style.apply(top_border, style)}")
        
        for line in content_lines:
            visible_length = len(line.replace('\033[', '').split('m')[-1]) if '\033[' in line else len(line)
            padding = max(0, width - visible_length - 2)
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
            
            print("\033[" + str(len(options) + 5) + "A\033[J")
        
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
    
    @staticmethod
    def display_hardware_req(hardware_req, current_hardware=None):
        hw_style, hw_symbol = Style.hardware_style(hardware_req.type)
        
        main_text = f"{Style.apply(hw_symbol, hw_style)} {Style.apply('Recommended:', Style.BOLD)} {Style.apply(str(hardware_req), hw_style)}"
        
        if hardware_req.description:
            main_text += f" - {Style.apply(hardware_req.description, Style.MUTED)}"
            
        print(main_text)
        
        if current_hardware:
            hardware_match = True
            warning_message = None
            
            if hardware_req.type == HardwareType.GPU and not current_hardware.get("gpu", False):
                hardware_match = False
                warning_message = "Pipeline requires GPU acceleration, but only CPU detected"
            elif hardware_req.type == HardwareType.CPU and current_hardware.get("cpu_cores", 0) < hardware_req.min_cores:
                hardware_match = False
                warning_message = f"Pipeline works best with {hardware_req.min_cores}+ CPU cores, but only {current_hardware.get('cpu_cores', 0)} detected"
            elif hardware_req.high_ram and not current_hardware.get("high_ram_enabled", False):
                hardware_match = False
                warning_message = "Pipeline requires High RAM toggle in Colab, but it appears to be disabled"
                
            if not hardware_match and warning_message:
                UI.warning(warning_message, indent=4)
                UI.note("Performance will be significantly impacted", indent=4)
                
                if hardware_req.high_ram and not current_hardware.get("high_ram_enabled", False):
                    UI.info("To enable High RAM in Google Colab: Runtime → Change runtime type → Hardware accelerator → High-RAM", indent=4)
                
                if hardware_req.type == HardwareType.CPU:
                    UI.info("For data processing, select: Runtime → Change runtime type → Hardware accelerator → None", indent=4)
                elif hardware_req.type == HardwareType.GPU:
                    UI.info("For model training, select: Runtime → Change runtime type → Hardware accelerator → GPU → A100", indent=4)

class Spinner:
    def __init__(self, message="Processing", style=Style.CYAN):
        self.message = message
        self.style = style
        self.running = False
        self.spinner_thread = None
        self.frames = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        self.delay = 0.1
        self.last_frame_length = 0
        
    def spin(self):
        i = 0
        while self.running:
            frame = self.frames[i % len(self.frames)]
            timestamp = Style.apply(f"[{datetime.now().strftime('%H:%M:%S')}]", Style.MUTED)
            
            output = f"{timestamp} {Style.apply(frame, self.style)} {self.message}"
            
            if self.last_frame_length > 0:
                sys.stdout.write("\r" + " " * self.last_frame_length + "\r")
            
            sys.stdout.write(f"\r{output}")
            sys.stdout.flush()
            
            self.last_frame_length = len(output.replace('\033[', '').split('m')[-1])
            
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
        sys.stdout.write("\r" + " " * (self.last_frame_length) + "\r")
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
        self.last_line_length = 0

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
        
        visible_length = len(progress_text.replace('\033[', '').split('m')[-1])
        if self.last_line_length > visible_length:
            sys.stdout.write("\r" + " " * self.last_line_length + "\r")
        
        self.last_line_length = visible_length
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
    device_type = "GPU" if hardware["gpu"] else "CPU"
    device_color = Style.LIME if hardware["gpu"] else Style.YELLOW
    
    content = []
    
    device_info = f"{Style.apply(Style.SYMBOL_ZAP, Style.ORANGE)} Running on: {Style.apply(device_type, device_color, Style.BOLD)}"
    if device_type == "GPU" and hardware.get("gpu_name"):
        device_info += f" - {Style.apply(hardware['gpu_name'], Style.CYAN)}"
        if hardware.get("gpu_memory"):
            device_info += f" ({hardware['gpu_memory']:.1f}GB)"
    content.append(device_info)
    
    ram_status = hardware.get("high_ram_enabled", False)
    ram_color = Style.LIME if ram_status else Style.YELLOW
    ram_text = "ENABLED" if ram_status else "DISABLED"
    content.append(f"{Style.apply('⚡', Style.PURPLE)} High RAM Toggle: {Style.apply(ram_text, ram_color, Style.BOLD)}")
    
    if hardware.get("ram_gb"):
        ram_gb = hardware.get('ram_gb', 0)
        content.append(f"{Style.apply('🧠', Style.PURPLE)} Total RAM: {Style.apply(f'{ram_gb:.1f}GB', Style.PURPLE, Style.BOLD)}")
    
    cpu_cores = hardware.get("cpu_cores", 0)
    if cpu_cores:
        content.append(f"{Style.apply('⚙', Style.BLUE)} CPU Cores: {Style.apply(str(cpu_cores), Style.BLUE, Style.BOLD)} (important for data processing)")
    
    drive_space = hardware.get("drive_space_gb")
    if drive_space:
        space_color = Style.RED if drive_space < 5 else Style.GREEN
        content.append(f"{Style.apply('💾', space_color)} Drive Space: {Style.apply(f'{drive_space:.1f}GB free', space_color, Style.BOLD)}")
        
        if drive_space < 5:
            content.append(f"{Style.apply('⚠', Style.WARNING, Style.BOLD)} {Style.apply('Low disk space warning!', Style.WARNING)}")
    
    content.append(f"\n{Style.apply('💡', Style.GOLD)} {Style.apply('Hardware Setup Guide:', Style.GOLD, Style.BOLD)}")
    content.append(f"{Style.apply('1.', Style.CYAN)} Use {Style.apply('CPU-only', Style.CYAN, Style.BOLD)} for data processing (saves GPU credits)")
    content.append(f"{Style.apply('2.', Style.PURPLE)} Use {Style.apply('GPU (A100)', Style.PURPLE, Style.BOLD)} for model training pipelines")
    content.append(f"{Style.apply('3.', Style.LIME)} Enable {Style.apply('High-RAM', Style.LIME, Style.BOLD)} toggle for all pipelines in production mode")
    
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

    if "eval" in pipelines:
        UI.header("Stockfish Check")
        if not shutil.which("stockfish"):
            UI.info("Stockfish not found on PATH, installing via apt...")
            try:
                with spinner_context("Updating apt cache", Style.CYAN):
                    subprocess.run(
                        ["apt-get", "update"],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                with spinner_context("Installing Stockfish", Style.CYAN):
                    subprocess.run(
                        ["apt-get", "install", "-y", "stockfish"],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                if shutil.which("stockfish"):
                    UI.success("Stockfish installed and available on PATH")
                else:
                    UI.error("Stockfish installation completed but binary not found on PATH")
            except subprocess.CalledProcessError as e:
                UI.error(f"Failed to install Stockfish: {e}")
        else:
            UI.success("Stockfish binary already present")

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

def display_pipelines(completed_pipelines=None, hardware_info=None):
    UI.header("Pipeline Selection")
    
    if completed_pipelines is None:
        completed_pipelines = detect_completed_pipelines()
    
    max_name_len = max(len(p) for p in PIPELINE_ORDER)
    
    UI.info("Available pipelines:")
    print()
    
    for i, name in enumerate(PIPELINE_ORDER, 1):
        pipeline_config = PIPELINES[name]
        is_completed = completed_pipelines.get(name, False)
        
        status_icon = Style.apply(Style.SYMBOL_CHECKED, Style.SUCCESS, Style.BOLD) if is_completed else Style.apply(Style.SYMBOL_CIRCLE, Style.MUTED)
        name_str = Style.apply(name.ljust(max_name_len), Style.BOLD)
        desc = pipeline_config.description
        depends = ", ".join(pipeline_config.dependencies) if pipeline_config.dependencies else "None"
        deps_display = f" {Style.apply('(Deps:', Style.MUTED)} {Style.apply(depends, Style.CYAN)}{Style.apply(')', Style.MUTED)}"
        
        hw_style, hw_symbol = Style.hardware_style(pipeline_config.hardware.type)
        hardware_icon = Style.apply(hw_symbol, hw_style)
        
        print(f"  {Style.apply(str(i), Style.CYAN, Style.BOLD)} {status_icon} {name_str}: {desc}{deps_display} {hardware_icon}")
        
        if hardware_info:
            hw_match = True
            if (pipeline_config.hardware.type == HardwareType.GPU and not hardware_info.get("gpu", False)) or \
               (pipeline_config.hardware.type == HardwareType.CPU and hardware_info.get("cpu_cores", 0) < pipeline_config.hardware.min_cores) or \
               (pipeline_config.hardware.high_ram and not hardware_info.get("high_ram_enabled", False)):
                hw_match = False
                
            hw_rec = str(pipeline_config.hardware)
            if hw_match:
                print(f"     {Style.apply('Recommended:', Style.MUTED)} {Style.apply(hw_rec, Style.MUTED)}")
            else:
                print(f"     {Style.apply(Style.SYMBOL_WARNING, Style.WARNING)} {Style.apply('Required:', Style.MUTED)} {Style.apply(hw_rec, Style.WARNING)}")
                
                if name == "data" and hardware_info.get("gpu", False):
                    print(f"     {Style.apply(Style.SYMBOL_INFO, Style.INFO)} {Style.apply('TIP: Switch to CPU-only runtime to save GPU credits for this pipeline', Style.CYAN)}")
    
    print()

def show_execution_plan(pipelines, skipped, hardware_info=None):
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
    
    if hardware_info:
        UI.subheader("Hardware Compatibility Check")
        
        UI.info("Hardware recommendations for each pipeline:")
        
        data_hw_needed = False
        gpu_hw_needed = False
        high_ram_needed = False
        
        for pipeline in pipelines:
            hw_req = PIPELINES[pipeline].hardware
            print(f"  {Style.apply(pipeline, Style.CYAN, Style.BOLD)}:")
            UI.display_hardware_req(hw_req, hardware_info)
            
            if hw_req.type == HardwareType.CPU:
                data_hw_needed = True
            if hw_req.type == HardwareType.GPU:
                gpu_hw_needed = True
            if hw_req.high_ram:
                high_ram_needed = True
        
        print()
        UI.panel("Hardware Switch Guide", 
                f"{Style.apply('For optimal resource usage in Google Colab:', Style.BOLD)}\n\n"
                f"{Style.apply('1.', Style.CYAN)} {Style.apply('Data pipeline:', Style.CYAN)} Use CPU runtime (save GPU credits)\n"
                f"{Style.apply('2.', Style.PURPLE)} {Style.apply('Training pipelines:', Style.PURPLE)} Use GPU (A100) runtime\n"
                f"{Style.apply('3.', Style.GOLD)} {Style.apply('All pipelines in prod mode:', Style.GOLD)} Enable High-RAM\n\n"
                f"{Style.apply('To change runtime:', Style.BOLD)} Runtime → Change runtime type", 
                style=Style.TEAL)
    
    print()

def run_pipeline(pipeline_name, config, hardware_info, progress=None):
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
        
        hw_req = PIPELINES[pipeline_name].hardware
        hw_style, hw_symbol = Style.hardware_style(hw_req.type)
        hw_text = f"{Style.apply(hw_symbol, hw_style)} Hardware: {Style.apply(str(hw_req), hw_style)}"
        desc_lines.append(hw_text)
        
        hardware_match = True
        if hw_req.type == HardwareType.GPU and not hardware_info.get("gpu", False):
            hardware_match = False
            desc_lines.append(f"{Style.apply(Style.SYMBOL_WARNING, Style.WARNING)} Alert: GPU required but running on CPU")
            desc_lines.append(f"{Style.apply(Style.SYMBOL_INFO, Style.INFO)} Change to: Runtime → Change runtime type → Hardware accelerator → GPU → A100")
        elif hw_req.type == HardwareType.CPU and hardware_info.get("gpu", True):
            if pipeline_name == "data":
                desc_lines.append(f"{Style.apply(Style.SYMBOL_INFO, Style.INFO)} Note: Data processing runs efficiently on CPU (saves GPU credits)")
                desc_lines.append(f"{Style.apply(Style.SYMBOL_INFO, Style.INFO)} Consider: Runtime → Change runtime type → Hardware accelerator → None")
        
        if hw_req.high_ram and not hardware_info.get("high_ram_enabled", False):
            hardware_match = False
            desc_lines.append(f"{Style.apply(Style.SYMBOL_WARNING, Style.WARNING)} Alert: High RAM toggle required but appears to be disabled")
            desc_lines.append(f"{Style.apply(Style.SYMBOL_INFO, Style.INFO)} Enable: Runtime → Change runtime type → Hardware accelerator → High-RAM")
            
        for line in desc_lines:
            visible_length = len(line.replace('\033[', '').split('m')[-1]) if '\033[' in line else len(line)
            padding = max(0, width - visible_length - 4)
            print(f"{Style.apply('║', Style.CYAN)} {line}{' ' * padding}{Style.apply('║', Style.CYAN)}")
            
        print(f"{Style.apply('╚' + '═' * (width - 2) + '╝', Style.CYAN)}")
        
        start_time = time.time()
        pipeline = pipeline_class(config)
        
        print(f"\n{Style.apply(Style.SYMBOL_PLAY, Style.CYAN, Style.BOLD)} {Style.apply('Starting pipeline execution...', Style.CYAN)}")
        
        if not hardware_match:
            print(f"{Style.apply(Style.SYMBOL_WARNING, Style.WARNING)} {Style.apply('Performance will be significantly impacted by hardware limitations', Style.WARNING)}")
        elif pipeline_name == "data" and hardware_info.get("gpu", False):
            print(f"{Style.apply(Style.SYMBOL_INFO, Style.INFO)} {Style.apply('Note: Data pipeline is running on GPU but would be more efficient on CPU to save GPU credits', Style.CYAN)}")
            
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

def print_summary(results, hardware_info=None):
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
    
    if hardware_info:
        device_type = "GPU" if hardware_info.get("gpu", False) else "CPU"
        device_info = f"{Style.apply(Style.SYMBOL_CPU if device_type == 'CPU' else Style.SYMBOL_GPU, Style.ORANGE)} " \
                     f"Hardware: {Style.apply(device_type, Style.ORANGE, Style.BOLD)}"
        if device_type == "GPU" and hardware_info.get("gpu_name"):
            device_info += f" ({Style.apply(hardware_info.get('gpu_name'), Style.CYAN)})"
        stats_content += f"\n{device_info}"
    
    UI.panel("Stats", stats_content, style=Style.TEAL)
    
    if all(result.succeeded for result in results):
        UI.success("\nAll pipelines completed successfully!")
    else:
        failed = [result.name for result in results if not result.succeeded]
        UI.error(f"\nSome pipelines failed: {', '.join(Style.apply(p, Style.BOLD) for p in failed)}")
        UI.info("Check logs for more details")
    
    hardware_tips = (
        f"{Style.apply('Hardware Optimization Tips:', Style.GOLD, Style.BOLD)}\n\n"
        f"{Style.apply('1.', Style.CYAN)} {Style.apply('For data processing:', Style.BOLD)} Use CPU runtime to save GPU credits\n"
        f"{Style.apply('2.', Style.PURPLE)} {Style.apply('For model training:', Style.BOLD)} Use GPU (A100) runtime for maximum performance\n"
        f"{Style.apply('3.', Style.LIME)} {Style.apply('For production workloads:', Style.BOLD)} Always enable High-RAM toggle\n"
        f"{Style.apply('4.', Style.ORANGE)} {Style.apply('To maximize Colab usage:', Style.BOLD)} Switch hardware between pipelines as needed"
    )
    
    UI.panel("Hardware Tips", hardware_tips, style=Style.CYAN)

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
        display_pipelines(completed, hardware)
        
        UI.info(f"Previously completed: {', '.join(p for p, v in completed.items() if v) or 'None'}")
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
        
        show_execution_plan(ordered_pipelines, skipped_deps, hardware)
        
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
            
            result = run_pipeline(pipeline, config, hardware, overall_progress)
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
            
            if i < len(ordered_pipelines):
                next_pipeline = ordered_pipelines[i]
                next_hw_req = PIPELINES[next_pipeline].hardware
                
                if pipeline == "data" and next_hw_req.type == HardwareType.GPU and not hardware.get("gpu", False):
                    UI.panel("Hardware Switch Recommended", 
                        f"{Style.apply(Style.SYMBOL_WARNING, Style.WARNING)} Next pipeline '{Style.apply(next_pipeline, Style.BOLD)}' requires GPU but you're currently on CPU.\n\n"
                        f"{Style.apply(Style.SYMBOL_INFO, Style.INFO)} Recommended action: Runtime → Change runtime type → Hardware accelerator → A100",
                        style=Style.ORANGE)
                elif next_hw_req.type == HardwareType.CPU and hardware.get("gpu", False):
                    UI.panel("Hardware Optimization Tip", 
                        f"{Style.apply(Style.SYMBOL_INFO, Style.INFO)} Next pipeline '{Style.apply(next_pipeline, Style.BOLD)}' runs efficiently on CPU.\n\n"
                        f"{Style.apply(Style.SYMBOL_LIGHT, Style.CYAN)} To save GPU credits: Runtime → Change runtime type → Hardware accelerator → CPU",
                        style=Style.CYAN)
        
        print_summary(results, hardware)
        
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