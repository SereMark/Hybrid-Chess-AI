#!/usr/bin/env python3
import importlib
import time
import sys
import os
from typing import Any, List, Dict
from datetime import datetime
import threading

from src.utils.config import get_config
from src.utils.drive import get_drive

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

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
            sys.stdout.write(f"\r{Colors.CYAN}{frame}{Colors.RESET} {self.message}")
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
    def __init__(self, total, prefix="", size=30, file=sys.stdout):
        self.total = total
        self.prefix = prefix
        self.size = size
        self.file = file
        self.start_time = time.time()
        self.count = 0

    def update(self, count=1):
        self.count += count
        filled_length = int(self.size * self.count / self.total)
        bar = f"{Colors.GREEN}{'█' * filled_length}{Colors.RESET}{'░' * (self.size - filled_length)}"
        elapsed_time = time.time() - self.start_time
        percent = 100 * self.count / self.total
        
        sys.stdout.write(f"\r{self.prefix} |{bar}| {percent:.1f}% Complete ({elapsed_time:.1f}s)")
        sys.stdout.flush()
        
        if self.count >= self.total:
            sys.stdout.write('\n')

PIPELINE_ORDER = ['data', 'hyperopt', 'supervised', 'reinforcement', 'eval', 'benchmark']
PIPELINES = {
    'supervised': 'SupervisedPipeline',
    'reinforcement': 'ReinforcementPipeline',
    'data': 'DataPipeline',
    'eval': 'EvalPipeline',
    'hyperopt': 'HyperoptPipeline',
    'benchmark': 'BenchmarkPipeline'
}

DEPENDENCIES = {
    'supervised': ['data'],
    'reinforcement': ['supervised', 'data'],
    'eval': ['data'],
    'hyperopt': ['data'],
    'benchmark': ['supervised', 'data']
}

PIPELINE_INFO = {
    'data': "Processes raw PGN files into efficient training dataset",
    'hyperopt': "Finds optimal hyperparameters for the model",
    'supervised': "Trains the model on human games using supervised learning",
    'reinforcement': "Improves model through self-play reinforcement learning",
    'eval': "Evaluates model performance against benchmarks",
    'benchmark': "Compares model performance against other engines"
}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    clear_screen()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{Colors.BG_BLUE}{Colors.WHITE}{Colors.BOLD}" + " " * 70 + f"{Colors.RESET}")
    print(f"{Colors.BG_BLUE}{Colors.WHITE}{Colors.BOLD}{'CHESS AI PIPELINE MANAGER':^70}{Colors.RESET}")
    print(f"{Colors.BG_BLUE}{Colors.WHITE}{Colors.BOLD}" + " " * 70 + f"{Colors.RESET}")
    print(f"{Colors.YELLOW}Date:{Colors.RESET} {now} | {Colors.YELLOW}Environment:{Colors.RESET} Google Colab")
    print(f"{Colors.CYAN}{'─' * 70}{Colors.RESET}")

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.CYAN}{'─' * 50}{Colors.RESET}")

def fancy_input(prompt, default=None):
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
        
    sys.stdout.write(f"{Colors.GREEN}⮞ {Colors.BOLD}{prompt}{Colors.RESET}")
    sys.stdout.flush()
    response = input().strip()
    
    if not response and default:
        return default
    return response

def prompt_user(question: str, options: List[str], default=None) -> str:
    print_section(question)
    
    for i, option in enumerate(options, 1):
        default_mark = f" {Colors.YELLOW}(default){Colors.RESET}" if option == default else ""
        desc = PIPELINE_INFO.get(option, "") if question.lower().find("pipeline") >= 0 else ""
        if desc:
            print(f"  {Colors.CYAN}{i}{Colors.RESET}. {Colors.BOLD}{option:<12}{Colors.RESET} {desc}{default_mark}")
        else:
            print(f"  {Colors.CYAN}{i}{Colors.RESET}. {Colors.BOLD}{option}{Colors.RESET}{default_mark}")
    
    while True:
        choice = fancy_input("Enter your choice (number or name)", default)
        
        try:
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return options[int(choice) - 1]
            elif choice in options:
                return choice
            else:
                print(f"{Colors.RED}Invalid choice. Please select a number between 1-{len(options)} or enter the name.{Colors.RESET}")
        except Exception:
            print(f"{Colors.RED}Invalid choice. Please try again.{Colors.RESET}")

def prompt_multiple(question: str, options: List[str], default=None) -> List[str]:
    print_section(question)
    
    print(f"  {Colors.CYAN}0{Colors.RESET}. {Colors.BOLD}All pipelines{Colors.RESET} {Colors.YELLOW}(default){Colors.RESET}")
    
    for i, option in enumerate(options, 1):
        desc = PIPELINE_INFO.get(option, "")
        print(f"  {Colors.CYAN}{i}{Colors.RESET}. {Colors.BOLD}{option:<12}{Colors.RESET} {desc}")
    
    while True:
        choice = fancy_input("Enter your choices (comma-separated numbers or names, 0 for all)", "0")
        
        if choice == "0":
            return options
        
        try:
            choices = []
            for item in choice.split(','):
                item = item.strip()
                if item.isdigit() and 1 <= int(item) <= len(options):
                    choices.append(options[int(item) - 1])
                elif item in options:
                    choices.append(item)
                else:
                    print(f"{Colors.RED}Invalid choice: {item}{Colors.RESET}")
                    break
            else:
                if choices:
                    return choices
                else:
                    print(f"{Colors.RED}Please select at least one pipeline.{Colors.RESET}")
        except Exception:
            print(f"{Colors.RED}Invalid choice. Please try again.{Colors.RESET}")

def resolve_dependencies(selected_pipelines: List[str]) -> List[str]:
    required = set(selected_pipelines)
    
    spinner = Spinner("Resolving pipeline dependencies")
    spinner.start()
    
    added = True
    while added:
        added = False
        for pipeline in list(required):
            if pipeline in DEPENDENCIES:
                for dependency in DEPENDENCIES[pipeline]:
                    if dependency not in required:
                        required.add(dependency)
                        added = True
    
    spinner.stop()
    
    ordered = [p for p in PIPELINE_ORDER if p in required]
    
    added_deps = [p for p in ordered if p not in selected_pipelines]
    if added_deps:
        print(f"\n{Colors.YELLOW}Added required dependencies:{Colors.RESET}")
        for dep in added_deps:
            needed_by = [p for p in selected_pipelines if dep in DEPENDENCIES.get(p, [])]
            print(f"  {Colors.CYAN}•{Colors.RESET} {dep} (needed by: {', '.join(needed_by)})")
    
    return ordered

def load_pipeline(name: str) -> Any:
    module_path = f"src.pipelines.{name}"
    
    spinner = Spinner(f"Loading {name} pipeline module")
    spinner.start()
    
    try:
        module = importlib.import_module(module_path)
        pipeline_class = getattr(module, PIPELINES[name])
        spinner.stop()
        return pipeline_class
    except Exception as e:
        spinner.stop()
        print(f"{Colors.RED}Error loading pipeline {name}: {e}{Colors.RESET}")
        raise

def setup_drive(config):
    print_section("Google Drive Setup")
    
    spinner = Spinner("Mounting Google Drive")
    spinner.start()
    
    try:
        drive = get_drive()
        project_name = config.get('project.name', 'chess_ai')
        spinner.stop()
        
        spinner = Spinner(f"Setting up project directory: {project_name}")
        spinner.start()
        drive.setup(project_name)
        spinner.stop()
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Drive mounted successfully at: {Colors.BOLD}{drive.project_dir}{Colors.RESET}")
        return drive
    except Exception as e:
        spinner.stop()
        print(f"{Colors.RED}✗ Drive setup failed: {e}{Colors.RESET}")
        choice = fancy_input("Continue without Drive? (y/n)", "y")
        if choice.lower() != 'y':
            print(f"{Colors.YELLOW}Operation cancelled by user.{Colors.RESET}")
            sys.exit(0)
        return None

def run_pipeline(pipeline_name: str, config) -> Dict:
    print_section(f"Running {pipeline_name.upper()} Pipeline")
    print(f"{Colors.CYAN}• Description:{Colors.RESET} {PIPELINE_INFO.get(pipeline_name, 'No description available')}")
    
    pipeline_class = load_pipeline(pipeline_name)
    
    start_time = time.time()
    
    try:
        pipeline = pipeline_class(config)
        
        spinner = Spinner(f"Executing {pipeline_name} pipeline")
        spinner.start()
        
        success = pipeline.run()
        spinner.stop()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            status = f"{Colors.GREEN}✓ SUCCESS{Colors.RESET}"
        else:
            status = f"{Colors.RED}✗ FAILED{Colors.RESET}"
            
        return {
            "name": pipeline_name,
            "success": success,
            "duration": duration,
            "status": status
        }
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"{Colors.RED}✗ Error in {pipeline_name} pipeline: {e}{Colors.RESET}")
        return {
            "name": pipeline_name,
            "success": False,
            "duration": duration,
            "status": f"{Colors.RED}✗ ERROR{Colors.RESET}",
            "error": str(e)
        }

def print_summary(results: List[Dict]):
    print_section("Pipeline Execution Summary")
    
    total_time = sum(result["duration"] for result in results)
    success_count = sum(1 for result in results if result["success"])
    
    max_name_length = max(len(result["name"]) for result in results)
    
    print(f"{Colors.BOLD}{'Pipeline'.ljust(max_name_length + 2)}{'Status'.ljust(20)}{'Duration'.ljust(15)}{Colors.RESET}")
    print(f"{Colors.CYAN}{'─' * (max_name_length + 35)}{Colors.RESET}")
    
    for result in results:
        name = result["name"].ljust(max_name_length + 2)
        status = result["status"].ljust(20) 
        duration = f"{result['duration']:.2f}s".ljust(15)
        print(f"{Colors.BOLD}{name}{Colors.RESET}{status}{duration}")
    
    print(f"{Colors.CYAN}{'─' * (max_name_length + 35)}{Colors.RESET}")
    print(f"Total pipelines: {len(results)} | Successful: {Colors.GREEN}{success_count}{Colors.RESET} | Failed: {Colors.RED}{len(results) - success_count}{Colors.RESET}")
    print(f"Total execution time: {Colors.YELLOW}{total_time:.2f}s{Colors.RESET}")
    
    if success_count == len(results):
        print(f"{Colors.GREEN}All pipelines completed successfully!{Colors.RESET}")
    else:
        failed = [result["name"] for result in results if not result["success"]]
        print(f"{Colors.RED}Some pipelines failed: {', '.join(failed)}{Colors.RESET}")
        print(f"{Colors.YELLOW}Check logs for more details.{Colors.RESET}")

def show_execution_plan(ordered_pipelines):
    print_section("Pipeline Execution Plan")
    
    print(f"{Colors.BOLD}Execution order:{Colors.RESET}")
    
    flow = ""
    for i, pipeline in enumerate(ordered_pipelines):
        if i > 0:
            flow += f" {Colors.BLUE}→{Colors.RESET} "
        flow += f"{Colors.CYAN}{pipeline}{Colors.RESET}"
    
    print(f"  {flow}")
    print()
    
    print(f"{Colors.BOLD}Pipeline details:{Colors.RESET}")
    for i, pipeline in enumerate(ordered_pipelines, 1):
        dependencies = DEPENDENCIES.get(pipeline, [])
        deps_str = f"{Colors.YELLOW}(depends on: {', '.join(dependencies)}){Colors.RESET}" if dependencies else ""
        print(f"  {Colors.CYAN}{i}.{Colors.RESET} {Colors.BOLD}{pipeline}{Colors.RESET} - {PIPELINE_INFO.get(pipeline, 'No description')} {deps_str}")

def main():
    print_header()
    
    try:
        mode = prompt_user("Select running mode", ["test", "prod"], "test")
        
        config_path = 'config.yml'
        print(f"\n{Colors.YELLOW}Loading configuration from:{Colors.RESET} {config_path} ({mode} mode)")
        config = get_config(config_path, mode)
        
        available_pipelines = list(PIPELINES.keys())
        selected = prompt_multiple("Select pipelines to run", available_pipelines, available_pipelines)
        
        if not selected:
            print(f"{Colors.RED}No pipelines selected. Exiting.{Colors.RESET}")
            return 0
        
        ordered_pipelines = resolve_dependencies(selected)
        
        show_execution_plan(ordered_pipelines)
        
        confirm = fancy_input("Proceed with execution? (y/n)", "y")
        if confirm.lower() != 'y':
            print(f"{Colors.YELLOW}Operation cancelled by user.{Colors.RESET}")
            return 0
        
        drive = setup_drive(config)
        
        results = []
        progress = ProgressBar(len(ordered_pipelines), prefix=f"{Colors.BOLD}Overall Progress:{Colors.RESET}", size=30)
        
        for i, pipeline in enumerate(ordered_pipelines, 1):
            progress.update(1)
            
            result = run_pipeline(pipeline, config)
            results.append(result)
            
            status_emoji = "✓" if result["success"] else "✗"
            status_color = Colors.GREEN if result["success"] else Colors.RED
            print(f"{status_color}{status_emoji}{Colors.RESET} {pipeline} completed in {result['duration']:.2f}s")
            
            if i < len(ordered_pipelines):
                print(f"{Colors.CYAN}{'─' * 50}{Colors.RESET}")
        
        print_summary(results)
        
        return 0 if all(result["success"] for result in results) else 1
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Operation cancelled by user.{Colors.RESET}")
        return 130
    except Exception as e:
        print(f"\n{Colors.BG_RED}{Colors.WHITE}ERROR: {e}{Colors.RESET}")
        print(f"{Colors.RED}An unexpected error occurred. Check the traceback above for details.{Colors.RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())