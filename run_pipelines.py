#!/usr/bin/env python3
import argparse
import importlib
from typing import Any

from src.utils.config import get_config
from src.utils.drive import get_drive

PIPELINES = {
    'supervised': 'SupervisedPipeline',
    'reinforcement': 'ReinforcementPipeline',
    'data': 'DataPipeline',
    'eval': 'EvalPipeline',
    'hyperopt': 'HyperoptPipeline',
    'benchmark': 'BenchmarkPipeline'
}

def parse_args():
    parser = argparse.ArgumentParser(description="Chess AI Pipeline Management")
    parser.add_argument('--config', type=str, default='config.yml', help='Config file path')
    parser.add_argument('--mode', type=str, choices=['test', 'prod'], default='test', help='Running mode')
    parser.add_argument('--pipeline', type=str, choices=['all'] + list(PIPELINES.keys()), default='all', help='Pipeline to run')
    parser.add_argument('--skip', type=str, nargs='*', default=[], help='Pipelines to skip when running "all"')
    parser.add_argument('--params', type=str, nargs='*', default=[], help='Override config parameters (key=value)')
    parser.add_argument('--no-drive', action='store_true', help='Skip Google Drive mounting')
    return parser.parse_args()

def load_pipeline(name: str) -> Any:
    module_path = f"src.pipelines.{name}"
    module = importlib.import_module(module_path)
    return getattr(module, PIPELINES[name])

def override_config(config: Any, params: list) -> None:
    for param in params:
        if '=' not in param:
            print(f"Warning: Ignoring invalid parameter: {param}")
            continue
        key, value = param.split('=', 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ('true', 'yes', 'y', '1'):
                    value = True
                elif value.lower() in ('false', 'no', 'n', '0'):
                    value = False
        print(f"Overriding config: {key} = {value}")

def main():
    args = parse_args()
    config = get_config(args.config, args.mode)
    
    if args.params:
        override_config(config, args.params)
    
    if not args.no_drive:
        try:
            drive = get_drive()
            project_name = config.get('project.name', 'chess_ai')
            drive.setup(project_name)
            print(f"Drive mounted at: {drive.project_dir}")
        except Exception as e:
            print(f"Warning: Drive setup failed: {e}")
    
    pipelines = [p for p in PIPELINES.keys() if p not in args.skip] if args.pipeline == 'all' else [args.pipeline]
    results = {}
    
    for pipeline in pipelines:
        print(f"\n{'='*50}\nRunning {pipeline} pipeline\n{'='*50}")
        pipeline_class = load_pipeline(pipeline)
        success = pipeline_class(config).run()
        results[pipeline] = "SUCCESS" if success else "FAILED"
    
    print("\n\nPipeline Execution Summary\n" + "="*30)
    for pipeline, status in results.items():
        print(f"{pipeline.ljust(15)}: {status}")
    print("="*30)
    
    return 0 if all(status == "SUCCESS" for status in results.values()) else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())