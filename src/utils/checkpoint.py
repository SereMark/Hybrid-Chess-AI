import os
import time
import torch
from src.utils.tpu import get_tpu

class Checkpoint:
    def __init__(self, dir_path, progress_type, interval):
        self.dir = dir_path
        self.type = progress_type
        self.interval = interval
        os.makedirs(self.dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, progress, path=None, tag=None):
        cpu_model_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                         for k, v in model.state_dict().items()}
        
        cpu_optimizer_state = None
        if optimizer:
            try:
                cpu_optimizer_state = optimizer.state_dict()
                for param_group in cpu_optimizer_state['param_groups']:
                    for k, v in param_group.items():
                        if isinstance(v, torch.Tensor):
                            param_group[k] = v.cpu()
                
                optimizer_state = cpu_optimizer_state['state']
                for _, state in optimizer_state.items():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cpu()
            except Exception as e:
                print(f"Warning: Could not convert optimizer state to CPU: {e}")
                cpu_optimizer_state = optimizer.state_dict()
        
        cpu_scheduler_state = None
        if scheduler is not None:
            try:
                cpu_scheduler_state = scheduler.state_dict()
                for k, v in cpu_scheduler_state.items():
                    if isinstance(v, torch.Tensor):
                        cpu_scheduler_state[k] = v.cpu()
            except Exception as e:
                print(f"Warning: Could not convert scheduler state to CPU: {e}")
                cpu_scheduler_state = scheduler.state_dict()
        
        data = {
            'model': cpu_model_state,
            'optimizer': cpu_optimizer_state,
            'scheduler': cpu_scheduler_state,
            self.type: progress
        }
        
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(data, path)
            print(f"Checkpoint saved to {path}")
        else:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            tag_str = f"_{tag}" if tag else ""
            filename = f"ckpt_{timestamp}{tag_str}.pth"
            out_file = os.path.join(self.dir, filename)
            torch.save(data, out_file)
            print(f"Checkpoint saved to {out_file}")
            return out_file

    def load(self, path, device, model, optimizer, scheduler):
        tpu = get_tpu()
        
        try:
            checkpoint = tpu.load(path, map_location='cpu')
            
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
                print(f"Model state loaded from {path}")
                
                if optimizer and 'optimizer' in checkpoint and checkpoint['optimizer']:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        print("Optimizer state loaded")
                    except Exception as e:
                        print(f"Warning: Could not load optimizer state: {e}")
                
                if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler']:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler'])
                        print("Scheduler state loaded")
                    except Exception as e:
                        print(f"Warning: Could not load scheduler state: {e}")
                        
                model.to(device)
                return checkpoint
            else:
                model.load_state_dict(checkpoint, strict=False)
                model.to(device)
                print(f"Model state loaded from {path}")
                return {'model': checkpoint}
        except Exception as e:
            print(f"Error loading checkpoint: {e}, trying fallback method...")
            try:
                checkpoint = torch.load(path, map_location='cpu')
                
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                    
                model.to(device)
                print("Model loaded using fallback method")
                return checkpoint if isinstance(checkpoint, dict) else {'model': checkpoint}
            except Exception as e2:
                print(f"All loading methods failed: {e2}")
                raise