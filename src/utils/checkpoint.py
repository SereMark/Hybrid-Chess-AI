import os
import time
import torch

class Checkpoint:
    def __init__(self, dir_path, progress_type, interval):
        self.dir = dir_path
        self.type = progress_type
        self.interval = interval
        os.makedirs(self.dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, progress, path=None, tag=None):
        data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            self.type: progress
        }
        
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(data, path)
        else:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            tag_str = f"_{tag}" if tag else ""
            filename = f"ckpt_{timestamp}{tag_str}.pth"
            out_file = os.path.join(self.dir, filename)
            torch.save(data, out_file)
            return out_file

    def load(self, path, device, model, optimizer, scheduler):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and checkpoint.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint