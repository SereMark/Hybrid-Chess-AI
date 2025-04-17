import os, time, torch

class CheckpointManager:
    def __init__(self, checkpoint_dir, checkpoint_type, checkpoint_interval):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_type = checkpoint_type
        self.checkpoint_interval = checkpoint_interval
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, progress, final_path=None):
        data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            self.checkpoint_type: progress
        }
        if final_path:
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            torch.save(data, final_path)
        else:
            out_file = os.path.join(self.checkpoint_dir, f"checkpoint_{time.strftime('%Y%m%d_%H%M%S')}.pth")
            torch.save(data, out_file)

    def load(self, checkpoint_path, device, model, optimizer, scheduler):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint