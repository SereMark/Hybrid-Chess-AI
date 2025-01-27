import os, time, torch

class CheckpointManager:
    def __init__(self, checkpoint_dir, checkpoint_type='epoch', checkpoint_interval=5):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_type = checkpoint_type
        self.checkpoint_interval = checkpoint_interval
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def should_save(self, epoch=None, iteration=None):
        if self.checkpoint_type == 'epoch' and epoch is not None:
            return (epoch % self.checkpoint_interval) == 0
        elif self.checkpoint_type == 'iteration' and iteration is not None:
            return (iteration % self.checkpoint_interval) == 0
        return False

    def save(self, model, optimizer=None, scheduler=None, epoch=None, iteration=None):
        if not self.should_save(epoch=epoch, iteration=iteration):
            return
        checkpoint_data = {
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None
        }
        if self.checkpoint_type == 'epoch':
            checkpoint_data['epoch'] = epoch
        elif self.checkpoint_type == 'iteration':
            checkpoint_data['iteration'] = iteration
        os.makedirs(os.path.dirname(self.checkpoint_dir), exist_ok=True)
        torch.save(checkpoint_data, os.path.join(self.checkpoint_dir, f"checkpoint_{time.strftime('%Y%m%d_%H%M%S')}.pth"))

    def save_final_model(self, model, optimizer=None, scheduler=None, epoch=None, iteration=None, final_path=None):
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None
        }
        if self.checkpoint_type == 'epoch':
            checkpoint_data['epoch'] = epoch
        elif self.checkpoint_type == 'iteration':
            checkpoint_data['iteration'] = iteration
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        torch.save(checkpoint_data, final_path or os.path.join(self.checkpoint_dir, "final_model.pth"))

    def load(self, checkpoint_path, device, model, optimizer=None, scheduler=None):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint