import os, time, torch

class CheckpointManager:
    def __init__(self, checkpoint_dir, checkpoint_type='epoch', checkpoint_interval=5, log_fn=None):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_type = checkpoint_type
        self.checkpoint_interval = checkpoint_interval
        self.log_fn = log_fn
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def should_save(self, epoch=None, batch_idx=None, iteration=None):
        if self.checkpoint_type == 'epoch' and epoch is not None:
            return epoch % self.checkpoint_interval == 0
        elif self.checkpoint_type == 'batch' and batch_idx is not None:
            return batch_idx % self.checkpoint_interval == 0
        elif self.checkpoint_type == 'iteration' and iteration is not None:
            return iteration % self.checkpoint_interval == 0
        return False

    def save(self, checkpoint_data, prefix='checkpoint'):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f'{prefix}_{timestamp}.pth'
        temp_path = os.path.join(self.checkpoint_dir, f'.temp_{checkpoint_name}')
        final_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        try:
            torch.save(checkpoint_data, temp_path)
            os.replace(temp_path, final_path)
            if self.log_fn:
                self.log_fn(f"Checkpoint saved: {checkpoint_name}")
        except Exception as e:
            if self.log_fn:
                self.log_fn(f"Error saving checkpoint: {str(e)}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as remove_e:
                    if self.log_fn:
                        self.log_fn(f"Failed to remove temp checkpoint: {str(remove_e)}")
            raise

    def load(self, checkpoint_path, device, model, optimizer=None, scheduler=None):
        if not os.path.exists(checkpoint_path):
            if self.log_fn:
                self.log_fn(f"Checkpoint not found at {checkpoint_path}.")
            return None
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint and hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.log_fn:
                self.log_fn(f"Checkpoint loaded from {checkpoint_path}.")
            return checkpoint
        except Exception as e:
            if self.log_fn:
                self.log_fn(f"Error loading checkpoint: {str(e)}")
            raise