import os
import time
import torch

class CheckpointManager:
    def __init__(self, checkpoint_dir, checkpoint_type='epoch', checkpoint_interval=5, logger=None):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_type = checkpoint_type
        self.checkpoint_interval = checkpoint_interval
        self.logger = logger
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def should_save(self, epoch=None, batch_idx=None, iteration=None):
        if not self.checkpoint_interval or self.checkpoint_interval <= 0:
            return False

        if self.checkpoint_type == 'epoch' and epoch is not None:
            return (epoch % self.checkpoint_interval) == 0

        if self.checkpoint_type == 'batch' and batch_idx is not None:
            return (batch_idx % self.checkpoint_interval) == 0

        if self.checkpoint_type == 'iteration' and iteration is not None:
            return (iteration % self.checkpoint_interval) == 0

        return False

    def save(self, model, optimizer=None, scheduler=None, epoch=None, batch_idx=None, iteration=None, training_stats=None):
        if not self.should_save(epoch=epoch, batch_idx=batch_idx, iteration=iteration):
            return

        # Build the checkpoint data
        checkpoint_data = {
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'iteration': iteration,
            'training_stats': training_stats if training_stats else {}
        }
        
        # Perform the actual save inline
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        temp_name = f".temp_checkpoint_{timestamp}.pth"
        final_name = f"checkpoint_{timestamp}.pth"
        temp_path = os.path.join(self.checkpoint_dir, temp_name)
        final_path = os.path.join(self.checkpoint_dir, final_name)

        try:
            torch.save(checkpoint_data, temp_path)
            os.replace(temp_path, final_path)
            if self.logger:
                self.logger.info(f"Checkpoint saved: {final_name}")
        except Exception as e:
            # Clean up any leftover temp file
            if self.logger:
                self.logger.error(f"Error saving checkpoint: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as remove_e:
                    if self.logger:
                        self.logger.error(f"Failed to remove temp checkpoint: {remove_e}")
            raise

    def save_final_model(self, model, optimizer=None, scheduler=None, training_stats=None, epoch=None, batch_idx=None, iteration=None, final_path=None):
        if final_path is None:
            final_path = os.path.join(self.checkpoint_dir, "final_model.pth")

        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'iteration': iteration,
            'training_stats': training_stats if training_stats else {}
        }

        temp_path = final_path + ".temp"
        try:
            torch.save(checkpoint_data, temp_path)
            os.replace(temp_path, final_path)
            if self.logger:
                self.logger.info(f"Final model saved at {final_path}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving final model: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as remove_e:
                    if self.logger:
                        self.logger.error(f"Failed to remove temp final file: {remove_e}")
            raise

    def load(self, checkpoint_path, device, model, optimizer=None, scheduler=None):
        if not os.path.exists(checkpoint_path):
            if self.logger:
                self.logger.warning(f"Checkpoint not found at {checkpoint_path}.")
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if self.logger:
                self.logger.info(f"Checkpoint loaded from {checkpoint_path}.")
            return checkpoint
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading checkpoint: {e}")
            raise