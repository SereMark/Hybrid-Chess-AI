import os, time, torch

class CheckpointManager:
    def __init__(self, checkpoint_dir, checkpoint_type='epoch', checkpoint_interval=5):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_type = checkpoint_type
        self.checkpoint_interval = checkpoint_interval
        self.time_interval = checkpoint_interval * 60 if checkpoint_type == 'time' else None
        self.last_save_time = time.time()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def should_save(self, epoch=None, iteration=None):
        condition = False
        if self.checkpoint_type == 'epoch' and epoch is not None:
            condition = (epoch % self.checkpoint_interval) == 0
        elif self.checkpoint_type == 'iteration' and iteration is not None:
            condition = (iteration % self.checkpoint_interval) == 0
        if self.time_interval and (time.time() - self.last_save_time) >= self.time_interval:
            condition = True
        return condition

    def save(self, model, optimizer=None, scheduler=None, epoch=None, iteration=None, training_stats=None):
        if not self.should_save(epoch=epoch, iteration=iteration):
            return
        checkpoint_data = {
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'iteration': iteration,
            'training_stats': training_stats if training_stats else {}
        }
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{timestamp}.pth")
        torch.save(checkpoint_data, path)
        if self.time_interval:
            self.last_save_time = time.time()

    def save_final_model(self, model, optimizer=None, scheduler=None, training_stats=None, epoch=None, iteration=None, final_path=None):
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'iteration': iteration,
            'training_stats': training_stats if training_stats else {}
        }
        torch.save(checkpoint_data, final_path or os.path.join(self.checkpoint_dir, "final_model.pth"))

    def load(self, checkpoint_path, device, model, optimizer=None, scheduler=None):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return checkpoint
        except Exception:
            raise