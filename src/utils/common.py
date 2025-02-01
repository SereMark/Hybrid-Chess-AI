import torch
try:
    import wandb
except ImportError:
    wandb = None
from src.models.transformer import TransformerCNNChessModel
from src.utils.chess_utils import get_total_moves

def load_model_from_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = TransformerCNNChessModel(num_moves=get_total_moves()).to(device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

def init_wandb_run(run_name, config, entity="chess_ai", project="chess_ai_app", reinit=True):
    if wandb is not None:
        return wandb.init(entity=entity, project=project, name=run_name, config=config, reinit=reinit)
    return None

def wandb_log(data):
    if wandb is not None:
        wandb.log(data)

def wandb_watch(model, log="all", log_freq=100):
    if wandb is not None:
        wandb.watch(model, log=log, log_freq=log_freq)

def finish_wandb():
    if wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass