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

def wandb_log(data):
    if wandb is not None:
        wandb.log(data)