import os
import time
import chess
import random
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from typing import Tuple, Dict
from src.utils.chess_utils import convert_board_to_tensor, get_move_mapping

def initialize_random_seeds(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # If CUDA is available, also set the GPU seeds.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    # Make PyTorch's behavior deterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_optimizer(model: torch.nn.Module, optimizer_type: str, learning_rate: float, weight_decay: float, logger=None) -> optim.Optimizer:
    # Normalize optimizer type string (case-insensitive).
    optimizer_type = optimizer_type.lower()

    # Dict mapping strings to their respective optimizer constructors.
    optimizers = {
        'adamw': optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9),
        'adam': optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
    }

    optimizer = optimizers.get(optimizer_type)
    if optimizer is None:
        if logger:
            logger.warning(f"Unsupported optimizer type: {optimizer_type}. Using AdamW by default.")
        optimizer = optimizers['adamw']
    return optimizer

def initialize_scheduler(optimizer: optim.Optimizer, scheduler_type: str, total_steps: int = None, logger=None):
    scheduler_type = scheduler_type.lower()
    schedulers = {
        'cosineannealingwarmrestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2),
        'cosineannealing': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
        'none': None,
    }

    # OneCycleLR requires knowing the total number of steps in advance.
    if scheduler_type == 'onecyclelr':
        if total_steps is None:
            if logger:
                logger.error("total_steps must be provided for OneCycleLR scheduler.")
            raise ValueError("total_steps must be provided for OneCycleLR scheduler.")
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'], total_steps=total_steps)

    scheduler = schedulers.get(scheduler_type)
    if scheduler is None:
        # Default to CosineAnnealingWarmRestarts if the requested type is unsupported.
        if logger:
            logger.warning(f"Unsupported scheduler type: {scheduler_type}. Using CosineAnnealingWarmRestarts by default.")
        scheduler = schedulers['cosineannealingwarmrestarts']
    return scheduler

def compute_policy_loss(policy_preds: torch.Tensor, policy_targets: torch.Tensor) -> torch.Tensor:
    """
    Compute a label-smoothed cross-entropy loss for policies, assuming 'policy_targets' are integer move indices.

    NOTE: This method is for single-move label classification. If the target is a distribution, use compute_policy_loss_MCTS.

    Args:
        policy_preds (torch.Tensor): Raw policy logits of shape [batch_size, num_moves].
        policy_targets (torch.Tensor): Indices of shape [batch_size], each an integer move index.

    Returns:
        torch.Tensor: A scalar loss for the policy head.
    """
    smoothing = 0.1
    confidence = 1.0 - smoothing
    n_classes = policy_preds.size(1)

    # Construct a one-hot vector with label smoothing.
    one_hot = torch.zeros_like(policy_preds).scatter(1, policy_targets.unsqueeze(1), 1)
    smoothed_labels = one_hot * confidence + (1 - one_hot) * (smoothing / (n_classes - 1))

    # Convert policy logits to log probabilities.
    log_probs = F.log_softmax(policy_preds, dim=1)

    # Cross-entropy calculation with smoothed labels.
    policy_loss = -(smoothed_labels * log_probs).sum(dim=1).mean()
    return policy_loss

def compute_policy_loss_MCTS(policy_preds: torch.Tensor, policy_targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss for MCTS-based distributions.

    Args:
        policy_preds (torch.Tensor): Raw policy logits of shape [batch_size, total_moves].
        policy_targets (torch.Tensor): Target distribution of the same shape, representing MCTS probabilities for each move.

    Returns:
        torch.Tensor: A scalar loss value representing the cross-entropy between predicted logits and the MCTS probability distribution.
    """
    # Convert policy logits to log probabilities.
    log_probs = F.log_softmax(policy_preds, dim=1)
    # Cross-entropy with the provided distribution
    loss = -(policy_targets * log_probs).sum(dim=1).mean()
    return loss

def compute_value_loss(value_preds: torch.Tensor, value_targets: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared error (MSE) loss for value head.

    Args:
        value_preds (torch.Tensor): Model's value output of shape [batch_size], or [batch_size, 1].
        value_targets (torch.Tensor): Ground truth scalar values, e.g. +1 for win, -1 for loss, 0 for draw.

    Returns:
        torch.Tensor: A scalar MSE loss.
    """
    return F.mse_loss(value_preds.view(-1), value_targets)

def compute_total_loss(policy_loss: torch.Tensor, value_loss: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Combine policy and value losses, accounting for gradient accumulation.

    Args:
        policy_loss (torch.Tensor): The policy part of the total loss.
        value_loss (torch.Tensor): The value part of the total loss.
        batch_size (int): The current batch size (important for deciding accumulation steps).

    Returns:
        torch.Tensor: A single scalar representing the combined, scaled loss.
    """
    # Attempt to keep the effective batch size around 256 for stable training.
    accumulation_steps = max(256 // batch_size, 1)
    return (policy_loss + value_loss) / accumulation_steps

def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy by comparing predicted classes to target classes.

    Args:
        predictions (torch.Tensor): The model’s raw logits of shape [batch_size, n_classes].
        targets (torch.Tensor): True class labels of shape [batch_size].

    Returns:
        float: Fraction of correct predictions in the batch.
    """
    # Predicted class is the index of max logit.
    _, predicted = torch.max(predictions.data, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

@torch.no_grad()
def policy_value_fn(board: chess.Board, model, device) -> Tuple[Dict[chess.Move, float], float]:
    """
    Given a chess board state, apply the model to get policy logits and a value estimate,
    and then map them to legal chess moves.

    Args:
        board (chess.Board): The current chess board state.
        model (torch.nn.Module): The neural network that outputs policy/value.
        device (torch.device): The device to run inference on.

    Returns:
        (action_probs, value_float):
            action_probs (Dict[chess.Move, float]): Probability distribution over legal moves.
            value_float (float): Estimated value of the current position (e.g., range -1 to +1).
    """
    # Convert board to tensor and run inference.
    board_tensor = convert_board_to_tensor(board)
    board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(device)
    policy_logits, value_out = model(board_tensor)

    # Convert policy logits to a probability distribution.
    policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
    value_float = value_out.cpu().item()
    legal_moves = list(board.legal_moves)

    # If no legal moves, return empty distribution
    if not legal_moves:
        return {}, value_float

    action_probs = {}
    total_prob = 0.0
    # Map each legal move to its probability (index-based from the move_mapping).
    for move in legal_moves:
        idx = get_move_mapping().get_index_by_move(move)
        if idx is not None and idx < len(policy):
            prob = max(policy[idx], 1e-8)  # Avoid zero probabilities.
            action_probs[move] = prob
            total_prob += prob
        else:
            action_probs[move] = 1e-8  # Very small probability for moves out of index range.

    # Normalize probabilities over legal moves if total_prob > 0.
    if total_prob > 0:
        for move in action_probs:
            action_probs[move] /= total_prob
    else:
        # Fallback to uniform distribution if total_prob is zero.
        uniform_prob = 1.0 / len(legal_moves)
        for move in action_probs:
            action_probs[move] = uniform_prob

    return action_probs, value_float

def format_time_left(seconds: float) -> str:
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    if days >= 1:
        return f"{int(days)}d {int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"

def update_progress_time_left(progress_signal, time_left_signal, start_time: float, current_step: int, total_steps: int) -> None:
    # If total steps is invalid, reset signals and return.
    if total_steps <= 0:
        if progress_signal:
            progress_signal.emit(0)
        if time_left_signal:
            time_left_signal.emit("Calculating...")
        return

    # Convert step to percentage and emit to progress bar.
    progress = max(0, min(100, int((current_step / total_steps) * 100)))
    if progress_signal:
        progress_signal.emit(progress)

    # Estimate time left based on elapsed time and steps remaining.
    elapsed = time.time() - start_time
    if current_step > 0:
        steps_left = max(0, total_steps - current_step)
        time_left = max(0, (elapsed / current_step) * steps_left)
        time_left_str = format_time_left(time_left)
        if time_left_signal:
            time_left_signal.emit(time_left_str)
    else:
        # If we haven't made progress, we can't estimate time yet.
        if time_left_signal:
            time_left_signal.emit("Calculating...")

def wait_if_paused(pause_event):
    while not pause_event.is_set():
        time.sleep(0.1)

def estimate_total_games(file_paths, avg_game_size=5000, max_games=None, logger=None) -> int:
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    total_games = 0
    for file_path in file_paths:
        try:
            if not os.path.isfile(file_path):
                if logger:
                    logger.warning(f"File not found: {file_path}. Skipping.")
                continue
            fsize = os.path.getsize(file_path)
            estimated_games = fsize // avg_game_size
            total_games += estimated_games
        except Exception as e:
            if logger:
                logger.error(f"Error estimating games for {file_path}: {e}")
    
    if max_games is not None:
        return min(total_games, max_games)
    return total_games

def get_game_result(board: chess.Board) -> float:
    result_map = {'1-0': 1.0, '0-1': -1.0, '1/2-1/2': 0.0}
    return result_map.get(board.result(), 0.0)

def parse_game_result(result: str) -> float:
    result_map = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
    return result_map.get(result, None)

def determine_outcome(result: str) -> str:
    outcome_map = {'1-0': 'win', '0-1': 'loss', '1/2-1/2': 'draw'}
    return outcome_map.get(result)