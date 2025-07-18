import numpy as np
import torch
from config import get_config
from move_encoder import MoveEncoder


class ExperienceBuffer:
    def __init__(self, max_size: int = 50000, device: str = "cuda") -> None:
        self.max_size = max_size
        self.device = device
        self.size = 0
        self.pos = 0

        board_encoding_size = 8 * 8 * 6 * 2
        move_space_size = get_config("model", "move_space_size") or 4096

        self.boards = np.zeros((max_size, board_encoding_size), dtype=np.float32)
        self.values = np.zeros(max_size, dtype=np.float32)
        self.policies = np.zeros((max_size, move_space_size), dtype=np.float32)

        self._move_encoder = MoveEncoder()

    def add_batch(self, games) -> int:
        if not games:
            return 0

        positions_added = 0
        for game in games:
            training_data = game.get_training_data()

            for position in training_data:
                board_tensor = position["board_tensor"]
                value = position["value"]
                move_probs = position["move_probs"]

                if isinstance(board_tensor, torch.Tensor):
                    board_flat = board_tensor.cpu().numpy().flatten()
                else:
                    board_flat = board_tensor.flatten()
                self.boards[self.pos] = board_flat

                self.values[self.pos] = value

                if isinstance(move_probs, dict):
                    policy_vector = np.zeros(self.policies.shape[1], dtype=np.float32)
                    for move, prob in move_probs.items():
                        idx = self._move_encoder.encode_move(move)
                        if 0 <= idx < len(policy_vector):
                            policy_vector[idx] = prob
                    self.policies[self.pos] = policy_vector
                elif isinstance(move_probs, torch.Tensor):
                    self.policies[self.pos] = move_probs.cpu().numpy()
                else:
                    self.policies[self.pos] = move_probs

                self.pos = (self.pos + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)
                positions_added += 1

        return positions_added

    def sample(self, batch_size: int | None = None):
        if batch_size is None:
            batch_size = get_config("training", "batch_size") or 128
        if self.size == 0:
            return None

        batch_size = min(batch_size, self.size)
        indices = np.random.default_rng().choice(self.size, batch_size, replace=False)

        return {
            "board_tensors": torch.from_numpy(self.boards[indices]).to(self.device),
            "target_values": torch.from_numpy(self.values[indices]).to(self.device),
            "target_policies": torch.from_numpy(self.policies[indices]).to(self.device),
        }
