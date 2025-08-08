from collections import deque

import chessai
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from encoder import PositionEncoder


class SelfPlayEngine:
    def __init__(self, model, device: str | torch.device = "cuda"):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.encoder = PositionEncoder()
        self.mcts = chessai.MCTS(
            Config.SIMULATIONS_TRAIN, Config.C_PUCT, Config.DIRICHLET_ALPHA, Config.DIRICHLET_WEIGHT
        )
        self.buffer = deque(maxlen=Config.BUFFER_SIZE)
        self.games_played = 0

    def evaluate_position(self, position):
        self.model.eval()
        x = torch.from_numpy(self.encoder.encode_single(position)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                policy_logits, value = self.model(x)
        policy = F.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]
        value = value.detach().cpu().item()
        return policy, value

    def play_single_game(self):
        position = chessai.Position()
        data = []
        move_count = 0
        while position.result() == chessai.ONGOING and move_count < 512:
            pos = chessai.Position(position)
            policy, value = self.evaluate_position(position)
            visits = self.mcts.search(position, policy, value)
            if not visits:
                break

            moves = position.legal_moves()
            target = np.zeros(Config.POLICY_OUTPUT, dtype=np.float32)
            for move, visit_count in zip(moves, visits):
                move_index = chessai.encode_move_index(move)
                if move_index is not None and move_index < Config.POLICY_OUTPUT:
                    target[move_index] = visit_count
            policy_sum = target.sum()
            if policy_sum > 0:
                target /= policy_sum

            data.append((pos, target))
            move = self._temp_select(moves, visits, move_count)
            position.make_move(move)
            move_count += 1

        self._process_result(data, position.result())
        self.games_played += 1
        return move_count, position.result()

    def _temp_select(self, moves, visits, move_number):
        if move_number < Config.TEMP_MOVES:
            temperature = Config.TEMP_HIGH
        else:
            temperature = Config.TEMP_LOW

        if temperature > 0.01:
            probs = np.array(visits, dtype=np.float64) ** (1.0 / temperature)
            probs /= probs.sum()
            move_idx = np.random.choice(len(moves), p=probs)
        else:
            move_idx = np.argmax(visits)
        return moves[move_idx]


    def _process_result(self, data, result):
        if result == chessai.WHITE_WIN:
            values = [1.0 if i % 2 == 0 else -1.0 for i in range(len(data))]
        elif result == chessai.BLACK_WIN:
            values = [-1.0 if i % 2 == 0 else 1.0 for i in range(len(data))]
        else:
            values = [0.0] * len(data)
        for (position, policy), value in zip(data, values):
            self.buffer.append((position, policy, value))

    def generate_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = Config.BATCH_SIZE
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]
        states, policies, values = zip(*batch)
        return list(states), list(policies), list(values)

    def play_games(self, num_games):
        results = {"games": 0, "moves": 0, "white_wins": 0, "black_wins": 0, "draws": 0}
        for game_num in range(num_games):
            moves, result = self.play_single_game()
            results["games"] += 1
            results["moves"] += moves
            if result == chessai.WHITE_WIN:
                results["white_wins"] += 1
            elif result == chessai.BLACK_WIN:
                results["black_wins"] += 1
            else:
                results["draws"] += 1
            if (game_num + 1) % max(1, num_games // 20) == 0 and num_games > 100:
                progress = (game_num + 1) / num_games * 100
                print(f"  {game_num + 1}/{num_games} ({progress:.0f}%)", end="\r", flush=True)
        if num_games > 100:
            print(" " * 50, end="\r")
        return results

    def get_buffer_size(self):
        return len(self.buffer)

    def clear_buffer(self):
        self.buffer.clear()
