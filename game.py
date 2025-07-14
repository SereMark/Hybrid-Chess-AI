from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import time
import chess
import numpy as np
import torch
from main import get_config

logger = logging.getLogger(__name__)


class Game:
    def __init__(self, fen: Optional[str] = None) -> None:
        self.board = chess.Board(fen) if fen else chess.Board()
        self.history: List[Dict[str, Any]] = []
        self.moves_played: List[chess.Move] = []
        self.game_over: bool = False
        self.result: Optional[str] = None

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def get_result(self) -> Optional[float]:
        if not self.board.is_game_over():
            return None
        result = self.board.result()
        if result == '1-0':
            return 1.0 if self.board.turn == chess.WHITE else -1.0
        elif result == '0-1':
            return -1.0 if self.board.turn == chess.WHITE else 1.0
        else:
            return 0.0

    def add_position(self, 
                    board_tensor: torch.Tensor, 
                    move_probs: Union[Dict[chess.Move, float], torch.Tensor], 
                    move_played: Optional[chess.Move]) -> None:
        self.history.append({
            'board_tensor': board_tensor.clone(),
            'move_probs': (move_probs.copy() if isinstance(move_probs, dict) 
                          else move_probs),
            'move_played': move_played,
            'player': self.board.turn
        })

    def make_move(self, move: chess.Move) -> bool:
        if move in self.board.legal_moves:
            self.moves_played.append(move)
            self.board.push(move)
            logger.debug(f"Move played: {move} (total moves: {len(self.moves_played)})")
            return True
        else:
            logger.warning(f"Illegal move attempted: {move} at position {self.board.fen()[:30]}...")
            logger.debug(f"Legal moves were: {list(self.board.legal_moves)[:5]}...")
            return False

    def get_training_data(self) -> List[Dict[str, Any]]:
        if logger.isEnabledFor(logging.DEBUG):
            extract_start = time.time()
            logger.debug(f"Extracting training data from game with {len(self.history)} positions")
        if not self.board.is_game_over():
            final_result = np.random.choice(get_config('game', 'timeout_game_values'))
            logger.debug(f"Game timeout - using random result: {final_result}")
        else:
            final_result = self.get_result()
            logger.debug(f"Game completed with result: {final_result} ({self.board.result()})")
        training_data = []
        for i, position in enumerate(self.history):
            if position['player'] == chess.WHITE:
                value = final_result if final_result is not None else 0.0
            else:
                value = -final_result if final_result is not None else 0.0
            training_data.append({
                'board_tensor': position['board_tensor'],
                'move_probs': position['move_probs'],
                'value': value,
                'move_played': position['move_played']
            })
        if logger.isEnabledFor(logging.DEBUG):
            extract_time = time.time() - extract_start
            white_positions = sum(1 for p in self.history if p['player'] == chess.WHITE)
            black_positions = len(self.history) - white_positions
            logger.debug(f"Training data extraction: {extract_time * 1000:.2f}ms")
            logger.debug(f"Position distribution - White: {white_positions}, Black: {black_positions}")
        return training_data

    def get_fen(self) -> str:
        return self.board.fen()

    def copy(self) -> 'Game':
        new_game = Game(self.board.fen())
        new_game.history = [pos.copy() for pos in self.history]
        new_game.moves_played = self.moves_played.copy()
        new_game.game_over = self.game_over
        new_game.result = self.result
        return new_game


def play_self_play_game(model: Any, 
                       move_encoder: Any, 
                       mcts: Any, 
                       device: str, 
                       max_moves: int = None) -> Game:
    if max_moves is None:
        max_moves = get_config('training', 'max_moves_per_game')
    if max_moves <= 0:
        raise ValueError(f"max_moves must be positive, got {max_moves}")
    if logger.isEnabledFor(logging.DEBUG):
        game_start = time.time()
        logger.debug(f"Starting self-play game (max {max_moves} moves)")
    game = Game()
    move_count = 0
    total_mcts_time = 0
    total_encoding_time = 0
    while not game.is_game_over() and move_count < max_moves:
        try:
            if logger.isEnabledFor(logging.DEBUG):
                encode_start = time.time()
            board_tensor = model.encode_board(game.board)
            if logger.isEnabledFor(logging.DEBUG):
                encode_time = time.time() - encode_start
                total_encoding_time += encode_time
            if logger.isEnabledFor(logging.DEBUG):
                mcts_start = time.time()
            move_probs = mcts.search(game.board)
            if logger.isEnabledFor(logging.DEBUG):
                mcts_time = time.time() - mcts_start
                total_mcts_time += mcts_time
            if not move_probs:
                logger.warning(f"No move probabilities from MCTS at move {move_count}")
                logger.debug(f"Position: {game.board.fen()}")
                break
            game.add_position(board_tensor, move_probs, None)
            high_temp = get_config('game', 'high_temperature')
            low_temp = get_config('game', 'low_temperature')
            exploration_moves = get_config('game', 'exploration_temperature_moves')
            temperature = (high_temp if move_count < exploration_moves else low_temp)
            if logger.isEnabledFor(logging.DEBUG) and move_count in [0, exploration_moves]:
                logger.debug(f"Temperature schedule: move {move_count} using temperature {temperature}")
            moves = list(move_probs.keys())
            probs = np.array(list(move_probs.values()), dtype=np.float64)
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / np.sum(probs)
            move = np.random.choice(moves, p=probs)
            if logger.isEnabledFor(logging.DEBUG):
                move_prob = move_probs.get(move, 0.0)
                adjusted_prob = probs[moves.index(move)] if move in moves else 0.0
                logger.debug(f"Move {move_count+1}: {move} (original prob: {move_prob:.3f}, "
                           f"temp-adjusted: {adjusted_prob:.3f}, temp: {temperature})")
                sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                alternatives = [f"{m} ({p:.3f})" for m, p in sorted_moves if m != move]
                if alternatives:
                    logger.debug(f"Alternatives: {', '.join(alternatives[:2])}")
            if game.history:
                game.history[-1]['move_played'] = move
            if not game.make_move(move):
                logger.error(f"Illegal move {move} at position {game.get_fen()[:20]}... (move {move_count})")
                logger.warning(f"Move selection failed - possible MCTS/encoding error")
                break
            move_count += 1
            if logger.isEnabledFor(logging.DEBUG) and move_count % 10 == 0:
                avg_mcts_time = total_mcts_time / move_count * 1000
                avg_encode_time = total_encoding_time / move_count * 1000
                logger.debug(f"Performance after {move_count} moves - "
                           f"avg MCTS: {avg_mcts_time:.1f}ms, avg encoding: {avg_encode_time:.1f}ms")
        except Exception as e:
            logger.error(f"Self-play failed at move {move_count}: {e}")
            logger.debug(f"Exception type: {type(e).__name__}")
            logger.debug(f"Position: {game.get_fen()[:30]}...")
            logger.warning(f"Game terminated early due to error")
            break
    final_result = game.board.result() if game.is_game_over() else "timeout"
    if logger.isEnabledFor(logging.DEBUG):
        game_time = time.time() - game_start
        avg_time_per_move = game_time / move_count if move_count > 0 else 0
        logger.debug(f"Game completed: {move_count} moves, {final_result}, {game_time:.2f}s total")
        logger.debug(f"Average time per move: {avg_time_per_move * 1000:.1f}ms")
        if game_time > 30.0:
            logger.warning(f"Very slow game: {game_time:.1f}s for {move_count} moves")
    logger.info(f"Self-play game: {move_count} moves, result: {final_result}")
    return game


def create_training_batch(games: List[Game], device: str) -> Optional[Dict[str, torch.Tensor]]:
    if logger.isEnabledFor(logging.DEBUG):
        batch_start = time.time()
        logger.debug(f"Creating training batch from {len(games)} games")
    all_data = []
    failed_extractions = 0
    for i, game in enumerate(games):
        try:
            training_data = game.get_training_data()
            all_data.extend(training_data)
            if logger.isEnabledFor(logging.DEBUG) and i < 3:  
                logger.debug(f"Game {i + 1}: {len(training_data)} positions extracted")
        except Exception as e:
            failed_extractions += 1
            if failed_extractions <= 3:  
                logger.warning(f"Training data extraction failed for game {i + 1}: {e}")
            continue
    if failed_extractions > 0:
        failure_rate = failed_extractions / len(games) * 100
        if failure_rate > 20:
            logger.warning(f"High extraction failure rate: {failure_rate:.1f}% ({failed_extractions}/{len(games)} games)")
        else:
            logger.debug(f"Extraction failures: {failed_extractions}/{len(games)} games ({failure_rate:.1f}%)")
    if not all_data:
        logger.warning("No training data from games")
        logger.error("Batch creation failed: no valid positions")
        return None
    logger.debug(f"Total positions collected: {len(all_data)}")
    try:
        if logger.isEnabledFor(logging.DEBUG):
            tensor_start = time.time()
        board_tensors = []
        target_values = []
        target_policies = []
        from model import MoveEncoder
        if not hasattr(create_training_batch, '_move_encoder'):
            create_training_batch._move_encoder = MoveEncoder()
            logger.debug("Created cached move encoder for batch processing")
        move_encoder = create_training_batch._move_encoder
        policy_nonzero_count = 0
        value_distribution = {'positive': 0, 'negative': 0, 'zero': 0}
        for i, data in enumerate(all_data):
            board_tensors.append(data['board_tensor'])
            value = data['value']
            target_values.append(value)
            if value > 0.1:
                value_distribution['positive'] += 1
            elif value < -0.1:
                value_distribution['negative'] += 1
            else:
                value_distribution['zero'] += 1
            move_space_size = get_config('model', 'move_space_size')
            policy_vector = torch.zeros(move_space_size, dtype=torch.float32)
            if isinstance(data['move_probs'], dict):
                valid_moves = 0
                for move, prob in data['move_probs'].items():
                    idx = move_encoder.encode_move(move)
                    if 0 <= idx < move_space_size:
                        policy_vector[idx] = prob
                        valid_moves += 1
                if valid_moves > 0:
                    policy_nonzero_count += 1
                elif i < 3:  
                    logger.debug(f"Position {i}: no valid policy moves encoded")
            target_policies.append(policy_vector)
        if logger.isEnabledFor(logging.DEBUG):
            tensor_time = time.time() - tensor_start
            logger.debug(f"Tensor preparation: {tensor_time * 1000:.1f}ms")
            logger.debug(f"Valid policies: {policy_nonzero_count}/{len(all_data)} positions")
            logger.debug(f"Value distribution: {value_distribution}")
        try:
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / 1024**2
        except:
            memory_before = 0
        if logger.isEnabledFor(logging.DEBUG):
            stack_start = time.time()
        batch = {
            'board_tensors': torch.stack(board_tensors).to(device),
            'target_values': torch.tensor(target_values, dtype=torch.float32).to(device),
            'target_policies': torch.stack(target_policies).to(device)
        }
        if logger.isEnabledFor(logging.DEBUG):
            stack_time = time.time() - stack_start
            logger.debug(f"Tensor stacking and device transfer: {stack_time * 1000:.1f}ms")
        try:
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**2
                memory_used = memory_after - memory_before
                if memory_used > 100:  
                    logger.warning(f"Large batch memory usage: {memory_used:.1f}MB")
                else:
                    logger.debug(f"Batch memory usage: {memory_used:.1f}MB")
        except:
            pass
        if logger.isEnabledFor(logging.DEBUG):
            total_time = time.time() - batch_start
            positions_per_sec = len(all_data) / total_time if total_time > 0 else 0
            logger.debug(f"Batch creation completed: {total_time * 1000:.1f}ms total ({positions_per_sec:.1f} positions/sec)")
            logger.debug(f"Final batch shapes - boards: {batch['board_tensors'].shape}, "
                        f"values: {batch['target_values'].shape}, policies: {batch['target_policies'].shape}")
        return batch
    except Exception as e:
        logger.error(f"Training batch creation failed with {len(all_data)} positions: {e}")
        logger.debug(f"Batch creation error details: {type(e).__name__}")
        if "memory" in str(e).lower():
            logger.warning(f"Memory pressure during batch creation (batch size: {len(all_data)} positions)")
            logger.warning("Consider reducing games per iteration or max moves per game")
        raise RuntimeError(f"Training batch creation failed: {e}") from e
