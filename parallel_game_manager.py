from typing import List, Dict, Optional, Any, Tuple
import logging
import time
import chess
import numpy as np
import torch
from game import Game
from batch_mcts import BatchMCTS
from main import get_config
from opening_book import get_opening_move_chess, OPENING_BOOK

logger = logging.getLogger(__name__)


class ParallelGameManager:
    def __init__(self, 
                 model: Any, 
                 move_encoder: Any, 
                 device: str,
                 num_parallel_games: int = 8,
                 batch_size: int = 16,
                 num_simulations: int = None,
                 max_moves_per_game: int = None) -> None:
        
        if num_parallel_games <= 0:
            raise ValueError(f"num_parallel_games must be positive, got {num_parallel_games}")
        if num_parallel_games > 64:
            logger.warning(f"Large num_parallel_games={num_parallel_games} may cause memory issues")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        self.model = model
        self.move_encoder = move_encoder
        self.device = device
        self.num_parallel_games = num_parallel_games
        self.max_moves_per_game = max_moves_per_game or get_config('training', 'max_moves_per_game')
        
        if self.max_moves_per_game <= 0:
            raise ValueError(f"max_moves_per_game must be positive, got {self.max_moves_per_game}")
        
        self.batch_mcts = BatchMCTS(
            model=model,
            move_encoder=move_encoder,
            device=device,
            batch_size=batch_size,
            num_simulations=num_simulations
        )
        
        logger.info(f"ParallelGameManager initialized: {num_parallel_games} parallel games, "
                    f"batch_size={batch_size}, max_moves={self.max_moves_per_game}")
        
    def play_games(self, use_opening_book: bool = True) -> List[Game]:
        if logger.isEnabledFor(logging.DEBUG):
            start_time = time.time()
            logger.debug(f"Starting {self.num_parallel_games} parallel games")
            
        games = [Game() for _ in range(self.num_parallel_games)]
        move_counts = [0] * self.num_parallel_games
        completed_games = []
        
        total_batch_evals = 0
        total_positions = 0
        batch_times = []
        
        high_temp = get_config('game', 'high_temperature')
        low_temp = get_config('game', 'low_temperature')
        exploration_moves = get_config('game', 'exploration_temperature_moves')
        
        while len(completed_games) < self.num_parallel_games:
            active_indices = []
            active_boards = []
            
            for i, game in enumerate(games):
                if i not in completed_games:
                    if game.is_game_over() or move_counts[i] >= self.max_moves_per_game:
                        completed_games.append(i)
                        if game.is_game_over():
                            logger.debug(f"Game {i} completed: {game.board.result()} in {move_counts[i]} moves")
                        else:
                            logger.debug(f"Game {i} reached move limit: {move_counts[i]} moves")
                    else:
                        active_indices.append(i)
                        active_boards.append(game.board)
            
            if not active_indices:
                break
                
            if logger.isEnabledFor(logging.DEBUG) and total_batch_evals % 10 == 0:
                logger.debug(f"Active games: {len(active_indices)}/{self.num_parallel_games}")
                
            board_tensors = []
            for board in active_boards:
                tensor = self.model.encode_board_vectorized(board)
                board_tensors.append(tensor)
                
            if logger.isEnabledFor(logging.DEBUG):
                batch_start = time.time()
                
            move_probs_batch = self.batch_mcts.search_batch(active_boards)
            
            if logger.isEnabledFor(logging.DEBUG):
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                logger.debug(f"Batch MCTS completed in {batch_time * 1000:.1f}ms for {len(active_boards)} positions")
                
            total_batch_evals += 1
            total_positions += len(active_boards)
            
            for idx, game_idx in enumerate(active_indices):
                game = games[game_idx]
                board = active_boards[idx]
                
                book_move = None
                if use_opening_book and move_counts[game_idx] < 15:
                    book_move = get_opening_move_chess(board)
                    
                if book_move:
                    move = book_move
                    board_tensor = board_tensors[idx]
                    move_probs = {m: 0.01 for m in board.legal_moves}
                    move_probs[move] = 0.9
                    game.add_position(board_tensor, move_probs, None)
                    
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Game {game_idx}: Using opening book move {move}")
                else:
                    move_probs = move_probs_batch[idx]
                    board_tensor = board_tensors[idx]
                    
                    if not move_probs:
                        logger.warning(f"No move probabilities for game {game_idx} at move {move_counts[game_idx]}")
                        completed_games.append(game_idx)
                        continue
                        
                    game.add_position(board_tensor, move_probs, None)
                    
                    temperature = high_temp if move_counts[game_idx] < exploration_moves else low_temp
                    
                    moves = list(move_probs.keys())
                    probs = np.array(list(move_probs.values()), dtype=np.float64)
                    
                    if temperature != 1.0:
                        probs = np.power(probs, 1.0 / temperature)
                        probs = probs / np.sum(probs)
                        
                    try:
                        move = np.random.choice(moves, p=probs)
                    except Exception as e:
                        logger.error(f"Move selection failed for game {game_idx}: {e}")
                        completed_games.append(game_idx)
                        continue
                    
                if game.history:
                    game.history[-1]['move_played'] = move
                    
                if game.make_move(move):
                    move_counts[game_idx] += 1
                    
                    if logger.isEnabledFor(logging.DEBUG) and move_counts[game_idx] % 10 == 0:
                        logger.debug(f"Game {game_idx}: {move_counts[game_idx]} moves played")
                else:
                    logger.error(f"Illegal move in game {game_idx}: {move}")
                    completed_games.append(game_idx)
                    
        if logger.isEnabledFor(logging.DEBUG):
            total_time = time.time() - start_time
            total_moves = sum(move_counts)
            avg_moves_per_game = total_moves / self.num_parallel_games
            
            logger.debug(f"Parallel games completed in {total_time:.2f}s")
            logger.debug(f"Total moves: {total_moves}, avg per game: {avg_moves_per_game:.1f}")
            logger.debug(f"Batch evaluations: {total_batch_evals}, positions evaluated: {total_positions}")
            
            if batch_times:
                avg_batch_time = sum(batch_times) / len(batch_times)
                logger.debug(f"Average batch MCTS time: {avg_batch_time * 1000:.1f}ms")
                
            games_per_minute = self.num_parallel_games / (total_time / 60)
            logger.info(f"Performance: {games_per_minute:.1f} games/minute")
            
        results = []
        for i, game in enumerate(games):
            result = game.board.result() if game.is_game_over() else "timeout"
            logger.info(f"Game {i}: {move_counts[i]} moves, result: {result}")
            results.append(game)
            
        return results
    
    def play_games_with_opening_book(self, opening_book: Optional[Dict[str, List[str]]] = None) -> List[Game]:
        games = []
        
        for i in range(self.num_parallel_games):
            game = Game()
            
            if opening_book:
                moves_applied = 0
                while game.board.fen() in opening_book and moves_applied < 10:
                    book_moves = opening_book[game.board.fen()]
                    if book_moves:
                        move_uci = np.random.choice(book_moves)
                        try:
                            move = chess.Move.from_uci(move_uci)
                            if move in game.board.legal_moves:
                                game.make_move(move)
                                moves_applied += 1
                            else:
                                break
                        except (ValueError, chess.InvalidMoveError) as e:
                            logger.warning(f"Invalid book move: {move_uci} - {e}")
                            break
                    else:
                        break
                        
                if moves_applied > 0:
                    logger.debug(f"Game {i}: Applied {moves_applied} opening book moves")
                    
            games.append(game)
            
        return self._continue_games(games)
    
    def _continue_games(self, games: List[Game]) -> List[Game]:
        move_counts = [len(game.moves_played) for game in games]
        completed_games = []
        
        high_temp = get_config('game', 'high_temperature')
        low_temp = get_config('game', 'low_temperature')
        exploration_moves = get_config('game', 'exploration_temperature_moves')
        
        while len(completed_games) < len(games):
            active_indices = []
            active_boards = []
            
            for i, game in enumerate(games):
                if i not in completed_games:
                    if game.is_game_over() or move_counts[i] >= self.max_moves_per_game:
                        completed_games.append(i)
                    else:
                        active_indices.append(i)
                        active_boards.append(game.board)
                        
            if not active_indices:
                break
                
            board_tensors = []
            for board in active_boards:
                tensor = self.model.encode_board_vectorized(board)
                board_tensors.append(tensor)
                
            move_probs_batch = self.batch_mcts.search_batch(active_boards)
            
            for idx, game_idx in enumerate(active_indices):
                game = games[game_idx]
                move_probs = move_probs_batch[idx]
                board_tensor = board_tensors[idx]
                
                if not move_probs:
                    completed_games.append(game_idx)
                    continue
                    
                game.add_position(board_tensor, move_probs, None)
                
                temperature = high_temp if move_counts[game_idx] < exploration_moves else low_temp
                
                moves = list(move_probs.keys())
                probs = np.array(list(move_probs.values()), dtype=np.float64)
                
                if temperature != 1.0:
                    probs = np.power(probs, 1.0 / temperature)
                    probs = probs / np.sum(probs)
                    
                try:
                    move = np.random.choice(moves, p=probs)
                    
                    if game.history:
                        game.history[-1]['move_played'] = move
                        
                    if game.make_move(move):
                        move_counts[game_idx] += 1
                    else:
                        completed_games.append(game_idx)
                        
                except Exception as e:
                    logger.error(f"Move selection failed: {e}")
                    completed_games.append(game_idx)
                    
        return games
