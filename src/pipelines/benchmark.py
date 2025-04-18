import os
import time
import json
import wandb
import chess
import chess.pgn
import chess.engine
import numpy as np
import torch
from tqdm.auto import tqdm

from src.utils.config import Config
from src.utils.chess import board_to_input, get_move_map, get_move_count
from src.utils.tpu import get_tpu
from src.model import ChessModel
from src.utils.mcts import MCTS

class ChessBot:
    def __init__(self, model_path, use_mcts=True, use_book=True, name="ChessAI"):
        self.name = name
        self.use_mcts = use_mcts
        self.use_book = use_book
        
        tpu = get_tpu()
        self.device = tpu.get_device() if tpu.available else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_type = tpu.get_type() if tpu.available else ('gpu' if torch.cuda.is_available() else 'cpu')
        
        self.model = ChessModel(
            moves=get_move_count(),
            use_tpu=(self.device_type == 'tpu')
        ).to(self.device)
        
        self._load_model(model_path)
        
        self.mcts = MCTS(
            self.model, self.device, c_puct=1.4, n_sims=100
        ) if use_mcts else None
        
        self.move_map = get_move_map()
    
    def _load_model(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def get_move(self, board, book):
        try:
            if self.use_book and book:
                fen = board.fen()
                moves_dict = book.get(fen)
                
                if moves_dict:
                    best_move = None
                    best_score = -1
                    
                    for uci_move, stats in moves_dict.items():
                        total_games = stats.get('win', 0) + stats.get('draw', 0) + stats.get('loss', 0)
                        
                        if total_games > 0:
                            score = (stats.get('win', 0) + 0.5 * stats.get('draw', 0)) / total_games
                            move = chess.Move.from_uci(uci_move)
                            
                            if move in board.legal_moves and score > best_score:
                                best_move, best_score = move, score
                    
                    if best_move:
                        return best_move
            
            if self.use_mcts and self.mcts:
                self.mcts.set_root(board.copy())
                action_probs = self.mcts.get_move_probs(temperature=1e-3)
                
                if board.fullmove_number == 1 and board.turn == chess.WHITE and len(action_probs) > 1:
                    moves_list = list(action_probs.keys())
                    noise = np.random.dirichlet([0.3] * len(moves_list))
                    for i, move in enumerate(moves_list):
                        action_probs[move] = 0.75 * action_probs[move] + 0.25 * noise[i]
                
                if action_probs:
                    return max(action_probs, key=action_probs.get)
                else:
                    return chess.Move.null()
            
            input_tensor = torch.from_numpy(board_to_input(board)).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                policy_logits, _ = self.model(input_tensor)
                policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
            
            move_probs = {}
            for move in board.legal_moves:
                move_idx = self.move_map.idx_by_move(move)
                prob = policy[move_idx] if move_idx is not None and move_idx < len(policy) else 1e-8
                move_probs[move] = prob
            
            total_prob = sum(move_probs.values())
            if total_prob > 0:
                for move in move_probs:
                    move_probs[move] /= total_prob
            else:
                for move in move_probs:
                    move_probs[move] = 1.0 / len(move_probs)
            
            return max(move_probs, key=move_probs.get) if move_probs else chess.Move.null()
        
        except Exception as e:
            print(f"Error getting move: {e}")
            return chess.Move.null()

class StockfishBot:
    def __init__(self, stockfish_path=None, elo=1500, time_limit=0.1, depth=None, name="Stockfish"):
        self.name = name
        self.elo = elo
        self.time_limit = time_limit
        self.depth = depth
        self.stockfish_path = stockfish_path
        self.engine = None
        self.transport = None
        self.setup()
    
    def setup(self):
        try:
            if not self.stockfish_path or not os.path.isfile(self.stockfish_path):
                raise FileNotFoundError("Stockfish not found. Please provide a valid path in the config.")
            
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            
            if self.elo:
                self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})
            
            print(f"Stockfish engine initialized with ELO: {self.elo} from path: {self.stockfish_path}")
        except Exception as e:
            print(f"Error setting up Stockfish: {e}")
            raise
    
    def _check_command(self, command):
        try:
            import subprocess
            result = subprocess.run([command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1)
            return result.returncode == 0
        except Exception:
            return False
    
    def get_move(self, board, _=None):
        if not self.engine:
            print("Stockfish engine not initialized")
            return chess.Move.null()
        
        try:
            if self.depth:
                result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
            else:
                result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
            
            return result.move
        except Exception as e:
            print(f"Error getting move from Stockfish: {e}")
            return chess.Move.null()
    
    def close(self):
        if self.engine:
            self.engine.quit()

class BenchmarkPipeline:
    def __init__(self, config: Config):
        self.config = config
        
        self.games = config.get('benchmark.games', 10)
        self.mcts = config.get('benchmark.mcts', True)
        self.opening_book = config.get('benchmark.opening_book', True)
        self.switch_colors = True
        
        self.stockfish_path = config.get('benchmark.stockfish_path', 'engines/stockfish-ubuntu-x86-64-avx2')
        self.stockfish_elo = config.get('benchmark.stockfish_elo', 1500)
        self.stockfish_time = config.get('benchmark.stockfish_time', 0.1)
        self.stockfish_depth = config.get('benchmark.stockfish_depth', None)
        
        self.model_path = None
        self.model_type = None
        
        self.book = {}
        
        self.output_dir = '/content/drive/MyDrive/chess_ai/benchmark'
        self.games_dir = os.path.join(self.output_dir, 'games')
        os.makedirs(self.games_dir, exist_ok=True)
    
    def setup(self):
        print("Setting up benchmark pipeline...")
        
        try:
            reinforcement_model_path = '/content/drive/MyDrive/chess_ai/models/reinforcement_model.pth'
            supervised_model_path = '/content/drive/MyDrive/chess_ai/models/supervised_model.pth'
            
            if os.path.exists(reinforcement_model_path):
                self.model_path = reinforcement_model_path
                self.model_type = "Reinforcement"
                print(f"Using reinforcement learning model: {self.model_path}")
            elif os.path.exists(supervised_model_path):
                self.model_path = supervised_model_path
                self.model_type = "Supervised"
                print(f"Using supervised learning model: {self.model_path}")
            else:
                print("No model found for benchmarking")
                return False
            
            book_path = '/content/drive/MyDrive/chess_ai/data/opening_book.json'
            if os.path.exists(book_path):
                with open(book_path, 'r') as f:
                    self.book = json.load(f)
                print(f"Loaded opening book with {len(self.book)} positions")
            else:
                print("Opening book not found, starting without book")
                self.book = {}
                
        except Exception as e:
            print(f"Error setting up benchmark: {e}")
            return False
        
        if not self.model_path:
            print("Error: No model available for benchmarking")
            return False
        
        if self.config.get('wandb.enabled', True):
            try:
                wandb.init(
                    project=self.config.get('wandb.project', 'chess_ai'),
                    name=f"stockfish_benchmark_{self.config.mode}_{time.strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "mode": self.config.mode,
                        "games": self.games,
                        "model_path": self.model_path,
                        "model_type": self.model_type,
                        "mcts": self.mcts,
                        "opening_book": self.opening_book,
                        "switch_colors": self.switch_colors,
                        "stockfish_elo": self.stockfish_elo
                    }
                )
            except Exception as e:
                print(f"Error initializing wandb: {e}")
        
        return True
    
    def run(self):
        if not self.setup():
            return False
        
        try:
            chess_ai = ChessBot(
                self.model_path, 
                use_mcts=self.mcts, 
                use_book=self.opening_book,
                name=f"{self.model_type}AI"
            )
            
            stockfish = StockfishBot(
                stockfish_path=self.stockfish_path,
                elo=self.stockfish_elo,
                time_limit=self.stockfish_time,
                depth=self.stockfish_depth,
                name="Stockfish"
            )
            
            results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, '*': 0}
            durations = []
            move_counts = []
            chess_ai_wins = 0
            stockfish_wins = 0
            draws = 0
            win_history = []
            
            ai_plays_white = True
            
            print(f"Running {self.games} benchmark games against Stockfish (ELO: {self.stockfish_elo})...")
            
            for game_idx in tqdm(range(1, self.games + 1)):
                game_start = time.time()
                
                board = chess.Board()
                game = chess.pgn.Game()
                
                white_name = chess_ai.name if ai_plays_white else stockfish.name
                black_name = stockfish.name if ai_plays_white else chess_ai.name
                
                game.headers.update({
                    'Event': 'Stockfish Benchmark',
                    'Site': 'Colab',
                    'Date': time.strftime('%Y.%m.%d'),
                    'Round': str(game_idx),
                    'White': white_name,
                    'Black': black_name,
                    'Result': '*'
                })
                
                node = game
                moves_played = 0
                
                while not board.is_game_over():
                    ai_turn = (board.turn == chess.WHITE and ai_plays_white) or \
                              (board.turn == chess.BLACK and not ai_plays_white)
                    
                    current_player = chess_ai if ai_turn else stockfish
                    
                    move = current_player.get_move(board, self.book)
                    
                    if move == chess.Move.null() or move not in board.legal_moves:
                        print(f"Invalid move from {current_player.name}")
                        break
                    
                    board.push(move)
                    node = node.add_variation(move)
                    moves_played += 1
                
                result = board.result()
                results[result] += 1
                
                game.headers['Result'] = result
                
                pgn_path = os.path.join(self.games_dir, f'game_{game_idx}.pgn')
                with open(pgn_path, 'w') as f:
                    f.write(str(game))
                
                game_duration = time.time() - game_start
                durations.append(game_duration)
                move_counts.append(moves_played)
                
                if result == '1-0':
                    if ai_plays_white:
                        chess_ai_wins += 1
                    else:
                        stockfish_wins += 1
                elif result == '0-1':
                    if ai_plays_white:
                        stockfish_wins += 1
                    else:
                        chess_ai_wins += 1
                elif result == '1/2-1/2':
                    draws += 1
                
                win_history.append((chess_ai_wins, stockfish_wins, draws))
                
                if wandb.run is not None:
                    wandb.log({
                        'game_idx': game_idx,
                        'result': result,
                        'duration': game_duration,
                        'moves': moves_played,
                        'chess_ai_wins': chess_ai_wins,
                        'stockfish_wins': stockfish_wins,
                        'draws': draws,
                        'ai_plays_white': ai_plays_white
                    })
                
                if self.switch_colors:
                    ai_plays_white = not ai_plays_white
            
            stockfish.close()
            
            avg_duration = float(np.mean(durations)) if durations else 0
            avg_moves = float(np.mean(move_counts)) if move_counts else 0
            win_rate = chess_ai_wins / (chess_ai_wins + stockfish_wins + draws) if (chess_ai_wins + stockfish_wins + draws) > 0 else 0
            
            print("\nBenchmark Results vs Stockfish:")
            print(f"{chess_ai.name} wins: {chess_ai_wins}")
            print(f"Stockfish wins: {stockfish_wins}")
            print(f"Draws: {draws}")
            print(f"Unfinished games: {results['*']}")
            print(f"Win rate: {win_rate:.2%}")
            print(f"Average game duration: {avg_duration:.2f} seconds")
            print(f"Average moves per game: {avg_moves:.1f}")
            
            if wandb.run is not None:
                wandb.log({
                    'total_games': self.games,
                    'chess_ai_wins': chess_ai_wins,
                    'stockfish_wins': stockfish_wins,
                    'draws': draws,
                    'unfinished': results['*'],
                    'win_rate': win_rate,
                    'avg_duration': avg_duration,
                    'avg_moves': avg_moves
                })
                
                results_table = wandb.Table(
                    data=[
                        ['ChessAI Win', chess_ai_wins],
                        ['Stockfish Win', stockfish_wins],
                        ['Draw', draws],
                        ['Unfinished', results['*']]
                    ],
                    columns=['Result', 'Count']
                )
                
                history_table = wandb.Table(
                    data=[(i+1, w[0], w[1], w[2]) for i, w in enumerate(win_history)],
                    columns=['Game', 'ChessAI', 'Stockfish', 'Draws']
                )
                
                wandb.log({
                    'results_dist': wandb.plot.bar(
                        results_table, 'Result', 'Count', title='Results Distribution'
                    ),
                    'win_history': wandb.plot.line(
                        history_table, 'Game', ['ChessAI', 'Stockfish', 'Draws'], 
                        title='Cumulative Results'
                    )
                })
                
                wandb.finish()
            
            summary_path = os.path.join(self.output_dir, 'stockfish_benchmark_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Model: {chess_ai.name}\n")
                f.write(f"Stockfish ELO: {self.stockfish_elo}\n")
                f.write(f"Games played: {self.games}\n\n")
                f.write(f"ChessAI wins: {chess_ai_wins}\n")
                f.write(f"Stockfish wins: {stockfish_wins}\n")
                f.write(f"Draws: {draws}\n")
                f.write(f"Unfinished: {results['*']}\n")
                f.write(f"Win rate: {win_rate:.2%}\n\n")
                f.write(f"Average duration: {avg_duration:.2f} seconds\n")
                f.write(f"Average moves: {avg_moves:.1f}\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"Saved benchmark results to {summary_path}")
            print(f"Saved game PGNs to {self.games_dir}")
            
            return {
                'chess_ai_wins': chess_ai_wins,
                'stockfish_wins': stockfish_wins,
                'draws': draws,
                'unfinished': results['*'],
                'win_rate': win_rate,
                'model_used': self.model_type
            }
            
        except Exception as e:
            print(f"Error during benchmark: {e}")
            import traceback
            traceback.print_exc()
            if wandb.run is not None:
                wandb.finish()
            return False