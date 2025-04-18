import os
import time
import json
import wandb
import chess
import chess.pgn
import numpy as np
import torch
from tqdm.auto import tqdm

from src.utils.config import Config
from src.utils.drive import get_drive
from src.utils.chess import board_to_input, get_move_map, get_move_count
from src.utils.tpu import get_tpu
from src.model import ChessModel
from src.utils.mcts import MCTS

class Bot:
    def __init__(self, model_path, use_mcts=True, use_book=True, name="Bot"):
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
                    probs = np.array(list(action_probs.values()), dtype=np.float32)
                    probs /= probs.sum()
                    action_probs = dict(zip(moves_list, probs))
                
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

class BenchmarkPipeline:
    def __init__(self, config: Config):
        self.config = config
        
        self.games = config.get('benchmark.games', 10)
        self.mcts = config.get('benchmark.mcts', True)
        self.opening_book = config.get('benchmark.opening_book', True)
        self.switch_colors = True
        
        self.model1_path = None
        self.model2_path = None
        
        self.book = {}
        
        self.output_dir = '/content/drive/MyDrive/chess_ai/benchmark'
        self.games_dir = os.path.join(self.output_dir, 'games')
        os.makedirs(self.games_dir, exist_ok=True)
    
    def setup(self):
        print("Setting up benchmark pipeline...")
        
        try:
            drive = get_drive()
            
            try:
                model1_path = 'models/supervised_model.pth'
                local_model1_path = '/content/drive/MyDrive/chess_ai/models/supervised_model.pth'
                os.makedirs(os.path.dirname(local_model1_path), exist_ok=True)
                self.model1_path = drive.load(model1_path, local_model1_path)
                print(f"Loaded model 1 from Drive: {self.model1_path}")
            except FileNotFoundError:
                print("Supervised model not found in Drive")
            
            try:
                model2_path = 'models/reinforcement_model.pth'
                local_model2_path = '/content/drive/MyDrive/chess_ai/models/reinforcement_model.pth'
                os.makedirs(os.path.dirname(local_model2_path), exist_ok=True)
                self.model2_path = drive.load(model2_path, local_model2_path)
                print(f"Loaded model 2 from Drive: {self.model2_path}")
            except FileNotFoundError:
                print("Reinforcement model not found in Drive")
                
            if not self.model1_path and os.path.exists('/content/drive/MyDrive/chess_ai/models/supervised_model.pth'):
                self.model1_path = '/content/drive/MyDrive/chess_ai/models/supervised_model.pth'
                print(f"Using local supervised model: {self.model1_path}")
            
            if not self.model2_path and os.path.exists('/content/drive/MyDrive/chess_ai/models/reinforcement_model.pth'):
                self.model2_path = '/content/drive/MyDrive/chess_ai/models/reinforcement_model.pth'
                print(f"Using local reinforcement model: {self.model2_path}")
            
            try:
                book_path = 'data/opening_book.json'
                local_book_path = '/content/drive/MyDrive/chess_ai/data/opening_book.json'
                os.makedirs(os.path.dirname(local_book_path), exist_ok=True)
                
                drive.load(book_path, local_book_path)
                
                with open(local_book_path, 'r') as f:
                    self.book = json.load(f)
                
                print(f"Loaded opening book with {len(self.book)} positions")
            except Exception as e:
                print(f"Error loading opening book: {e}")
                self.book = {}
                
        except Exception as e:
            print(f"Error setting up from Drive: {e}")
        
        if not self.model1_path or not self.model2_path:
            print("Error: Need two distinct models for benchmarking")
            return False
        
        if self.model1_path == self.model2_path:
            print("Error: Model paths are identical")
            return False
        
        if self.config.get('wandb.enabled', True):
            try:
                wandb.init(
                    project=self.config.get('wandb.project', 'chess_ai'),
                    name=f"benchmark_{self.config.mode}_{time.strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "mode": self.config.mode,
                        "games": self.games,
                        "model1_path": self.model1_path,
                        "model2_path": self.model2_path,
                        "mcts": self.mcts,
                        "opening_book": self.opening_book,
                        "switch_colors": self.switch_colors
                    }
                )
            except Exception as e:
                print(f"Error initializing wandb: {e}")
        
        return True
    
    def run(self):
        if not self.setup():
            return False
        
        try:
            bot1 = Bot(
                self.model1_path, 
                use_mcts=self.mcts, 
                use_book=self.opening_book,
                name="SupervisedBot"
            )
            
            bot2 = Bot(
                self.model2_path, 
                use_mcts=self.mcts, 
                use_book=self.opening_book,
                name="ReinforcementBot"
            )
            
            results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, '*': 0}
            durations = []
            move_counts = []
            bot1_wins = 0
            bot2_wins = 0
            draws = 0
            win_history = []
            
            white_is_bot1 = True
            
            print(f"Running {self.games} benchmark games...")
            
            for game_idx in tqdm(range(1, self.games + 1)):
                game_start = time.time()
                
                board = chess.Board()
                game = chess.pgn.Game()
                
                white_name = bot1.name if white_is_bot1 else bot2.name
                black_name = bot2.name if white_is_bot1 else bot1.name
                
                game.headers.update({
                    'Event': 'Benchmark',
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
                    current_bot = bot1 if (board.turn == chess.WHITE) == white_is_bot1 else bot2
                    
                    move = current_bot.get_move(board, self.book)
                    
                    if move == chess.Move.null() or move not in board.legal_moves:
                        print(f"Invalid move from {current_bot.name}")
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
                    if white_is_bot1:
                        bot1_wins += 1
                    else:
                        bot2_wins += 1
                elif result == '0-1':
                    if white_is_bot1:
                        bot2_wins += 1
                    else:
                        bot1_wins += 1
                elif result == '1/2-1/2':
                    draws += 1
                
                win_history.append((bot1_wins, bot2_wins, draws))
                
                if wandb.run is not None:
                    wandb.log({
                        'game_idx': game_idx,
                        'result': result,
                        'duration': game_duration,
                        'moves': moves_played,
                        'bot1_wins': bot1_wins,
                        'bot2_wins': bot2_wins,
                        'draws': draws
                    })
                
                if self.switch_colors:
                    white_is_bot1 = not white_is_bot1
            
            avg_duration = float(np.mean(durations)) if durations else 0
            avg_moves = float(np.mean(move_counts)) if move_counts else 0
            
            print("\nBenchmark Results:")
            print(f"Bot1 ({bot1.name}) wins: {bot1_wins}")
            print(f"Bot2 ({bot2.name}) wins: {bot2_wins}")
            print(f"Draws: {draws}")
            print(f"Unfinished games: {results['*']}")
            print(f"Average game duration: {avg_duration:.2f} seconds")
            print(f"Average moves per game: {avg_moves:.1f}")
            
            if wandb.run is not None:
                wandb.log({
                    'total_games': self.games,
                    'bot1_wins': bot1_wins,
                    'bot2_wins': bot2_wins,
                    'draws': draws,
                    'unfinished': results['*'],
                    'avg_duration': avg_duration,
                    'avg_moves': avg_moves
                })
                
                results_table = wandb.Table(
                    data=[
                        ['1-0', results['1-0']],
                        ['0-1', results['0-1']],
                        ['1/2-1/2', results['1/2-1/2']],
                        ['*', results['*']]
                    ],
                    columns=['Result', 'Count']
                )
                
                history_table = wandb.Table(
                    data=[(i+1, w[0], w[1], w[2]) for i, w in enumerate(win_history)],
                    columns=['Game', 'Bot1', 'Bot2', 'Draws']
                )
                
                wandb.log({
                    'results_dist': wandb.plot.bar(
                        results_table, 'Result', 'Count', title='Results Distribution'
                    ),
                    'win_history': wandb.plot.line(
                        history_table, 'Game', ['Bot1', 'Bot2', 'Draws'], 
                        title='Cumulative Results'
                    )
                })
                
                wandb.finish()
            
            summary_path = os.path.join(self.output_dir, 'benchmark_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Bot1 ({bot1.name}): {bot1_wins} wins\n")
                f.write(f"Bot2 ({bot2.name}): {bot2_wins} wins\n")
                f.write(f"Draws: {draws}\n")
                f.write(f"Unfinished: {results['*']}\n")
                f.write(f"Average duration: {avg_duration:.2f} seconds\n")
                f.write(f"Average moves: {avg_moves:.1f}\n")
            
            try:
                drive = get_drive()
                
                drive.save(
                    summary_path, 'benchmark_results/benchmark_summary.txt'
                )
                
                drive_dir = 'benchmark_results/games'
                for filename in os.listdir(self.games_dir):
                    if filename.endswith('.pgn'):
                        drive.save(
                            os.path.join(self.games_dir, filename),
                            os.path.join(drive_dir, filename)
                        )
                
                print(f"Saved benchmark results to Google Drive")
            except Exception as e:
                print(f"Error saving to Google Drive: {e}")
            
            return {
                'bot1_wins': bot1_wins,
                'bot2_wins': bot2_wins,
                'draws': draws,
                'unfinished': results['*']
            }
            
        except Exception as e:
            print(f"Error during benchmark: {e}")
            if wandb.run is not None:
                wandb.finish()
            return False