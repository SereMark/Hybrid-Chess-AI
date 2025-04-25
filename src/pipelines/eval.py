import os
import time
import json
import wandb
import torch
import chess
import signal
import chess.pgn
import subprocess
import numpy as np
import chess.engine
from tqdm.auto import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.model import ChessModel
from src.utils.config import Config
from src.utils.train import set_seed, get_device
from src.utils.chess import H5Dataset, get_move_count, BoardHistory, board_to_input, get_move_map

class StockfishManager:
    _instance = None
    _stockfish_path = None
    _is_initialized = False

    @classmethod
    def setup_stockfish(cls, force_reinstall=False):
        if cls._is_initialized and cls._stockfish_path and not force_reinstall:
            return True, cls._stockfish_path
            
        common_paths = [
            "/usr/games/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "stockfish",
            "./stockfish"
        ]
        
        env_path = os.environ.get("STOCKFISH_PATH")
        if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            cls._stockfish_path = env_path
            cls._is_initialized = True
            return True, cls._stockfish_path
            
        for p in common_paths:
            if os.path.isfile(p) and os.access(p, os.X_OK):
                cls._stockfish_path = p
                cls._is_initialized = True
                os.environ["STOCKFISH_PATH"] = p
                return True, cls._stockfish_path
                
        try:
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "stockfish"], check=True)
        except subprocess.CalledProcessError as e:
            cls._is_initialized = False
            return False, None
            
        for p in ["/usr/games/stockfish", "/usr/bin/stockfish"]:
            if os.path.isfile(p) and os.access(p, os.X_OK):
                cls._stockfish_path = p
                cls._is_initialized = True
                os.environ["STOCKFISH_PATH"] = p
                return True, cls._stockfish_path
                
        try:
            result = subprocess.run(
                ["which", "stockfish"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            path = result.stdout.strip()
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                cls._stockfish_path = path
                cls._is_initialized = True
                os.environ["STOCKFISH_PATH"] = path
                return True, cls._stockfish_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        cls._is_initialized = False
        return False, None
    
    @classmethod
    def get_stockfish_path(cls):
        if not cls._is_initialized:
            success, path = cls.setup_stockfish()
            if not success:
                return None
            
        return cls._stockfish_path
    
    @classmethod
    def verify_stockfish_working(cls):
        path = cls.get_stockfish_path()
        if not path:
            return False
            
        try:
            process = subprocess.Popen(
                [path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            process.stdin.write("uci\n")
            process.stdin.flush()
            
            lines = []
            for _ in range(10):
                line = process.stdout.readline().strip()
                lines.append(line)
                if "uciok" in line:
                    process.terminate()
                    return True
                    
            process.terminate()
            return False
            
        except Exception:
            return False

class EvalPipeline:
    def __init__(self, config: Config):
        self.config = config
        
        seed = config.get('project.seed', 42)
        set_seed(seed)
        
        self.device_info = get_device()
        self.device = self.device_info["device"]
        self.device_type = self.device_info["type"]
        
        root_dir = config.get('project.root_dir', '/content/drive/MyDrive/chess_ai')
        
        self.sl_model_path = os.path.join(root_dir, 'models/supervised_model.pth')
        self.rl_model_path = os.path.join(root_dir, 'models/reinforcement_model.pth')
        
        self.dataset = config.get('data.dataset', os.path.join(root_dir, 'data/dataset.h5'))
        self.test_idx = config.get('data.test_idx', os.path.join(root_dir, 'data/test_indices.npy'))
        
        self.max_eval_samples = config.get('eval.max_samples', 1000)
        self.visualize_moves = config.get('eval.visualize_moves', True)
        self.sl_vs_rl = config.get('eval.sl_vs_rl', True)
        
        self.benchmark_games = config.get('benchmark.games', 5)
        self.mcts = config.get('benchmark.mcts', True)
        self.opening_book = config.get('benchmark.opening_book', True)
        self.switch_colors = config.get('benchmark.switch_colors', True)
        self.stockfish_path = config.get('benchmark.stockfish_path', 'stockfish')
        self.stockfish_elo = config.get('benchmark.stockfish_elo', 1500)
        self.stockfish_time = config.get('benchmark.stockfish_time', 0.1)
        self.stockfish_depth = config.get('benchmark.stockfish_depth', None)
        
        self.output_dir = os.path.join(root_dir, 'evaluation')
        self.sl_eval_dir = os.path.join(self.output_dir, 'supervised')
        self.rl_eval_dir = os.path.join(self.output_dir, 'reinforcement')
        self.games_dir = os.path.join(self.output_dir, 'games')
        self.comparison_dir = os.path.join(self.output_dir, 'comparison')
        
        for directory in [self.output_dir, self.sl_eval_dir, self.rl_eval_dir, self.games_dir, self.comparison_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.sl_model = None
        self.rl_model = None
        self.sl_available = os.path.exists(self.sl_model_path)
        self.rl_available = os.path.exists(self.rl_model_path)
        
        self.book = {}
        book_path = os.path.join(root_dir, 'data/opening_book.json')
        if os.path.exists(book_path):
            try:
                with open(book_path, 'r') as f:
                    self.book = json.load(f)
            except Exception:
                self.book = {}
                
        self.stockfish_available = False
        
    def setup_stockfish_for_colab(self):
        success, stockfish_path = StockfishManager.setup_stockfish()
        if success and stockfish_path:
            self.stockfish_path = stockfish_path
            self.stockfish_available = True
        return success
    
    def setup(self):
        if not self.sl_available and not self.rl_available:
            return False
        
        stockfish_ok = self.setup_stockfish_for_colab()
        if not stockfish_ok:
            self.stockfish_available = False
        
        if self.config.get('wandb.enabled', True):
            try:
                wandb.init(
                    project=self.config.get('wandb.project', 'chess_ai'),
                    name=f"eval_{self.config.mode}_{time.strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "mode": self.config.mode,
                        "sl_model_available": self.sl_available,
                        "rl_model_available": self.rl_available,
                        "benchmark_games": self.benchmark_games,
                        "stockfish_elo": self.stockfish_elo,
                        "mcts_enabled": self.mcts,
                        "device": self.device_type
                    }
                )
            except Exception:
                pass
                
        return True
        
    def load_model(self, model_path, model_type="unknown"):
        if not os.path.isfile(model_path):
            return None
            
        try:
            ch = self.config.get('model.channels', 64)
            blocks = self.config.get('model.blocks', 4)
            use_attention = self.config.get('model.attention', True)
            
            model = ChessModel(
                moves=get_move_count(),
                ch=ch,
                blocks=blocks,
                use_attn=use_attention
            ).to(self.device)
            
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
            except Exception:
                checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
                
            model.eval()
            return model
            
        except Exception:
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_data(self):
        if not os.path.isfile(self.test_idx) or not os.path.isfile(self.dataset):
            return None
            
        try:
            test_indices = np.load(self.test_idx)
            
            if self.max_eval_samples > 0 and len(test_indices) > self.max_eval_samples:
                np.random.shuffle(test_indices)
                test_indices = test_indices[:self.max_eval_samples]
            
            test_dataset = H5Dataset(self.dataset, test_indices)
            
            batch = self.config.get('data.batch', 128)
            workers = self.config.get('hardware.workers', 2)
            pin_memory = self.config.get('hardware.pin_memory', True)
            prefetch = self.config.get('hardware.prefetch', 2)
            
            dataloader = DataLoader(
                test_dataset,
                batch_size=batch,
                shuffle=False,
                num_workers=workers,
                pin_memory=pin_memory,
                persistent_workers=(workers > 0),
                prefetch_factor=prefetch if workers > 0 else None
            )
            
            return dataloader
            
        except Exception:
            return None
    
    def run_inference(self, model, dataloader, model_type="unknown"):
        predictions, actuals, accuracies, logits = [], [], [], []
        
        total_batches = len(dataloader)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, values) in enumerate(tqdm(dataloader)):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = model(inputs)
                policy_output = outputs[0]
                
                pred = policy_output.argmax(dim=1)
                accuracy = (pred == targets).float().mean().item()
                
                predictions.extend(pred.cpu().numpy().tolist())
                actuals.extend(targets.cpu().numpy().tolist())
                accuracies.append(accuracy)
                logits.append(policy_output.cpu())
                
                if wandb.run is not None and (batch_idx + 1) % 10 == 0:
                    wandb.log({
                        f'{model_type}/batch_idx': batch_idx + 1,
                        f'{model_type}/batch_accuracy': accuracy
                    })
        
        logits_array = torch.cat(logits, 0).numpy() if logits else np.empty((0,))
        
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return predictions_array, actuals_array, accuracies, logits_array
    
    def evaluate_model_accuracy(self, model, model_type, output_dir):
        if model is None:
            return None
            
        dataloader = self.prepare_data()
        if dataloader is None:
            return None
            
        predictions, actuals, accuracies, logits = self.run_inference(model, dataloader, model_type)
        
        if len(predictions) == 0:
            return None
            
        overall_accuracy = float(np.mean(predictions == actuals))
        avg_batch_accuracy = float(np.mean(accuracies))
        
        scores = {
            f'{model_type}/accuracy': overall_accuracy,
            f'{model_type}/batch_accuracy': avg_batch_accuracy
        }
        
        if wandb.run is not None:
            wandb.log(scores)
            
        unique_classes = np.unique(np.concatenate([actuals, predictions]))
        if len(unique_classes) <= 20:
            cm = metrics.confusion_matrix(actuals, predictions, labels=unique_classes)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm, cmap='Blues')
            ax.figure.colorbar(im, ax=ax)
            
            ax.set(
                xticks=np.arange(len(unique_classes)),
                yticks=np.arange(len(unique_classes)),
                xticklabels=[str(c) for c in unique_classes],
                yticklabels=[str(c) for c in unique_classes],
                ylabel='True',
                xlabel='Predicted',
                title=f'{model_type} Model Confusion Matrix'
            )
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            threshold = cm.max() / 2
            for i in range(len(unique_classes)):
                for j in range(len(unique_classes)):
                    ax.text(
                        j, i, cm[i, j],
                        ha="center", va="center",
                        color="white" if cm[i, j] > threshold else "black"
                    )
            
            fig.tight_layout()
            
            fig_path = os.path.join(output_dir, 'confusion_matrix.png')
            fig.savefig(fig_path)
            
            if wandb.run is not None:
                wandb.log({f'{model_type}/confusion_matrix': wandb.Image(fig)})
            
            plt.close(fig)
            
            class_accuracy = cm.diagonal() / cm.sum(axis=1, where=cm.sum(axis=1) != 0)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(unique_classes)), class_accuracy)
            ax.set(
                xlabel='Class',
                ylabel='Accuracy',
                title=f'{model_type} Model Per-class Accuracy',
                xticks=range(len(unique_classes)),
                xticklabels=[str(c) for c in unique_classes]
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
            
            fig_path = os.path.join(output_dir, 'per_class_accuracy.png')
            fig.savefig(fig_path)
            
            if wandb.run is not None:
                wandb.log({f'{model_type}/per_class_accuracy': wandb.Image(fig)})
            
            plt.close(fig)
        
        summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"{model_type} Model Evaluation Summary\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Overall accuracy: {overall_accuracy:.4f}\n")
            f.write(f"Average batch accuracy: {avg_batch_accuracy:.4f}\n")
            
        return {
            'accuracy': overall_accuracy,
            'batch_accuracy': avg_batch_accuracy,
            'predictions': predictions,
            'actuals': actuals,
            'logits': logits
        }
    
    def visualize_move_comparison(self, sl_eval_results, rl_eval_results):
        if sl_eval_results is None or rl_eval_results is None:
            return
            
        try:
            max_samples = min(1000, len(sl_eval_results['predictions']))
            
            sample_indices = np.random.choice(
                len(sl_eval_results['predictions']), 
                size=max_samples, 
                replace=False
            )
            
            sl_preds = sl_eval_results['predictions'][sample_indices]
            rl_preds = rl_eval_results['predictions'][sample_indices]
            actuals = sl_eval_results['actuals'][sample_indices]
            
            agreement = np.mean(sl_preds == rl_preds)
            sl_accuracy = np.mean(sl_preds == actuals)
            rl_accuracy = np.mean(rl_preds == actuals)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sl_correct = sl_preds == actuals
            rl_correct = rl_preds == actuals
            
            confusion_data = np.zeros((2, 2))
            confusion_data[0, 0] = np.sum(sl_correct & rl_correct)
            confusion_data[0, 1] = np.sum(sl_correct & ~rl_correct)
            confusion_data[1, 0] = np.sum(~sl_correct & rl_correct)
            confusion_data[1, 1] = np.sum(~sl_correct & ~rl_correct)
            
            im = ax.imshow(confusion_data, cmap='Blues')
            ax.figure.colorbar(im, ax=ax)
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['RL Correct', 'RL Incorrect'])
            ax.set_yticklabels(['SL Correct', 'SL Incorrect'])
            ax.set_title('SL vs RL Move Agreement Matrix')
            
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{confusion_data[i, j]:.0f}",
                           ha="center", va="center", color="black")
            
            fig_path = os.path.join(self.comparison_dir, 'sl_rl_agreement.png')
            fig.savefig(fig_path)
            
            if wandb.run is not None:
                wandb.log({'comparison/sl_rl_agreement': wandb.Image(fig)})
            
            plt.close(fig)
            
            move_diff = {}
            for sl, rl, actual in zip(sl_preds, rl_preds, actuals):
                if sl != rl:
                    sl_correct = sl == actual
                    rl_correct = rl == actual
                    
                    if sl_correct and not rl_correct:
                        if sl not in move_diff:
                            move_diff[sl] = 0
                        move_diff[sl] += 1
                    elif rl_correct and not sl_correct:
                        if rl not in move_diff:
                            move_diff[rl] = 0
                        move_diff[rl] += 1
            
            top_moves = sorted(move_diff.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if top_moves:
                moves = [str(m[0]) for m in top_moves]
                counts = [m[1] for m in top_moves]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(moves, counts)
                ax.set_xlabel('Move')
                ax.set_ylabel('Correct prediction advantage count')
                ax.set_title('Moves Where Models Disagree (and one is correct)')
                plt.xticks(rotation=45)
                fig.tight_layout()
                
                fig_path = os.path.join(self.comparison_dir, 'move_disagreement.png')
                fig.savefig(fig_path)
                
                if wandb.run is not None:
                    wandb.log({'comparison/move_disagreement': wandb.Image(fig)})
                
                plt.close(fig)
            
            summary_path = os.path.join(self.comparison_dir, 'comparison_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("SL vs RL Model Comparison Summary\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"SL-RL move agreement: {agreement:.4f}\n")
                f.write(f"SL accuracy: {sl_accuracy:.4f}\n")
                f.write(f"RL accuracy: {rl_accuracy:.4f}\n\n")
                f.write("Model Strengths:\n")
                f.write(f"Cases where SL is correct but RL is wrong: {confusion_data[0, 1]:.0f}\n")
                f.write(f"Cases where RL is correct but SL is wrong: {confusion_data[1, 0]:.0f}\n")
            
            if wandb.run is not None:
                wandb.log({
                    'comparison/sl_rl_agreement_rate': agreement,
                    'comparison/sl_accuracy': sl_accuracy,
                    'comparison/rl_accuracy': rl_accuracy,
                    'comparison/sl_advantage_cases': confusion_data[0, 1],
                    'comparison/rl_advantage_cases': confusion_data[1, 0]
                })
                
        except Exception:
            import traceback
            traceback.print_exc()
    
    class ChessBot:
        def __init__(self, model, device, use_mcts=True, use_book=True, name="ChessAI"):
            self.model = model
            self.device = device
            self.name = name
            self.use_mcts = use_mcts
            self.use_book = use_book
            
            self.move_map = get_move_map()
            
            if self.use_mcts:
                try:
                    from src.utils.mcts import MCTS
                    self.mcts = MCTS(
                        self.model, 
                        self.device, 
                        c_puct=1.4, 
                        n_sims=100
                    )
                except Exception:
                    self.mcts = None
                    self.use_mcts = False
            else:
                self.mcts = None
            
            self.board_history = BoardHistory(max_history=7)
        
        def reset_history(self, board):
            self.board_history = BoardHistory(max_history=7)
            self.board_history.add_board(board.copy())
        
        def get_move(self, board, book):
            if not self.board_history:
                self.reset_history(board)
            
            try:
                if self.use_book and book is not None:
                    fen = board.fen()
                    moves_dict = book.get(fen, {})
                    
                    if moves_dict:
                        best_move = None
                        best_score = -1
                        
                        for uci_move, stats in moves_dict.items():
                            total_games = stats.get('win', 0) + stats.get('draw', 0) + stats.get('loss', 0)
                            
                            if total_games > 0:
                                score = (stats.get('win', 0) + 0.5 * stats.get('draw', 0)) / total_games
                                try:
                                    move = chess.Move.from_uci(uci_move)
                                    
                                    if move in board.legal_moves and score > best_score:
                                        best_move, best_score = move, score
                                except ValueError:
                                    continue
                        
                        if best_move:
                            board_copy = board.copy()
                            board_copy.push(best_move)
                            self.board_history.add_board(board_copy)
                            return best_move
                
                if self.use_mcts and self.mcts:
                    self.mcts.set_root(board.copy())
                    action_probs = self.mcts.get_move_probs(temp=1e-3)
                    
                    if board.fullmove_number == 1 and board.turn == chess.WHITE and len(action_probs) > 1:
                        moves_list = list(action_probs.keys())
                        noise = np.random.dirichlet([0.3] * len(moves_list))
                        for i, move in enumerate(moves_list):
                            action_probs[move] = 0.75 * action_probs[move] + 0.25 * noise[i]
                    
                    if action_probs:
                        best_move = max(action_probs, key=action_probs.get)
                        board_copy = board.copy()
                        board_copy.push(best_move)
                        self.board_history.add_board(board_copy)
                        return best_move
                    else:
                        return chess.Move.null()
                
                try:
                    input_array = board_to_input(board, self.board_history)
                    input_tensor = torch.from_numpy(input_array).float().unsqueeze(0).to(self.device)
                    
                    del input_array
                    
                    with torch.no_grad():
                        policy_logits, _ = self.model(input_tensor)
                        policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
                    
                    del input_tensor
                    del policy_logits
                    
                    move_probs = {}
                    legal_moves_list = list(board.legal_moves)
                    
                    if len(legal_moves_list) > 20:
                        move_indices = []
                        for move in legal_moves_list:
                            move_idx = self.move_map.idx_by_move(move)
                            if move_idx is not None and move_idx < len(policy):
                                move_indices.append((move, move_idx, policy[move_idx]))
                        
                        move_indices.sort(key=lambda x: x[2], reverse=True)
                        top_moves = move_indices[:20]
                        
                        for move, move_idx, prob in top_moves:
                            move_probs[move] = prob
                    else:
                        for move in legal_moves_list:
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
                except Exception:
                    move_probs = {move: 1.0/len(list(board.legal_moves)) for move in board.legal_moves}
                
                best_move = max(move_probs, key=move_probs.get) if move_probs else chess.Move.null()
                
                if best_move != chess.Move.null():
                    board_copy = board.copy()
                    board_copy.push(best_move)
                    self.board_history.add_board(board_copy)
                    
                return best_move
            
            except Exception:
                return chess.Move.null()
    
    class StockfishBot:
        def __init__(self, stockfish_path=None, elo=1500, time_limit=0.1, depth=None, name="Stockfish"):
            self.name = name
            self.elo = elo
            self.time_limit = time_limit
            self.depth = depth
            self.stockfish_path = stockfish_path
            self.engine = None
            self.setup()
            self.process_id = None
        
        def setup(self):
            try:
                if not self.stockfish_path or not os.path.isfile(self.stockfish_path):
                    success, stockfish_path = StockfishManager.setup_stockfish()
                    if success and stockfish_path:
                        self.stockfish_path = stockfish_path
                    else:
                        self.stockfish_path = 'stockfish'
                
                try:
                    self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                    
                    if self.elo is not None and self.elo < 3000:
                        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})
                    
                    return True
                    
                except Exception:
                    self.engine = None
                    return False
                    
            except Exception:
                self.engine = None
                return False
        
        def get_move(self, board, _=None):
            if not self.engine:
                return chess.Move.null()
            
            try:
                import threading
                import time
                
                move_result = [None]
                move_exception = [None]
                
                def get_stockfish_move():
                    try:
                        if self.depth:
                            result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
                        else:
                            result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
                        move_result[0] = result.move
                    except Exception as e:
                        move_exception[0] = e
                
                move_thread = threading.Thread(target=get_stockfish_move)
                move_thread.daemon = True
                move_thread.start()
                
                max_wait = 5.0
                start_time = time.time()
                
                while move_thread.is_alive() and time.time() - start_time < max_wait:
                    time.sleep(0.1)
                
                if move_thread.is_alive():
                    try:
                        if self.engine and hasattr(self.engine, 'transport') and hasattr(self.engine.transport, 'proc'):
                            self.process_id = self.engine.transport.proc.pid
                            try:
                                os.kill(self.process_id, signal.SIGKILL)
                            except:
                                pass
                    except:
                        pass
                        
                    try:
                        if self.engine:
                            self.engine.quit()
                    except:
                        pass
                        
                    self.engine = None
                    
                    try:
                        self.engine = chess.engine.SimpleEngine.popen_uci(
                            self.stockfish_path,
                            timeout=10
                        )
                        if self.elo:
                            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})
                    except Exception:
                        pass
                    
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        import random
                        return random.choice(legal_moves)
                    return chess.Move.null()
                
                if move_exception[0] is not None:
                    return chess.Move.null()
                
                return move_result[0] if move_result[0] is not None else chess.Move.null()
            except Exception:
                return chess.Move.null()
        
        def close(self):
            try:
                if self.engine:
                    self.engine.quit()
                if self.process_id:
                    try:
                        os.kill(self.process_id, signal.SIGKILL)
                    except:
                        pass
            except:
                pass
    
    def run_sl_rl_comparison(self):
        if not self.sl_available or not self.rl_available:
            return None
            
        if self.sl_model is None or self.rl_model is None:
            if self.sl_model is None:
                self.sl_model = self.load_model(self.sl_model_path, "Supervised")
            if self.rl_model is None:
                self.rl_model = self.load_model(self.rl_model_path, "Reinforcement")
                
            if self.sl_model is None or self.rl_model is None:
                return None
        
        sl_bot = self.ChessBot(
            self.sl_model,
            self.device,
            use_mcts=self.mcts,
            use_book=self.opening_book,
            name="SupervisedAI"
        )
        
        rl_bot = self.ChessBot(
            self.rl_model,
            self.device,
            use_mcts=self.mcts,
            use_book=self.opening_book,
            name="ReinforcementAI"
        )
        
        num_games = min(self.benchmark_games, 5)
        results = {'sl_wins': 0, 'rl_wins': 0, 'draws': 0}
        game_records = []
        
        for game_idx in range(1, num_games + 1):
            board = chess.Board()
            game = chess.pgn.Game()
            
            sl_plays_white = game_idx % 2 == 1
            white_name = sl_bot.name if sl_plays_white else rl_bot.name
            black_name = rl_bot.name if sl_plays_white else sl_bot.name
            
            game.headers.update({
                'Event': 'SL vs RL Comparison',
                'Site': 'Colab',
                'Date': time.strftime('%Y.%m.%d'),
                'Round': str(game_idx),
                'White': white_name,
                'Black': black_name,
                'Result': '*'
            })
            
            sl_bot.reset_history(board)
            rl_bot.reset_history(board)
            
            node = game
            moves_played = 0
            
            while not board.is_game_over() and moves_played < 200:
                sl_turn = (board.turn == chess.WHITE and sl_plays_white) or \
                          (board.turn == chess.BLACK and not sl_plays_white)
                
                current_bot = sl_bot if sl_turn else rl_bot
                
                move = current_bot.get_move(board, self.book)
                
                if move == chess.Move.null() or move not in board.legal_moves:
                    break
                
                board.push(move)
                node = node.add_variation(move)
                moves_played += 1
            
            result = board.result()
            game.headers['Result'] = result
            
            if result == '1-0':
                if sl_plays_white:
                    results['sl_wins'] += 1
                else:
                    results['rl_wins'] += 1
            elif result == '0-1':
                if sl_plays_white:
                    results['rl_wins'] += 1
                else:
                    results['sl_wins'] += 1
            else:
                results['draws'] += 1
            
            pgn_path = os.path.join(self.comparison_dir, f'sl_vs_rl_game_{game_idx}.pgn')
            with open(pgn_path, 'w') as f:
                f.write(str(game))
            
            game_records.append({
                'game': game_idx,
                'sl_plays_white': sl_plays_white,
                'result': result,
                'moves': moves_played
            })
        
        total_games = num_games
        sl_win_rate = results['sl_wins'] / total_games if total_games > 0 else 0
        rl_win_rate = results['rl_wins'] / total_games if total_games > 0 else 0
        draw_rate = results['draws'] / total_games if total_games > 0 else 0
        
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['SL Wins', 'RL Wins', 'Draws']
        values = [results['sl_wins'], results['rl_wins'], results['draws']]
        ax.bar(labels, values)
        ax.set_ylabel('Number of Games')
        ax.set_title('SL vs RL Match Results')
        
        fig_path = os.path.join(self.comparison_dir, 'sl_vs_rl_results.png')
        fig.savefig(fig_path)
        
        if wandb.run is not None:
            wandb.log({
                'sl_vs_rl/sl_wins': results['sl_wins'],
                'sl_vs_rl/rl_wins': results['rl_wins'],
                'sl_vs_rl/draws': results['draws'],
                'sl_vs_rl/sl_win_rate': sl_win_rate,
                'sl_vs_rl/rl_win_rate': rl_win_rate,
                'sl_vs_rl/draw_rate': draw_rate,
                'sl_vs_rl/results_chart': wandb.Image(fig)
            })
        
        plt.close(fig)
        
        summary_path = os.path.join(self.comparison_dir, 'sl_vs_rl_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("SL vs RL Game Results\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total games: {total_games}\n")
            f.write(f"SL wins: {results['sl_wins']} ({sl_win_rate:.1%})\n")
            f.write(f"RL wins: {results['rl_wins']} ({rl_win_rate:.1%})\n")
            f.write(f"Draws: {results['draws']} ({draw_rate:.1%})\n\n")
            f.write("Game records:\n")
            for game in game_records:
                white = "SL" if game['sl_plays_white'] else "RL"
                black = "RL" if game['sl_plays_white'] else "SL"
                f.write(f"Game {game['game']}: {white} vs {black} - Result: {game['result']} - Moves: {game['moves']}\n")
        
        return results
    
    def run_stockfish_benchmark(self, model, model_type):
        if model is None or not self.stockfish_available:
            return None
        
        model_bot = self.ChessBot(
            model,
            self.device,
            use_mcts=self.mcts,
            use_book=self.opening_book,
            name=f"{model_type}AI"
        )
        
        stockfish = self.StockfishBot(
            stockfish_path=self.stockfish_path,
            elo=self.stockfish_elo,
            time_limit=self.stockfish_time,
            depth=self.stockfish_depth,
            name="Stockfish"
        )
        
        if stockfish.engine is None:
            return None
        
        results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, '*': 0}
        durations = []
        move_counts = []
        ai_wins = 0
        stockfish_wins = 0
        draws = 0
        game_records = []
        
        model_dir = os.path.join(self.games_dir, model_type.lower())
        os.makedirs(model_dir, exist_ok=True)
        
        ai_plays_white = True
        
        for game_idx in range(1, self.benchmark_games + 1):
            game_start = time.time()
            
            board = chess.Board()
            game = chess.pgn.Game()
            
            white_name = model_bot.name if ai_plays_white else stockfish.name
            black_name = stockfish.name if ai_plays_white else model_bot.name
            
            game.headers.update({
                'Event': f'{model_type} vs Stockfish Benchmark',
                'Site': 'Colab',
                'Date': time.strftime('%Y.%m.%d'),
                'Round': str(game_idx),
                'White': white_name,
                'Black': black_name,
                'Result': '*'
            })
            
            model_bot.reset_history(board)
            
            node = game
            moves_played = 0
            
            while not board.is_game_over() and moves_played < 200:
                ai_turn = (board.turn == chess.WHITE and ai_plays_white) or \
                          (board.turn == chess.BLACK and not ai_plays_white)
                
                current_player = model_bot if ai_turn else stockfish
                
                move = current_player.get_move(board, self.book)
                
                if move == chess.Move.null() or move not in board.legal_moves:
                    break
                
                board.push(move)
                node = node.add_variation(move)
                moves_played += 1
            
            result = board.result()
            results[result] += 1
            
            game.headers['Result'] = result
            
            pgn_path = os.path.join(model_dir, f'game_{game_idx}.pgn')
            with open(pgn_path, 'w') as f:
                f.write(str(game))
            
            game_duration = time.time() - game_start
            durations.append(game_duration)
            move_counts.append(moves_played)
            
            if result == '1-0':
                if ai_plays_white:
                    ai_wins += 1
                else:
                    stockfish_wins += 1
            elif result == '0-1':
                if ai_plays_white:
                    stockfish_wins += 1
                else:
                    ai_wins += 1
            elif result == '1/2-1/2':
                draws += 1
            
            game_records.append({
                'game': game_idx,
                'ai_plays_white': ai_plays_white,
                'result': result,
                'moves': moves_played,
                'duration': game_duration
            })
            
            if wandb.run is not None:
                wandb.log({
                    f'{model_type}_stockfish/game_idx': game_idx,
                    f'{model_type}_stockfish/result': result,
                    f'{model_type}_stockfish/duration': game_duration,
                    f'{model_type}_stockfish/moves': moves_played,
                    f'{model_type}_stockfish/ai_wins': ai_wins,
                    f'{model_type}_stockfish/stockfish_wins': stockfish_wins,
                    f'{model_type}_stockfish/draws': draws,
                    f'{model_type}_stockfish/ai_plays_white': ai_plays_white
                })
            
            if self.switch_colors:
                ai_plays_white = not ai_plays_white
        
        stockfish.close()
        
        total_games = ai_wins + stockfish_wins + draws
        win_rate = ai_wins / total_games if total_games > 0 else 0
        avg_duration = float(np.mean(durations)) if durations else 0
        avg_moves = float(np.mean(move_counts)) if move_counts else 0
        
        try:
            if win_rate > 0 and win_rate < 1:
                elo_diff = 400 * np.log10(win_rate / (1 - win_rate))
            elif win_rate == 1:
                elo_diff = 400
            elif win_rate == 0:
                elo_diff = -400
            else:
                elo_diff = 0
                
            estimated_elo = self.stockfish_elo + elo_diff
            
            try:
                if total_games > 0:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    labels = [f'{model_type}', 'Stockfish', 'Draws']
                    values = [ai_wins, stockfish_wins, draws]
                    ax.bar(labels, values)
                    ax.set_ylabel('Number of Games')
                    ax.set_title(f'{model_type} vs Stockfish (ELO {self.stockfish_elo})')
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            estimated_elo = self.stockfish_elo
        
        fig_path = os.path.join(model_dir, 'stockfish_results.png')
        fig.savefig(fig_path)
        
        if wandb.run is not None:
            wandb.log({
                f'{model_type}_stockfish/total_games': self.benchmark_games,
                f'{model_type}_stockfish/ai_wins_total': ai_wins,
                f'{model_type}_stockfish/stockfish_wins_total': stockfish_wins,
                f'{model_type}_stockfish/draws_total': draws,
                f'{model_type}_stockfish/win_rate': win_rate,
                f'{model_type}_stockfish/estimated_elo': estimated_elo,
                f'{model_type}_stockfish/avg_duration': avg_duration,
                f'{model_type}_stockfish/avg_moves': avg_moves,
                f'{model_type}_stockfish/results_chart': wandb.Image(fig)
            })
        
        plt.close(fig)
        
        summary_path = os.path.join(model_dir, 'stockfish_benchmark_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"{model_type} Model vs Stockfish Benchmark\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Model: {model_type}AI\n")
            f.write(f"Stockfish ELO: {self.stockfish_elo}\n")
            f.write(f"Games played: {self.benchmark_games}\n\n")
            f.write(f"{model_type} AI wins: {ai_wins}\n")
            f.write(f"Stockfish wins: {stockfish_wins}\n")
            f.write(f"Draws: {draws}\n")
            f.write(f"Win rate: {win_rate:.2%}\n")
            f.write(f"Estimated ELO: {estimated_elo:.0f}\n\n")
            f.write(f"Average duration: {avg_duration:.2f} seconds\n")
            f.write(f"Average moves: {avg_moves:.1f}\n\n")
            f.write("Game records:\n")
            for game in game_records:
                white = f"{model_type}AI" if game['ai_plays_white'] else "Stockfish"
                black = "Stockfish" if game['ai_plays_white'] else f"{model_type}AI"
                f.write(f"Game {game['game']}: {white} vs {black} - Result: {game['result']} - Moves: {game['moves']} - Duration: {game['duration']:.2f}s\n")
        
        return {
            'ai_wins': ai_wins,
            'stockfish_wins': stockfish_wins,
            'draws': draws,
            'win_rate': win_rate,
            'estimated_elo': estimated_elo
        }
    
    def run(self):
        if not self.setup():
            return False
        
        try:
            eval_results = {
                'models_available': {
                    'supervised': self.sl_available,
                    'reinforcement': self.rl_available
                }
            }
            
            if self.sl_available:
                self.sl_model = self.load_model(self.sl_model_path, "Supervised")
            
            if self.rl_available:
                self.rl_model = self.load_model(self.rl_model_path, "Reinforcement")
            
            sl_eval_results = None
            if self.sl_available and self.sl_model:
                sl_eval_results = self.evaluate_model_accuracy(
                    self.sl_model, "supervised", self.sl_eval_dir
                )
                eval_results['supervised'] = sl_eval_results
            
            rl_eval_results = None
            if self.rl_available and self.rl_model:
                rl_eval_results = self.evaluate_model_accuracy(
                    self.rl_model, "reinforcement", self.rl_eval_dir
                )
                eval_results['reinforcement'] = rl_eval_results
            
            if self.sl_available and self.rl_available and self.sl_model and self.rl_model:
                if self.visualize_moves and sl_eval_results and rl_eval_results:
                    self.visualize_move_comparison(sl_eval_results, rl_eval_results)
                
                if self.sl_vs_rl:
                    sl_rl_results = self.run_sl_rl_comparison()
                    eval_results['sl_vs_rl'] = sl_rl_results
            
            if self.stockfish_available:
                if self.sl_available and self.sl_model:
                    sl_benchmark = self.run_stockfish_benchmark(self.sl_model, "supervised")
                    eval_results['supervised_benchmark'] = sl_benchmark
                
                if self.rl_available and self.rl_model:
                    rl_benchmark = self.run_stockfish_benchmark(self.rl_model, "reinforcement")
                    eval_results['reinforcement_benchmark'] = rl_benchmark
            
            self.generate_final_report(eval_results)
            
            if wandb.run is not None:
                wandb.finish()
            
            return True
            
        except Exception:
            import traceback
            traceback.print_exc()
            
            if wandb.run is not None:
                wandb.finish()
            
            return False
    
    def generate_final_report(self, results):
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=======================================\n")
            f.write("CHESS AI EVALUATION REPORT\n")
            f.write("=======================================\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MODELS AVAILABLE:\n")
            f.write(f"Supervised Learning: {'Yes' if results['models_available']['supervised'] else 'No'}\n")
            f.write(f"Reinforcement Learning: {'Yes' if results['models_available']['reinforcement'] else 'No'}\n\n")
            
            if 'supervised' in results:
                f.write("SUPERVISED MODEL EVALUATION:\n")
                f.write(f"Accuracy: {results['supervised']['accuracy']:.4f}\n")
                f.write(f"Batch Accuracy: {results['supervised']['batch_accuracy']:.4f}\n\n")
            
            if 'reinforcement' in results:
                f.write("REINFORCEMENT MODEL EVALUATION:\n")
                f.write(f"Accuracy: {results['reinforcement']['accuracy']:.4f}\n")
                f.write(f"Batch Accuracy: {results['reinforcement']['batch_accuracy']:.4f}\n\n")
            
            if 'sl_vs_rl' in results:
                f.write("SL VS RL COMPARISON:\n")
                f.write(f"SL Wins: {results['sl_vs_rl']['sl_wins']}\n")
                f.write(f"RL Wins: {results['sl_vs_rl']['rl_wins']}\n")
                f.write(f"Draws: {results['sl_vs_rl']['draws']}\n\n")
            
            if 'supervised_benchmark' in results and results['supervised_benchmark']:
                f.write("SUPERVISED MODEL VS STOCKFISH:\n")
                f.write(f"Stockfish ELO: {self.stockfish_elo}\n")
                f.write(f"AI Wins: {results['supervised_benchmark']['ai_wins']}\n")
                f.write(f"Stockfish Wins: {results['supervised_benchmark']['stockfish_wins']}\n")
                f.write(f"Draws: {results['supervised_benchmark']['draws']}\n")
                f.write(f"Win Rate: {results['supervised_benchmark']['win_rate']:.2%}\n")
                f.write(f"Estimated ELO: {results['supervised_benchmark']['estimated_elo']:.0f}\n\n")
            
            if 'reinforcement_benchmark' in results and results['reinforcement_benchmark']:
                f.write("REINFORCEMENT MODEL VS STOCKFISH:\n")
                f.write(f"Stockfish ELO: {self.stockfish_elo}\n")
                f.write(f"AI Wins: {results['reinforcement_benchmark']['ai_wins']}\n")
                f.write(f"Stockfish Wins: {results['reinforcement_benchmark']['stockfish_wins']}\n")
                f.write(f"Draws: {results['reinforcement_benchmark']['draws']}\n")
                f.write(f"Win Rate: {results['reinforcement_benchmark']['win_rate']:.2%}\n")
                f.write(f"Estimated ELO: {results['reinforcement_benchmark']['estimated_elo']:.0f}\n\n")
            
            if 'supervised_benchmark' in results and 'reinforcement_benchmark' in results:
                if results['supervised_benchmark'] and results['reinforcement_benchmark']:
                    f.write("MODEL COMPARISON SUMMARY:\n")
                    sl_elo = results['supervised_benchmark']['estimated_elo']
                    rl_elo = results['reinforcement_benchmark']['estimated_elo']
                    elo_diff = rl_elo - sl_elo
                    
                    f.write(f"Supervised ELO: {sl_elo:.0f}\n")
                    f.write(f"Reinforcement ELO: {rl_elo:.0f}\n")
                    f.write(f"ELO Difference (RL-SL): {elo_diff:+.0f}\n")
                    
                    if elo_diff > 0:
                        f.write("Reinforcement learning improved model performance!\n")
                    elif elo_diff < 0:
                        f.write("Reinforcement learning did not improve model performance.\n")
                    else:
                        f.write("Reinforcement learning had no clear impact on model performance.\n")
        
        if 'supervised_benchmark' in results and 'reinforcement_benchmark' in results:
            if results['supervised_benchmark'] and results['reinforcement_benchmark']:
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    models = ['Supervised', 'Reinforcement']
                    win_rates = [
                        results['supervised_benchmark']['win_rate'] * 100,
                        results['reinforcement_benchmark']['win_rate'] * 100
                    ]
                    elos = [
                        results['supervised_benchmark']['estimated_elo'],
                        results['reinforcement_benchmark']['estimated_elo']
                    ]
                    
                    x = np.arange(len(models))
                    width = 0.35
                    
                    ax.bar(x - width/2, win_rates, width, label='Win Rate (%)')
                    ax.set_ylabel('Win Rate (%)')
                    ax.set_ylim(0, 100)
                    
                    ax2 = ax.twinx()
                    ax2.bar(x + width/2, elos, width, color='orange', label='Estimated ELO')
                    ax2.set_ylabel('Estimated ELO')
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(models)
                    ax.set_title('Model Performance Comparison')
                    
                    ax.legend(loc='upper left')
                    ax2.legend(loc='upper right')
                    
                    fig.tight_layout()
                    
                    fig_path = os.path.join(self.comparison_dir, 'model_performance_comparison.png')
                    fig.savefig(fig_path)
                    
                    if wandb.run is not None:
                        wandb.log({'comparison/model_performance': wandb.Image(fig)})
                    
                    plt.close(fig)
                    
                except Exception:
                    pass