import os
import json
import time
import torch
import chess
import random
import berserk
import logging
import threading
import chess.pgn
from src.training.reinforcement.mcts import MCTS
from src.utils.chess_utils import get_total_moves
from typing import Any, Callable, Dict, Optional, Tuple
from src.models.transformer import TransformerCNNChessModel
from src.utils.chess_utils import convert_board_to_transformer_input

logger = logging.getLogger("LichessBot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def load_opening_book(opening_book_path: str) -> Dict[str, Any]:
    try:
        with open(opening_book_path, 'r') as f:
            book_data = json.load(f)
        return book_data
    except Exception as e:
        logger.exception(f"Failed to load opening book from {opening_book_path}: {e}")
        raise

def choose_opening_move(board: chess.Board, opening_book: Dict[str, Any]) -> Optional[chess.Move]:
    fen = board.fen()
    moves_data = opening_book.get(fen)
    if moves_data:
        move_scores = {}
        for move_str, stats in moves_data.items():
            wins = stats.get("win", 0)
            draws = stats.get("draw", 0)
            losses = stats.get("loss", 0)
            total = wins + draws + losses
            if total > 0:
                score = ((wins + 0.5 * draws) + 10 * 0.5) / (total + 10)
            else:
                score = 0
            move_scores[move_str] = score
        best_score = max(move_scores.values())
        best_moves = [move for move, score in move_scores.items() if score == best_score]
        best_move_str = random.choice(best_moves)
        return chess.Move.from_uci(best_move_str)
    return None

def update_board_from_moves(moves_str: str) -> chess.Board:
    board = chess.Board()
    if moves_str:
        for move in moves_str.split():
            try:
                board.push_uci(move)
            except Exception as e:
                logger.exception(f"Failed to apply move {move}: {e}")
                raise
    return board

def determine_bot_color(game_info: Dict[str, Any], bot_id: str) -> Optional[bool]:
    if "color" in game_info:
        if game_info["color"].lower() == "white":
            return chess.WHITE
        elif game_info["color"].lower() == "black":
            return chess.BLACK
    def extract_id(player_info: Dict[str, Any]) -> Optional[str]:
        return player_info.get("id") or player_info.get("user", {}).get("id")
    players = game_info.get("players")
    if players:
        white_id = extract_id(players.get("white", {}))
        black_id = extract_id(players.get("black", {}))
        if white_id == bot_id:
            return chess.WHITE
        elif black_id == bot_id:
            return chess.BLACK
    else:
        white_info = game_info.get("white", {})
        black_info = game_info.get("black", {})
        white_id = extract_id(white_info)
        black_id = extract_id(black_info)
        if white_id == bot_id:
            return chess.WHITE
        elif black_id == bot_id:
            return chess.BLACK
    logger.warning(f"Could not determine bot color. Full game info: {json.dumps(game_info, indent=2)}")
    return None

class LichessBotDeploymentWorker:
    def __init__(self, model_path: str, opening_book_path: str, lichess_token: str,
                 time_control: str, rating_range: Tuple[int, int], use_mcts: bool,
                 mcts_simulations: Optional[int], mcts_c_puct: Optional[float],
                 auto_resign: bool, save_game_logs: bool, enable_model_eval_fallback: bool,
                 wandb_flag: bool, progress_callback: Callable[[int], None],
                 status_callback: Callable[[str], None]) -> None:
        self.model_path = model_path
        self.opening_book_path = opening_book_path
        self.lichess_token = lichess_token
        self.time_control = time_control
        self.rating_range = rating_range
        self.use_mcts = use_mcts
        self.mcts_simulations = mcts_simulations
        self.mcts_c_puct = mcts_c_puct
        self.auto_resign = auto_resign
        self.save_game_logs = save_game_logs
        self.enable_model_eval_fallback = enable_model_eval_fallback
        self.wandb_flag = wandb_flag
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.model: Optional[TransformerCNNChessModel] = None
        self.opening_book: Optional[Dict[str, Any]] = None
        self.mcts: Optional[MCTS] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bot_id: Optional[str] = None
        logger.setLevel(logging.INFO)
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        try:
            self.status_callback("Loading model...")
            logger.info(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model = TransformerCNNChessModel(num_moves=get_total_moves()).to(self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint: self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else: self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
            self.progress_callback(25)
            self.status_callback("Loading opening book...")
            logger.info(f"Loading opening book from {self.opening_book_path}")
            self.opening_book = load_opening_book(self.opening_book_path)
            self.progress_callback(50)
            if self.use_mcts:
                self.status_callback("Initializing MCTS...")
                self.mcts = MCTS(model=self.model, device=self.device, c_puct=self.mcts_c_puct, n_simulations=self.mcts_simulations)
                logger.info(f"MCTS initialized with {self.mcts_simulations} simulations and c_puct={self.mcts_c_puct:.2f}")
            else:
                self.mcts = None
            self.progress_callback(75)
            self.status_callback("Chess engine compiled successfully.")
            self.progress_callback(100)
        except Exception as e:
            error_msg = f"Engine initialization failed: {e}"
            self.status_callback(error_msg)
            logger.exception(error_msg)
            raise

    def start_bot(self) -> None:
        self.status_callback("Starting Lichess bot...")
        logger.info("Starting Lichess bot...")
        try:
            session = berserk.TokenSession(self.lichess_token)
            client = berserk.Client(session=session)
            account = client.account.get()
            self.bot_id = account.get('id')
            if not self.bot_id:
                raise ValueError("Bot ID not found in account information.")
            bot_profile_url = f"https://lichess.org/@/{self.bot_id}"
            self.status_callback(f"Logged in as {self.bot_id}. Challenge the bot at: {bot_profile_url}")
            logger.info(f"Logged in as {self.bot_id}. Challenge your bot at: {bot_profile_url}")
            while True:
                try:
                    for event in client.bots.stream_incoming_events():
                        event_type = event.get('type')
                        if event_type == 'challenge':
                            challenge = event.get('challenge')
                            if challenge:
                                self._handle_challenge_event(client, challenge)
                            else:
                                logger.warning("Received challenge event without challenge details.")
                        elif event_type == 'gameStart':
                            game_info = event.get('game')
                            if not game_info:
                                logger.warning("Received gameStart event without game info.")
                                continue
                            bot_color = determine_bot_color(game_info, self.bot_id)
                            if bot_color is None:
                                self.status_callback("Bot color could not be determined. Skipping game.")
                                logger.warning(f"Bot color could not be determined for game {game_info.get('id')}")
                                continue
                            game_thread = threading.Thread(target=self._play_game, args=(client, game_info, bot_color), daemon=True)
                            game_thread.start()
                        else:
                            logger.debug(f"Unhandled event type: {event_type}")
                except Exception as e:
                    logger.exception(f"Error in event stream: {e}")
                    self.status_callback(f"Streaming error: {e}. Reconnecting in 5 seconds...")
                    time.sleep(5)
        except Exception as e:
            error_msg = f"Error starting bot or streaming events: {e}"
            self.status_callback(error_msg)
            logger.exception(error_msg)
            raise

    def _handle_challenge_event(self, client: berserk.Client, challenge: Dict[str, Any]) -> None:
        try:
            challenger = challenge.get('challenger', {})
            challenger_rating = challenger.get('rating')
            if challenger_rating is None:
                logger.warning("Challenge missing rating information. Declining challenge.")
                client.bots.decline_challenge(challenge['id'])
                return
            if self.rating_range[0] <= challenger_rating <= self.rating_range[1]:
                client.bots.accept_challenge(challenge['id'])
                self.status_callback(f"Accepted challenge from rating {challenger_rating}. Playing game...")
                logger.info(f"Accepted challenge from rating {challenger_rating}")
            else:
                client.bots.decline_challenge(challenge['id'])
                self.status_callback(f"Declined challenge from rating {challenger_rating}.")
                logger.info(f"Declined challenge from rating {challenger_rating}")
        except Exception as e:
            logger.exception(f"Error handling challenge event: {e}")
            self.status_callback(f"Error handling challenge event: {e}")

    def _play_game(self, client: berserk.Client, game_info: Dict[str, Any], bot_color: bool) -> None:
        game_id = game_info.get('id')
        if not game_id:
            logger.error("Game information missing game ID.")
            return
        color_str = 'White' if bot_color == chess.WHITE else 'Black'
        logger.info(f"Starting game {game_id} as {color_str}")
        board = chess.Board()
        moves_str = ""
        try:
            for event in client.bots.stream_game_state(game_id):
                state_type = event.get('type')
                if state_type == 'gameFull':
                    moves_str = event.get('state', {}).get('moves', '')
                    board = update_board_from_moves(moves_str)
                    if board.turn == bot_color:
                        self._make_move(client, game_id, board)
                elif state_type == 'gameState':
                    moves_str = event.get('moves', '')
                    board = update_board_from_moves(moves_str)
                    if board.is_game_over():
                        result = board.result()
                        logger.info(f"Game {game_id} over with result {result}")
                        break
                    if board.turn == bot_color:
                        self._make_move(client, game_id, board)
                else:
                    logger.debug(f"Unhandled game state event type: {state_type}")
                if board.is_game_over():
                    result = board.result()
                    logger.info(f"Game {game_id} over with result {result}")
                    break
            if self.save_game_logs:
                self._save_game_pgn(game_id, moves_str)
        except Exception as e:
            logger.exception(f"Error during game {game_id}: {e}")

    def _make_move(self, client: berserk.Client, game_id: str, board: chess.Board) -> None:
        try:
            best_move: Optional[chess.Move] = None
            if self.opening_book:
                best_move = choose_opening_move(board, self.opening_book)
                if best_move:
                    logger.info(f"Using opening book move: {best_move.uci()}")
            if best_move is None and self.mcts:
                mcts_start = time.time()
                self.mcts.set_root_node(board)
                move_probs = self.mcts.get_move_probs()
                mcts_time = time.time() - mcts_start
                if move_probs:
                    best_move = max(move_probs, key=move_probs.get)
                    logger.info(f"MCTS selected move: {best_move.uci()} (in {mcts_time:.3f} sec)")
                else:
                    logger.warning("MCTS returned no moves.")
            if best_move is None and self.enable_model_eval_fallback and self.model is not None:
                eval_start = time.time()
                best_move = self._evaluate_moves(board)
                eval_time = time.time() - eval_start
                logger.info(f"Model evaluation selected move: {best_move.uci()} (in {eval_time:.3f} sec)")
            if best_move is None:
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    if self.auto_resign:
                        try:
                            client.bots.resign(game_id)
                            logger.info(f"Resigned game {game_id} due to no legal moves.")
                        except Exception as e:
                            logger.exception(f"Failed to resign game {game_id}: {e}")
                        return
                    else:
                        logger.warning(f"No legal moves available in game {game_id}")
                        return
                best_move = legal_moves[0]
                logger.info(f"Defaulted to first legal move: {best_move.uci()}")
            if best_move not in board.legal_moves:
                logger.error(f"Selected move {best_move.uci()} is not legal in game {game_id}.")
                return
            board.push(best_move)
            client.bots.make_move(game_id, best_move.uci())
            logger.info(f"Made move {best_move.uci()} in game {game_id}")
        except Exception as e:
            logger.exception(f"Failed to make move in game {game_id}: {e}")
            raise

    def _evaluate_moves(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        current_color = board.turn
        best_move = None
        best_score = -float('inf') if current_color == chess.WHITE else float('inf')
        move_evaluations = {}
        for move in legal_moves:
            board.push(move)
            score = self._evaluate_board(board)
            move_evaluations[move.uci()] = score
            board.pop()
            if current_color == chess.WHITE and score > best_score:
                best_score = score
                best_move = move
            elif current_color == chess.BLACK and score < best_score:
                best_score = score
                best_move = move
        return best_move if best_move is not None else legal_moves[0]

    def _evaluate_board(self, board: chess.Board) -> float:
        try:
            input_tensor = convert_board_to_transformer_input(board)
            input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                score = self.model(input_tensor)[1]
            return score.item()
        except Exception as e:
            logger.exception(f"Failed to evaluate board: {e}")
            return 0.0

    def _save_game_pgn(self, game_id: str, moves_str: str) -> None:
        try:
            pgn_game = chess.pgn.Game()
            node = pgn_game
            board = chess.Board()
            for move_uci in moves_str.split():
                move = chess.Move.from_uci(move_uci)
                node = node.add_variation(move)
                board.push(move)
            pgn_path = os.path.join("data", "games", "lichess", f"{game_id}.pgn")
            os.makedirs(os.path.dirname(pgn_path), exist_ok=True)
            with open(pgn_path, "w") as pgn_file:
                pgn_file.write(str(pgn_game))
            logger.info(f"Saved game {game_id} PGN log at {pgn_path}")
        except Exception as e:
            logger.exception(f"Failed to save PGN log for game {game_id}: {e}")

    def run(self) -> None:
        try:
            self.start_bot()
        except Exception as e:
            error_msg = f"Pipeline failed: {e}"
            self.status_callback(error_msg)
            logger.exception(error_msg)
            raise