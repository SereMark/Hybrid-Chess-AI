import os, pickle, threading, zlib, chess
from typing import Dict, List

class OpeningBook:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(OpeningBook, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if not self._initialized:
            self.opening_sequences: Dict[str, List[str]] = {}
            self.opening_positions: Dict[str, List[str]] = {}
            self._initialized = True
            self._load_complete = False

    def start_loading(self, data_file: str) -> None:
        def load_thread():
            try:
                self._load_data(data_file)
                self._load_complete = True
            except Exception as e:
                print(f"Error loading opening book data: {e}")
                self._load_complete = True

        thread = threading.Thread(target=load_thread)
        thread.daemon = True
        thread.start()

    def is_loaded(self) -> bool:
        return self._load_complete

    def _load_data(self, data_file_path: str) -> None:
        if not os.path.exists(data_file_path):
            print(f"Opening book data file not found: {data_file_path}")
            self._load_complete = True
            return

        try:
            with open(data_file_path, 'rb') as f:
                compressed_data = f.read()
                data = pickle.loads(zlib.decompress(compressed_data))
                self.opening_sequences = data.get('sequences', {})
                self.opening_positions = data.get('positions', {})
        except Exception as e:
            print(f"Error loading opening book data: {e}")
            self._load_complete = True

    def get_opening_name(self, board: chess.Board) -> str:
        move_sequence = []
        temp_board = chess.Board()
        for move in board.move_stack:
            san = temp_board.san(move)
            move_sequence.append(san)
            temp_board.push(move)
        
        max_length = min(len(move_sequence), 10)
        for length in reversed(range(1, max_length + 1)):
            sub_sequence = ' '.join(move_sequence[:length])
            if sub_sequence in self.opening_sequences:
                opening_names = self.opening_sequences[sub_sequence]
                if opening_names:
                    return opening_names[0]
        return ""

    def get_opening_moves(self, board: chess.Board) -> List[chess.Move]:
        if not self._load_complete:
            return []
        fen = board.fen()
        move_uci_list = self.opening_positions.get(fen, [])
        moves = []
        for move_uci in move_uci_list:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                moves.append(move)
        return moves

    def is_position_in_opening_book(self, board: chess.Board) -> bool:
        if not self._load_complete:
            return False
        fen = board.fen()
        return fen in self.opening_positions