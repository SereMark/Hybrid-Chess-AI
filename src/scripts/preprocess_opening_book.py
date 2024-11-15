import os, pickle, zlib, chess, multiprocessing
from typing import Dict, List, Tuple
from multiprocessing import Pool

def process_pgn_file(args: Tuple[str, int]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    file_path, max_depth = args
    opening_name = os.path.splitext(os.path.basename(file_path))[0]
    opening_sequences: Dict[str, List[str]] = {}
    opening_positions: Dict[str, List[str]] = {}
    print(f"Processing file: {os.path.basename(file_path)}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    board = game.board()
                    move_sequence = []
                    for move_number, move in enumerate(game.mainline_moves(), start=1):
                        if move_number > max_depth:
                            break
                        san_move = board.san(move)
                        move_sequence.append(san_move)
                        board.push(move)
                        key_seq = ' '.join(move_sequence)
                        if key_seq not in opening_sequences:
                            opening_sequences[key_seq] = [opening_name]
                        else:
                            if opening_name not in opening_sequences[key_seq]:
                                opening_sequences[key_seq].append(opening_name)
                        fen = board.fen()
                        if fen not in opening_positions:
                            opening_positions[fen] = []
                        move_uci = move.uci()
                        if move_uci not in opening_positions[fen]:
                            opening_positions[fen].append(move_uci)
                except Exception as e:
                    print(f"Error processing game in {os.path.basename(file_path)}: {e}")
                    continue
    except Exception as e:
        print(f"Error processing file {os.path.basename(file_path)}: {e}")
    return opening_sequences, opening_positions

def merge_dictionaries(dicts: List[Tuple[Dict[str, List[str]], Dict[str, List[str]]]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    merged_sequences: Dict[str, List[str]] = {}
    merged_positions: Dict[str, List[str]] = {}
    for sequences_dict, positions_dict in dicts:
        for key, names in sequences_dict.items():
            if key not in merged_sequences:
                merged_sequences[key] = names
            else:
                for name in names:
                    if name not in merged_sequences[key]:
                        merged_sequences[key].append(name)
        for fen, moves in positions_dict.items():
            if fen not in merged_positions:
                merged_positions[fen] = moves
            else:
                for move in moves:
                    if move not in merged_positions[fen]:
                        merged_positions[fen].append(move)
    return merged_sequences, merged_positions

def save_opening_book(opening_sequences: Dict[str, List[str]], opening_positions: Dict[str, List[str]], output_file: str):
    data = {
        'sequences': opening_sequences,
        'positions': opening_positions
    }
    with open(output_file, 'wb') as f:
        compressed_data = zlib.compress(pickle.dumps(data))
        f.write(compressed_data)

def process_pgn_files_in_parallel(pgn_directory: str, output_file: str, max_depth: int = 20):
    if not os.path.exists(pgn_directory):
        print(f"PGN directory not found: {pgn_directory}")
        return

    try:
        files = [os.path.join(pgn_directory, f) for f in os.listdir(pgn_directory) if f.endswith('.pgn')]
        if not files:
            print("No PGN files found in the specified directory.")
            return

        num_processes = min(multiprocessing.cpu_count(), len(files))
        print(f"Using {num_processes} processes for parallel processing.")

        args = [(file_path, max_depth) for file_path in files]

        with Pool(processes=num_processes) as pool:
            results = pool.map(process_pgn_file, args)

        print("Merging results from all processes...")
        opening_sequences, opening_positions = merge_dictionaries(results)

        print("Finished processing PGN files. Saving opening book...")
        save_opening_book(opening_sequences, opening_positions, output_file)
        print(f"Opening book saved to {output_file}")

    except Exception as e:
        print(f"Error processing PGN files: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pgn_directory = os.path.join(current_dir, '..', '..', 'data', 'opening_book', 'raw')
    output_file = os.path.join(current_dir, '..', '..', 'data', 'opening_book', 'processed', 'opening_book.bin')

    process_pgn_files_in_parallel(pgn_directory, output_file, max_depth=20)