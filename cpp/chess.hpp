#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mcts {
class MCTS;
}

namespace chess {
using Bitboard = uint64_t;
using Square = int;
using Hash = uint64_t;

constexpr int BOARD_SIZE = 8;
constexpr int NSQUARES = 64;
constexpr int MAX_MOVES_PER_POSITION = 256;
constexpr int NUM_PIECE_TYPES = 6;
constexpr int NUM_COLORS = 2;
constexpr int NCASTLING = 16;

enum Piece : uint8_t { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_NONE };

enum Color : uint8_t { WHITE, BLACK, COLOR_NONE };

enum Result : uint8_t { ONGOING, WHITE_WIN, BLACK_WIN, DRAW };

constexpr uint8_t ALL_CASTLING_RIGHTS = 15;
constexpr uint8_t WHITE_KINGSIDE = 1;
constexpr uint8_t WHITE_QUEENSIDE = 2;
constexpr uint8_t BLACK_KINGSIDE = 4;
constexpr uint8_t BLACK_QUEENSIDE = 8;

constexpr Square INVALID_SQUARE = -1;

constexpr Square A1 = 0, B1 = 1, C1 = 2, D1 = 3;
constexpr Square E1 = 4, F1 = 5, G1 = 6, H1 = 7;
constexpr Square A8 = 56, B8 = 57, C8 = 58, D8 = 59;
constexpr Square E8 = 60, F8 = 61, G8 = 62, H8 = 63;

constexpr Square WHITE_KING_START = E1;
constexpr Square WHITE_KING_KINGSIDE = G1;
constexpr Square WHITE_KING_QUEENSIDE = C1;
constexpr Square WHITE_ROOK_KINGSIDE_START = H1;
constexpr Square WHITE_ROOK_KINGSIDE_END = F1;
constexpr Square WHITE_ROOK_QUEENSIDE_START = A1;
constexpr Square WHITE_ROOK_QUEENSIDE_END = D1;

constexpr Square BLACK_KING_START = E8;
constexpr Square BLACK_KING_KINGSIDE = G8;
constexpr Square BLACK_KING_QUEENSIDE = C8;
constexpr Square BLACK_ROOK_KINGSIDE_START = H8;
constexpr Square BLACK_ROOK_KINGSIDE_END = F8;
constexpr Square BLACK_ROOK_QUEENSIDE_START = A8;
constexpr Square BLACK_ROOK_QUEENSIDE_END = D8;

constexpr Bitboard WHITE_KINGSIDE_CLEAR = 0x60ULL;
constexpr Bitboard WHITE_QUEENSIDE_CLEAR = 0xEULL;
constexpr Bitboard BLACK_KINGSIDE_CLEAR = 0x6000000000000000ULL;
constexpr Bitboard BLACK_QUEENSIDE_CLEAR = 0xE00000000000000ULL;

struct Move {
  uint16_t data;

  Move() : data(0) {}

  Move(Square from, Square to, Piece promo = PIECE_NONE)
      : data(from | (to << 6) | (promo << 12)) {}

  Square from() const { return data & 63; }

  Square to() const { return (data >> 6) & 63; }

  Piece promotion() const { return Piece(data >> 12); }

  bool operator==(const Move &other) const { return data == other.data; }
};

struct MoveList {
  Move moves[MAX_MOVES_PER_POSITION];
  size_t count = 0;

  [[gnu::always_inline]]
  void add(Move move) {
    if (count < MAX_MOVES_PER_POSITION) {
      moves[count++] = move;
    }
  }

  [[gnu::always_inline]]
  void add(Square from, Square to, Piece promo = PIECE_NONE) {
    if (count < MAX_MOVES_PER_POSITION) {
      moves[count++] = Move(from, to, promo);
    }
  }

  [[gnu::always_inline]]
  void clear() {
    count = 0;
  }

  [[gnu::always_inline]]
  size_t size() const {
    return count;
  }

  [[gnu::always_inline]]
  bool empty() const {
    return count == 0;
  }

  [[gnu::always_inline]]
  Move &operator[](size_t index) {
    return moves[index];
  }

  [[gnu::always_inline]]
  const Move &operator[](size_t index) const {
    return moves[index];
  }

  [[gnu::always_inline]]
  Move *begin() {
    return moves;
  }

  [[gnu::always_inline]]
  Move *end() {
    return moves + count;
  }

  [[gnu::always_inline]]
  const Move *begin() const {
    return moves;
  }

  [[gnu::always_inline]]
  const Move *end() const {
    return moves + count;
  }
};

class alignas(64) Position {
public:
  Position();
  void reset();
  void from_fen(const std::string &fen);
  std::string to_fen() const;
  void legal_moves(MoveList &moves);
  MoveList legal_moves();
  Result make_move(const Move &move);
  Result result();
  Bitboard occupied() const;
  Bitboard occupied(Color color) const;
  int piece_at(Square square) const;

  [[nodiscard]]
  inline Color get_turn() const {
    return turn;
  }

  [[nodiscard]]
  inline uint8_t get_castling() const {
    return castling;
  }

  [[nodiscard]]
  inline Square get_ep_square() const {
    return ep_square;
  }

  [[nodiscard]]
  inline uint16_t get_halfmove() const {
    return halfmove;
  }

  [[nodiscard]]
  inline uint16_t get_fullmove() const {
    return fullmove;
  }

  [[nodiscard]]
  inline Hash get_hash() const {
    return hash;
  }

  [[nodiscard]]
  inline Bitboard get_pieces(Piece piece, Color color) const {
    return pieces[piece][color];
  }

private:
  alignas(64) Bitboard pieces[6][2] = {};
  Bitboard occupancy_all = 0;
  Bitboard occupancy_color[2] = {0, 0};
  Hash hash = 0;
  Color turn = WHITE;
  uint8_t castling = 15;
  Square ep_square = -1;
  uint16_t halfmove = 0;
  uint16_t fullmove = 1;
  int8_t mailbox[64];
  std::vector<Hash> history;

  struct MoveInfo {
    int captured_piece;
    Square captured_square;
    Square old_ep_square;
    uint8_t old_castling;
    Hash old_hash;
    Bitboard old_occupancy_all;
    Bitboard old_occupancy_color[2];
  };

  void make_move_fast(const Move &move, MoveInfo &info);
  void unmake_move_fast(const Move &move, const MoveInfo &info);
  void update_hash();
  void update_occupancy();
  int count_repetitions() const;
  bool is_insufficient_material() const;
  void add_promotion_moves(Square from, Square to, MoveList &moves) const;
  void add_pawn_moves(Square from, MoveList &moves) const;
  void generate_piece_moves(Piece piece, MoveList &moves) const;
  bool is_square_attacked(Square square, Color by_color) const;
  void execute_move_core(const Move &move, MoveInfo *undo_info = nullptr);
  void update_castling_rights(Square from, Square to);
  void handle_castling_rook_move(Square from, Square to, Color color);
  void clear_position();
  void parse_board_from_fen(const std::string &board_part);
  void parse_game_state_from_fen(const std::string &side,
                                 const std::string &castle,
                                 const std::string &ep, uint16_t halfmove_count,
                                 uint16_t fullmove_count);
  void finalize_position_setup();
  std::string serialize_board_to_fen() const;
  std::string serialize_game_state_to_fen() const;

  friend class mcts::MCTS;
};

[[gnu::always_inline, gnu::const]]
inline Square lsb(Bitboard bitboard) {
  return __builtin_ctzll(bitboard);
}

[[gnu::always_inline, gnu::const]]
inline int popcount(Bitboard bitboard) {
  return __builtin_popcountll(bitboard);
}

[[gnu::always_inline, gnu::const]]
inline Bitboard bit(Square square) {
  return (square >= 0 && square < NSQUARES) ? (1ULL << square) : 0ULL;
}

[[gnu::always_inline, gnu::const]]
inline bool test_bit(Bitboard bitboard, Square square) {
  return (square >= 0 && square < NSQUARES) ? ((bitboard >> square) & 1)
                                            : false;
}

[[gnu::always_inline]]
inline void pop_lsb(Bitboard &bitboard) {
  bitboard &= bitboard - 1;
}

template <typename Func>
[[gnu::always_inline]]
inline void for_each_piece(Bitboard pieces, Func func) {
  while (pieces) {
    const Square square = lsb(pieces);
    pop_lsb(pieces);
    func(square);
  }
}

template <typename Func>
[[gnu::always_inline]]
inline void for_each_attack(Bitboard attacks, Func func) {
  while (attacks) {
    const Square to = lsb(attacks);
    pop_lsb(attacks);
    func(to);
  }
}

Bitboard get_pawn_attacks(Square square, Color color);
Bitboard knight_attacks(Square square);
Bitboard king_attacks(Square square);
Bitboard bishop_attacks(Square square, Bitboard occupancy);
Bitboard rook_attacks(Square square, Bitboard occupancy);
Bitboard queen_attacks(Square square, Bitboard occupancy);

extern Hash zob_pieces[6][2][64];
extern Hash zob_castle[16];
extern Hash zobrist_ep[64];
extern Hash zobrist_turn;
extern Bitboard pawn_attack_table[2][64];
extern Bitboard knight_attack_table[64];
extern Bitboard king_attack_table[64];

void init_tables();
} // namespace chess
