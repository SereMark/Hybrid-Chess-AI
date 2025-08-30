#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mcts {
class MCTS;
}

namespace chess {

using Bitboard = uint64_t; // 64-bit bitboard
using Square = int;        // 0..63 (A1..H8)
using Hash = uint64_t;     // Zobrist hash

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

constexpr Square A1 = 0, B1 = 1, C1 = 2, D1 = 3, E1 = 4, F1 = 5, G1 = 6, H1 = 7,
                 A8 = 56, B8 = 57, C8 = 58, D8 = 59, E8 = 60, F8 = 61, G8 = 62,
                 H8 = 63;

constexpr Square WHITE_KING_START = E1, WHITE_KING_KINGSIDE = G1,
                 WHITE_KING_QUEENSIDE = C1, WHITE_ROOK_KINGSIDE_START = H1,
                 WHITE_ROOK_KINGSIDE_END = F1, WHITE_ROOK_QUEENSIDE_START = A1,
                 WHITE_ROOK_QUEENSIDE_END = D1;

constexpr Square BLACK_KING_START = E8, BLACK_KING_KINGSIDE = G8,
                 BLACK_KING_QUEENSIDE = C8, BLACK_ROOK_KINGSIDE_START = H8,
                 BLACK_ROOK_KINGSIDE_END = F8, BLACK_ROOK_QUEENSIDE_START = A8,
                 BLACK_ROOK_QUEENSIDE_END = D8;

constexpr Bitboard WHITE_KINGSIDE_CLEAR = 0x60ULL,
                   WHITE_QUEENSIDE_CLEAR = 0xEULL,
                   BLACK_KINGSIDE_CLEAR = 0x6000000000000000ULL,
                   BLACK_QUEENSIDE_CLEAR = 0x0E00000000000000ULL;

constexpr int PAWN_DOUBLE_MOVE_DISTANCE = 16, KING_CASTLE_DISTANCE = 2;
// Packed move (from, to, promotion)
struct Move {
  uint16_t data = 0;
  Move() noexcept = default;
  Move(Square from, Square to, Piece promo = PIECE_NONE) noexcept
      : data(static_cast<uint16_t>(from | (to << 6) | (promo << 12))) {}
  [[nodiscard]] Square from() const noexcept { return data & 63; }
  [[nodiscard]] Square to() const noexcept { return (data >> 6) & 63; }
  [[nodiscard]] Piece promotion() const noexcept { return Piece(data >> 12); }
  bool operator==(const Move &o) const noexcept { return data == o.data; }
};

// Fixed-capacity collection of moves
struct MoveList {
  Move moves[MAX_MOVES_PER_POSITION];
  size_t count = 0;
  [[gnu::always_inline]] void add(Move m) noexcept {
    if (count < MAX_MOVES_PER_POSITION)
      moves[count++] = m;
  }
  [[gnu::always_inline]] void add(Square from, Square to,
                                  Piece promo = PIECE_NONE) noexcept {
    if (count < MAX_MOVES_PER_POSITION)
      moves[count++] = Move(from, to, promo);
  }
  [[gnu::always_inline]] void clear() noexcept { count = 0; }
  [[nodiscard]] [[gnu::always_inline]] size_t size() const noexcept {
    return count;
  }
  [[nodiscard]] [[gnu::always_inline]] bool empty() const noexcept {
    return count == 0;
  }
  [[gnu::always_inline]] Move &operator[](size_t i) noexcept {
    return moves[i];
  }
  [[gnu::always_inline]] const Move &operator[](size_t i) const noexcept {
    return moves[i];
  }
  [[gnu::always_inline]] Move *begin() noexcept { return moves; }
  [[gnu::always_inline]] Move *end() noexcept { return moves + count; }
  [[gnu::always_inline]] const Move *begin() const noexcept { return moves; }
  [[gnu::always_inline]] const Move *end() const noexcept {
    return moves + count;
  }
};

// Board state, move generation, hashing
class alignas(64) Position {
public:
  Position();
  void reset();
  void from_fen(const std::string &);
  [[nodiscard]] std::string to_fen() const;
  void legal_moves(MoveList &);
  [[nodiscard]] MoveList legal_moves();
  Result make_move(const Move &);
  Result result();
  [[nodiscard]] inline int repetition_count() const {
    return count_repetitions();
  }
  [[nodiscard]] Bitboard occupied() const noexcept;
  [[nodiscard]] Bitboard occupied(Color) const noexcept;
  [[nodiscard]] int piece_at(Square) const noexcept;
  [[nodiscard]] inline Color get_turn() const noexcept { return turn; }
  [[nodiscard]] inline uint8_t get_castling() const noexcept {
    return castling;
  }
  [[nodiscard]] inline Square get_ep_square() const noexcept {
    return ep_square;
  }
  [[nodiscard]] inline uint16_t get_halfmove() const noexcept {
    return halfmove;
  }
  [[nodiscard]] inline uint16_t get_fullmove() const noexcept {
    return fullmove;
  }
  [[nodiscard]] inline Hash get_hash() const noexcept { return hash; }
  [[nodiscard]] inline Bitboard get_pieces(Piece p, Color c) const noexcept {
    return pieces[p][c];
  }

private:
  alignas(64) Bitboard pieces[6][2]{}; // piece type x color
  Bitboard occupancy_all = 0, occupancy_color[2]{0, 0};
  Hash hash = 0;
  Color turn = WHITE;
  uint8_t castling = 15;
  Square ep_square = -1;
  uint16_t halfmove = 0, fullmove = 1;
  int8_t mailbox[64];
  std::vector<Hash> history;
  struct MoveInfo {
    int captured_piece = -1;
    Square captured_square = INVALID_SQUARE;
    Square old_ep_square = INVALID_SQUARE;
    uint8_t old_castling = 0;
    Hash old_hash = 0;
    Bitboard old_occupancy_all = 0;
    Bitboard old_occupancy_color[2]{0, 0};
    uint16_t old_halfmove = 0;
    uint16_t old_fullmove = 1;
    int moved_piece = -1;
  };
  void make_move_fast(const Move &, MoveInfo &);
  void unmake_move_fast(const Move &, const MoveInfo &);
  void update_hash();
  void update_occupancy();
  [[nodiscard]] int count_repetitions() const;
  [[nodiscard]] bool is_insufficient_material() const;
  void add_promotion_moves(Square, Square, MoveList &) const;
  void add_pawn_moves(Square, MoveList &) const;
  void generate_piece_moves(Piece, MoveList &) const;
  [[nodiscard]] bool is_square_attacked(Square, Color) const;
  void execute_move_core(const Move &, MoveInfo *undo_info = nullptr);
  void update_castling_rights(Square, Square);
  void handle_castling_rook_move(Square, Square, Color);
  void clear_position();
  void parse_board_from_fen(const std::string &);
  void parse_game_state_from_fen(const std::string &, const std::string &,
                                 const std::string &, uint16_t, uint16_t);
  void finalize_position_setup();
  [[nodiscard]] std::string serialize_board_to_fen() const;
  [[nodiscard]] std::string serialize_game_state_to_fen() const;
  friend class mcts::MCTS;
};
[[gnu::always_inline, gnu::const]] inline Square lsb(Bitboard b) noexcept {
  return __builtin_ctzll(b);
}
[[gnu::always_inline, gnu::const]] inline int popcount(Bitboard b) noexcept {
  return __builtin_popcountll(b);
}
[[gnu::always_inline, gnu::const]] inline Bitboard bit(Square s) noexcept {
  return (s >= 0 && s < NSQUARES) ? (1ULL << s) : 0ULL;
}
[[gnu::always_inline]] inline void pop_lsb(Bitboard &b) noexcept { b &= b - 1; }
template <typename F>
[[gnu::always_inline]] inline void for_each_piece(Bitboard bb, F f) {
  while (bb) {
    const Square s = lsb(bb);
    pop_lsb(bb);
    f(s);
  }
}
template <typename F>
[[gnu::always_inline]] inline void for_each_attack(Bitboard a, F f) {
  while (a) {
    const Square t = lsb(a);
    pop_lsb(a);
    f(t);
  }
}
Bitboard get_pawn_attacks(Square, Color);
Bitboard knight_attacks(Square);
Bitboard king_attacks(Square);
Bitboard bishop_attacks(Square, Bitboard);
Bitboard rook_attacks(Square, Bitboard);
Bitboard queen_attacks(Square, Bitboard);
extern Hash zob_pieces[6][2][64];
extern Hash zob_castle[16];
extern Hash zobrist_ep[64];
extern Hash zobrist_turn;
extern Bitboard pawn_attack_table[2][64];
extern Bitboard knight_attack_table[64];
extern Bitboard king_attack_table[64];
void init_tables();
} // namespace chess
