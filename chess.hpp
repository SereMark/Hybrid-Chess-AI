#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace chess {

using Bitboard = uint64_t;
using Square = int;
using Hash = uint64_t;

enum Piece { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum Color { WHITE, BLACK };
enum Result { ONGOING, WHITE_WIN, BLACK_WIN, DRAW };

struct Move {
  uint16_t data;

  Move() : data(0) {}
  Move(Square from, Square to, Piece promo = Piece(6))
      : data(from | (to << 6) | (promo << 12)) {}

  Square from() const { return data & 63; }
  Square to() const { return (data >> 6) & 63; }
  Piece promotion() const { return Piece(data >> 12); }
  bool operator==(const Move &o) const { return data == o.data; }
};

class Position {
public:
  Bitboard pieces[6][2] = {};
  Color turn = WHITE;
  uint8_t castling = 15;
  Square ep_square = -1;
  uint16_t halfmove = 0;
  uint16_t fullmove = 1;
  Hash hash = 0;

  std::vector<Hash> history;

  Position();
  void reset();
  void from_fen(const std::string &fen);
  std::string to_fen() const;

  std::vector<Move> legal_moves() const;
  Result make_move(const Move &move);
  Result result() const;

  Bitboard occupied() const;
  Bitboard occupied(Color c) const;
  int piece_at(Square sq) const;

  void update_hash();
  int count_repetitions() const;
};

[[gnu::always_inline, gnu::const]] inline Square lsb(Bitboard bb) {
  return __builtin_ctzll(bb);
}
[[gnu::always_inline, gnu::const]] inline int popcount(Bitboard bb) {
  return __builtin_popcountll(bb);
}
[[gnu::always_inline, gnu::const]] inline Bitboard bit(Square sq) {
  return 1ULL << sq;
}
[[gnu::always_inline]] inline void pop_lsb(Bitboard &bb) { bb &= bb - 1; }

Bitboard pawn_attacks(Square sq, Color c);
Bitboard knight_attacks(Square sq);
Bitboard king_attacks(Square sq);
Bitboard bishop_attacks(Square sq, Bitboard occ);
Bitboard rook_attacks(Square sq, Bitboard occ);
Bitboard queen_attacks(Square sq, Bitboard occ);

extern Hash zobrist_pieces[6][2][64];
extern Hash zobrist_castling[16];
extern Hash zobrist_ep[64];
extern Hash zobrist_turn;

void init_tables();

}