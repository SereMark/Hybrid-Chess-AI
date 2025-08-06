#include "chess.hpp"
#include <algorithm>
#include <cctype>
#include <random>
#include <sstream>

namespace chess {

Hash zobrist_pieces[6][2][64];
Hash zobrist_castling[16];
Hash zobrist_ep[64];
Hash zobrist_turn;

alignas(16) const int ROOK_DIRS[4] = {8, -8, 1, -1};
alignas(16) const int BISHOP_DIRS[4] = {9, -9, 7, -7};

void init_tables() {
  static bool initialized = false;
  if (initialized)
    return;

  std::mt19937_64 rng(0x1337BEEF);

  for (int p = 0; p < 6; p++)
    for (int c = 0; c < 2; c++)
      for (int sq = 0; sq < 64; sq++)
        zobrist_pieces[p][c][sq] = rng();

  for (int i = 0; i < 16; i++)
    zobrist_castling[i] = rng();

  for (int i = 0; i < 64; i++)
    zobrist_ep[i] = rng();

  zobrist_turn = rng();
  initialized = true;
}

Bitboard pawn_attacks(Square sq, Color c) {
  Bitboard att = 0;
  int r = sq / 8, f = sq % 8;

  if (c == WHITE) {
    if (r < 7) {
      if (f > 0)
        att |= bit(sq + 7);
      if (f < 7)
        att |= bit(sq + 9);
    }
  } else {
    if (r > 0) {
      if (f > 0)
        att |= bit(sq - 9);
      if (f < 7)
        att |= bit(sq - 7);
    }
  }
  return att;
}

Bitboard knight_attacks(Square sq) {
  Bitboard att = 0;
  int r = sq / 8, f = sq % 8;

  const int moves[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                           {1, -2},  {1, 2},  {2, -1},  {2, 1}};
  for (auto [dr, df] : moves) {
    int nr = r + dr, nf = f + df;
    if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
      att |= bit(nr * 8 + nf);
  }
  return att;
}

Bitboard king_attacks(Square sq) {
  Bitboard att = 0;
  int r = sq / 8, f = sq % 8;

  for (int dr = -1; dr <= 1; dr++) {
    for (int df = -1; df <= 1; df++) {
      if (dr == 0 && df == 0)
        continue;
      int nr = r + dr, nf = f + df;
      if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
        att |= bit(nr * 8 + nf);
    }
  }
  return att;
}

[[gnu::hot, gnu::flatten]] Bitboard
sliding_attacks(Square sq, Bitboard occ, const int dirs[], int n_dirs) {
  Bitboard att = 0;
  const int r = sq >> 3, f = sq & 7;

  for (int i = 0; i < n_dirs; i++) {
    const int dir = dirs[i];
    int dr, df;
    if (dir > 0) {
      dr = dir >> 3;
      df = dir & 7;
    } else {
      dr = -((-dir) >> 3);
      df = -((-dir) & 7);
    }

    int nr = r + dr, nf = f + df;
    while (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
      const Square s = (nr << 3) + nf;
      const Bitboard sq_bit = 1ULL << s;
      att |= sq_bit;
      if (occ & sq_bit)
        break;
      nr += dr;
      nf += df;
    }
  }
  return att;
}

Bitboard bishop_attacks(Square sq, Bitboard occ) {
  return sliding_attacks(sq, occ, BISHOP_DIRS, 4);
}

Bitboard rook_attacks(Square sq, Bitboard occ) {
  return sliding_attacks(sq, occ, ROOK_DIRS, 4);
}

Bitboard queen_attacks(Square sq, Bitboard occ) {
  return bishop_attacks(sq, occ) | rook_attacks(sq, occ);
}

Position::Position() {
  init_tables();
  reset();
}

void Position::reset() {
  from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Position::from_fen(const std::string &fen) {
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 2; j++)
      pieces[i][j] = 0;

  history.clear();

  std::istringstream ss(fen);
  std::string board, side, castle, ep;
  ss >> board >> side >> castle >> ep >> halfmove >> fullmove;

  Square sq = 56;
  for (char c : board) {
    if (c == '/') {
      sq -= 16;
    } else if (std::isdigit(c)) {
      sq += c - '0';
    } else {
      Color col = std::isupper(c) ? WHITE : BLACK;
      Piece p;
      switch (std::tolower(c)) {
      case 'p':
        p = PAWN;
        break;
      case 'n':
        p = KNIGHT;
        break;
      case 'b':
        p = BISHOP;
        break;
      case 'r':
        p = ROOK;
        break;
      case 'q':
        p = QUEEN;
        break;
      case 'k':
        p = KING;
        break;
      }
      pieces[p][col] |= bit(sq);
      sq++;
    }
  }

  turn = (side == "w") ? WHITE : BLACK;

  castling = 0;
  if (castle.find('K') != std::string::npos)
    castling |= 1;
  if (castle.find('Q') != std::string::npos)
    castling |= 2;
  if (castle.find('k') != std::string::npos)
    castling |= 4;
  if (castle.find('q') != std::string::npos)
    castling |= 8;

  ep_square = (ep == "-") ? -1 : ((ep[0] - 'a') + (ep[1] - '1') * 8);

  update_hash();
}

std::string Position::to_fen() const {
  std::string fen;

  for (int r = 7; r >= 0; r--) {
    int empty = 0;
    for (int f = 0; f < 8; f++) {
      Square sq = r * 8 + f;
      int p = piece_at(sq);

      if (p == -1) {
        empty++;
      } else {
        if (empty) {
          fen += std::to_string(empty);
          empty = 0;
        }
        const char *pieces_str = "pnbrqk";
        char c = pieces_str[p / 2];
        if (p % 2 == 0)
          c = std::toupper(c);
        fen += c;
      }
    }
    if (empty)
      fen += std::to_string(empty);
    if (r > 0)
      fen += '/';
  }

  fen += (turn == WHITE) ? " w " : " b ";

  std::string castle;
  if (castling & 1)
    castle += 'K';
  if (castling & 2)
    castle += 'Q';
  if (castling & 4)
    castle += 'k';
  if (castling & 8)
    castle += 'q';
  if (castle.empty())
    castle = "-";
  fen += castle + " ";

  if (ep_square == -1) {
    fen += "-";
  } else {
    fen += char('a' + (ep_square % 8)) + char('1' + (ep_square / 8));
  }

  fen += " " + std::to_string(halfmove) + " " + std::to_string(fullmove);
  return fen;
}

void Position::update_hash() {
  hash = 0;

  for (int p = 0; p < 6; p++) {
    for (int c = 0; c < 2; c++) {
      Bitboard bb = pieces[p][c];
      while (bb) {
        Square sq = lsb(bb);
        bb &= bb - 1;
        hash ^= zobrist_pieces[p][c][sq];
      }
    }
  }

  hash ^= zobrist_castling[castling];
  if (ep_square >= 0)
    hash ^= zobrist_ep[ep_square];
  if (turn == BLACK)
    hash ^= zobrist_turn;
}

Bitboard Position::occupied() const {
  Bitboard occ = 0;
  for (int p = 0; p < 6; p++)
    for (int c = 0; c < 2; c++)
      occ |= pieces[p][c];
  return occ;
}

Bitboard Position::occupied(Color c) const {
  Bitboard occ = 0;
  for (int p = 0; p < 6; p++)
    occ |= pieces[p][c];
  return occ;
}

int Position::piece_at(Square sq) const {
  if (sq < 0 || sq >= 64)
    return -1;
  Bitboard b = bit(sq);

  for (int p = 0; p < 6; p++) {
    if (pieces[p][WHITE] & b)
      return p * 2;
    if (pieces[p][BLACK] & b)
      return p * 2 + 1;
  }
  return -1;
}

std::vector<Move> Position::legal_moves() const {
  std::vector<Move> moves;
  moves.reserve(256);

  Bitboard occ = occupied();
  Bitboard us = occupied(turn);
  Bitboard them = occupied(Color(1 - turn));


  Bitboard pawns = pieces[PAWN][turn];
  while (pawns) {
    const Square from = lsb(pawns);
    pop_lsb(pawns);

    int dir = (turn == WHITE) ? 8 : -8;
    int start_rank = (turn == WHITE) ? 1 : 6;
    int promo_rank = (turn == WHITE) ? 6 : 1;

    Square to = from + dir;
    if (to >= 0 && to < 64 && !(occ & bit(to))) {
      if (from / 8 == promo_rank) {
        for (Piece p = QUEEN; p >= KNIGHT; p = Piece(p - 1))
          moves.emplace_back(from, to, p);
      } else {
        moves.emplace_back(from, to);
      }

      if (from / 8 == start_rank) {
        to = from + dir * 2;
        if (!(occ & bit(to)))
          moves.emplace_back(from, to);
      }
    }

    Bitboard attacks = pawn_attacks(from, turn) & them;
    while (attacks) {
      to = lsb(attacks);
      attacks &= attacks - 1;

      if (from / 8 == promo_rank) {
        for (Piece p = QUEEN; p >= KNIGHT; p = Piece(p - 1))
          moves.emplace_back(from, to, p);
      } else {
        moves.emplace_back(from, to);
      }
    }

    if (ep_square >= 0) {
      if (pawn_attacks(from, turn) & bit(ep_square))
        moves.emplace_back(from, ep_square);
    }
  }

  Bitboard knights = pieces[KNIGHT][turn];
  while (knights) {
    const Square from = lsb(knights);
    pop_lsb(knights);

    Bitboard attacks = knight_attacks(from) & ~us;
    while (attacks) {
      const Square to = lsb(attacks);
      pop_lsb(attacks);
      moves.emplace_back(from, to);
    }
  }

  Bitboard bishops = pieces[BISHOP][turn];
  while (bishops) {
    Square from = lsb(bishops);
    bishops &= bishops - 1;

    Bitboard attacks = bishop_attacks(from, occ) & ~us;
    while (attacks) {
      const Square to = lsb(attacks);
      pop_lsb(attacks);
      moves.emplace_back(from, to);
    }
  }

  Bitboard rooks = pieces[ROOK][turn];
  while (rooks) {
    Square from = lsb(rooks);
    rooks &= rooks - 1;

    Bitboard attacks = rook_attacks(from, occ) & ~us;
    while (attacks) {
      const Square to = lsb(attacks);
      pop_lsb(attacks);
      moves.emplace_back(from, to);
    }
  }

  Bitboard queens = pieces[QUEEN][turn];
  while (queens) {
    Square from = lsb(queens);
    queens &= queens - 1;

    Bitboard attacks = queen_attacks(from, occ) & ~us;
    while (attacks) {
      const Square to = lsb(attacks);
      pop_lsb(attacks);
      moves.emplace_back(from, to);
    }
  }

  Bitboard king = pieces[KING][turn];
  if (king) {
    Square from = lsb(king);

    Bitboard attacks = king_attacks(from) & ~us;
    while (attacks) {
      const Square to = lsb(attacks);
      pop_lsb(attacks);
      moves.emplace_back(from, to);
    }

      if (turn == WHITE) {
      if ((castling & 1) && !(occ & 0x60) && from == 4)
        moves.emplace_back(4, 6);
      if ((castling & 2) && !(occ & 0xE) && from == 4)
        moves.emplace_back(4, 2);
    } else {
      if ((castling & 4) && !(occ & 0x6000000000000000ULL) && from == 60)
        moves.emplace_back(60, 62);
      if ((castling & 8) && !(occ & 0xE00000000000000ULL) && from == 60)
        moves.emplace_back(60, 58);
    }
  }

  std::vector<Move> legal;
  for (const Move &m : moves) {
    Position temp = *this;

    Square from = m.from();
    Square to = m.to();

    int piece = piece_at(from);
    if (piece == -1)
      continue;

    Piece p = Piece(piece / 2);
    Color c = Color(piece % 2);

    temp.pieces[p][c] &= ~bit(from);

    int captured = piece_at(to);
    if (captured >= 0) {
      temp.pieces[captured / 2][captured % 2] &= ~bit(to);
    }

    if (m.promotion() <= KING) {
      temp.pieces[m.promotion()][c] |= bit(to);
    } else {
      temp.pieces[p][c] |= bit(to);
    }

    if (p == KING && abs(to - from) == 2) {
      if (to > from) {
        temp.pieces[ROOK][c] &= ~bit(from + 3);
        temp.pieces[ROOK][c] |= bit(from + 1);
      } else {
        temp.pieces[ROOK][c] &= ~bit(from - 4);
        temp.pieces[ROOK][c] |= bit(from - 1);
      }
    }

    if (p == PAWN && to == ep_square && ep_square >= 0) {
      Square cap_sq = ep_square + (turn == WHITE ? -8 : 8);
      temp.pieces[PAWN][1 - turn] &= ~bit(cap_sq);
    }

    Bitboard king_bb = temp.pieces[KING][turn];
    if (!king_bb)
      continue;

    Square king_sq = lsb(king_bb);

    Bitboard temp_occ = temp.occupied();
    Color enemy = Color(1 - turn);

    bool in_check = false;

    if (pawn_attacks(king_sq, turn) & temp.pieces[PAWN][enemy])
      in_check = true;

    if (!in_check && knight_attacks(king_sq) & temp.pieces[KNIGHT][enemy])
      in_check = true;

    if (!in_check &&
        bishop_attacks(king_sq, temp_occ) &
            (temp.pieces[BISHOP][enemy] | temp.pieces[QUEEN][enemy]))
      in_check = true;

    if (!in_check && rook_attacks(king_sq, temp_occ) &
                         (temp.pieces[ROOK][enemy] | temp.pieces[QUEEN][enemy]))
      in_check = true;

    if (!in_check && king_attacks(king_sq) & temp.pieces[KING][enemy])
      in_check = true;

    if (!in_check) {
      legal.push_back(m);
    }
  }

  return legal;
}

Result Position::make_move(const Move &move) {
  history.push_back(hash);

  Square from = move.from();
  Square to = move.to();

  int piece = piece_at(from);
  if (piece == -1)
    return ONGOING;

  Piece p = Piece(piece / 2);
  Color c = Color(piece % 2);

  hash ^= zobrist_pieces[p][c][from];
  if (move.promotion() <= KING) {
    hash ^= zobrist_pieces[move.promotion()][c][to];
  } else {
    hash ^= zobrist_pieces[p][c][to];
  }

  int captured = piece_at(to);
  if (captured >= 0) {
    hash ^= zobrist_pieces[captured / 2][captured % 2][to];
    pieces[captured / 2][captured % 2] &= ~bit(to);
  }

  pieces[p][c] &= ~bit(from);
  if (move.promotion() <= KING) {
    pieces[move.promotion()][c] |= bit(to);
  } else {
    pieces[p][c] |= bit(to);
  }

  if (p == KING) {
    if (abs(to - from) == 2) {
      if (to > from) {
        pieces[ROOK][c] &= ~bit(from + 3);
        pieces[ROOK][c] |= bit(from + 1);
        hash ^= zobrist_pieces[ROOK][c][from + 3];
        hash ^= zobrist_pieces[ROOK][c][from + 1];
      } else {
        pieces[ROOK][c] &= ~bit(from - 4);
        pieces[ROOK][c] |= bit(from - 1);
        hash ^= zobrist_pieces[ROOK][c][from - 4];
        hash ^= zobrist_pieces[ROOK][c][from - 1];
      }
    }
    if (c == WHITE)
      castling &= ~3;
    else
      castling &= ~12;
  }

  if (from == 0 || to == 0)
    castling &= ~2;
  if (from == 7 || to == 7)
    castling &= ~1;
  if (from == 56 || to == 56)
    castling &= ~8;
  if (from == 63 || to == 63)
    castling &= ~4;

  hash ^= zobrist_castling[castling];
  if (ep_square >= 0)
    hash ^= zobrist_ep[ep_square];

  if (p == PAWN && to == ep_square && ep_square >= 0) {
    Square cap_sq = ep_square + (turn == WHITE ? -8 : 8);
    pieces[PAWN][1 - turn] &= ~bit(cap_sq);
    hash ^= zobrist_pieces[PAWN][1 - turn][cap_sq];
  }

  ep_square = -1;
  if (p == PAWN && abs(to - from) == 16) {
    ep_square = (from + to) / 2;
  }

  if (ep_square >= 0)
    hash ^= zobrist_ep[ep_square];
  hash ^= zobrist_castling[castling];

  if (captured >= 0 || p == PAWN) {
    halfmove = 0;
  } else {
    halfmove++;
  }

  if (turn == BLACK)
    fullmove++;

  turn = Color(1 - turn);
  hash ^= zobrist_turn;

  return result();
}

Result Position::result() const {
  auto moves = legal_moves();
  if (!moves.empty())
    return ONGOING;

  Bitboard king_bb = pieces[KING][turn];
  if (!king_bb)
    return turn == WHITE ? BLACK_WIN : WHITE_WIN;

  Square king_sq = lsb(king_bb);
  Bitboard occ = occupied();
  Color enemy = Color(1 - turn);

  bool in_check = false;

  if (pawn_attacks(king_sq, turn) & pieces[PAWN][enemy])
    in_check = true;

  if (!in_check && knight_attacks(king_sq) & pieces[KNIGHT][enemy])
    in_check = true;

  if (!in_check && bishop_attacks(king_sq, occ) &
                       (pieces[BISHOP][enemy] | pieces[QUEEN][enemy]))
    in_check = true;

  if (!in_check &&
      rook_attacks(king_sq, occ) & (pieces[ROOK][enemy] | pieces[QUEEN][enemy]))
    in_check = true;

  if (!in_check && king_attacks(king_sq) & pieces[KING][enemy])
    in_check = true;

  if (in_check) {
    return turn == WHITE ? BLACK_WIN : WHITE_WIN;
  }

  if (halfmove >= 100)
    return DRAW;
  if (count_repetitions() >= 2)
    return DRAW;

  return DRAW;
}

int Position::count_repetitions() const {
  int count = 0;
  for (Hash h : history) {
    if (h == hash)
      count++;
  }
  return count;
}

}