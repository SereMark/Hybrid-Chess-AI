#include "chess.hpp"
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <random>
#include <sstream>
namespace chess {
Hash zob_pieces[6][2][64];
Hash zob_castle[16];
Hash zobrist_ep[64];
Hash zobrist_turn;
Bitboard pawn_attack_table[2][64];
Bitboard knight_attack_table[64];
Bitboard king_attack_table[64];
alignas(64) static unsigned char ray_squares[8][64][7];
alignas(64) static unsigned char ray_len[8][64];
static constexpr int DIR[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                  {0, 1},   {1, -1}, {1, 0},  {1, 1}};
static constexpr int KJ[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                                 {1, -2},  {1, 2},  {2, -1},  {2, 1}};
static constexpr int ROOK_DIRS[4] = {1, 3, 4, 6};
static constexpr int BISHOP_DIRS[4] = {0, 2, 5, 7};
void init_tables() {
  static std::once_flag once;
  std::call_once(once, [] {
    std::mt19937_64 rng(0x1337BEEF);
    for (int p = 0; p < NUM_PIECE_TYPES; ++p)
      for (int c = 0; c < NUM_COLORS; ++c)
        for (int s = 0; s < NSQUARES; ++s)
          zob_pieces[p][c][s] = rng();
    for (int i = 0; i < NCASTLING; ++i)
      zob_castle[i] = rng();
    for (int i = 0; i < NSQUARES; ++i)
      zobrist_ep[i] = rng();
    zobrist_turn = rng();
    for (int s = 0; s < NSQUARES; ++s) {
      int r = s / BOARD_SIZE, f = s % BOARD_SIZE;
      pawn_attack_table[WHITE][s] = 0;
      pawn_attack_table[BLACK][s] = 0;
      if (r < BOARD_SIZE - 1) {
        if (f > 0)
          pawn_attack_table[WHITE][s] |= bit(s + 7);
        if (f < BOARD_SIZE - 1)
          pawn_attack_table[WHITE][s] |= bit(s + 9);
      }
      if (r > 0) {
        if (f > 0)
          pawn_attack_table[BLACK][s] |= bit(s - 9);
        if (f < BOARD_SIZE - 1)
          pawn_attack_table[BLACK][s] |= bit(s - 7);
      }
      knight_attack_table[s] = 0;
      for (auto &d : KJ) {
        int nr = r + d[0], nf = f + d[1];
        if (nr >= 0 && nr < BOARD_SIZE && nf >= 0 && nf < BOARD_SIZE)
          knight_attack_table[s] |= bit(nr * BOARD_SIZE + nf);
      }
      king_attack_table[s] = 0;
      for (int dr = -1; dr <= 1; ++dr)
        for (int df = -1; df <= 1; ++df) {
          if (!dr && !df)
            continue;
          int nr = r + dr, nf = f + df;
          if (nr >= 0 && nr < BOARD_SIZE && nf >= 0 && nf < BOARD_SIZE)
            king_attack_table[s] |= bit(nr * BOARD_SIZE + nf);
        }
      for (int d = 0; d < 8; ++d) {
        int nr = r + DIR[d][0], nf = f + DIR[d][1];
        int k = 0;
        while (nr >= 0 && nr < BOARD_SIZE && nf >= 0 && nf < BOARD_SIZE) {
          ray_squares[d][s][k++] =
              static_cast<unsigned char>(nr * BOARD_SIZE + nf);
          nr += DIR[d][0];
          nf += DIR[d][1];
        }
        ray_len[d][s] = static_cast<unsigned char>(k);
      }
    }
  });
}
Bitboard get_pawn_attacks(Square s, Color c) { return pawn_attack_table[c][s]; }
Bitboard knight_attacks(Square s) { return knight_attack_table[s]; }
Bitboard king_attacks(Square s) { return king_attack_table[s]; }
[[gnu::hot]] static inline Bitboard
sliding_attacks(Square s, Bitboard occ, const int dir_ids[], int nd) {
  Bitboard a = 0;
  for (int i = 0; i < nd; ++i) {
    const int d = dir_ids[i];
    const int len = ray_len[d][s];
    const unsigned char *rs = ray_squares[d][s];
    for (int k = 0; k < len; ++k) {
      const Square t = static_cast<Square>(rs[k]);
      const Bitboard b = 1ULL << t;
      a |= b;
      if (occ & b)
        break;
    }
  }
  return a;
}
Bitboard bishop_attacks(Square s, Bitboard o) {
  return sliding_attacks(s, o, BISHOP_DIRS, 4);
}
Bitboard rook_attacks(Square s, Bitboard o) {
  return sliding_attacks(s, o, ROOK_DIRS, 4);
}
Bitboard queen_attacks(Square s, Bitboard o) {
  return bishop_attacks(s, o) | rook_attacks(s, o);
}
Position::Position() {
  init_tables();
  for (int i = 0; i < NSQUARES; ++i)
    mailbox[i] = -1;
  reset();
}
void Position::reset() {
  from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}
void Position::clear_position() {
  for (int i = 0; i < NUM_PIECE_TYPES; ++i)
    for (int j = 0; j < NUM_COLORS; ++j)
      pieces[i][j] = 0;
  for (int i = 0; i < NSQUARES; ++i)
    mailbox[i] = -1;
  history.clear();
}
void Position::parse_board_from_fen(const std::string &board) {
  Square s = 56;
  for (char ch : board) {
    if (ch == '/')
      s -= 16;
    else if (std::isdigit(static_cast<unsigned char>(ch))) {
      int k = ch - '0';
      if (k >= 1 && k <= 8)
        s += k;
    } else {
      const Color c =
          std::isupper(static_cast<unsigned char>(ch)) ? WHITE : BLACK;
      Piece pc = PIECE_NONE;
      switch (std::tolower(static_cast<unsigned char>(ch))) {
      case 'p':
        pc = PAWN;
        break;
      case 'n':
        pc = KNIGHT;
        break;
      case 'b':
        pc = BISHOP;
        break;
      case 'r':
        pc = ROOK;
        break;
      case 'q':
        pc = QUEEN;
        break;
      case 'k':
        pc = KING;
        break;
      default:
        continue;
      }
      if (s >= 0 && s < NSQUARES) {
        pieces[pc][c] |= bit(s);
        mailbox[s] = pc * 2 + c;
      }
      ++s;
    }
  }
}
void Position::parse_game_state_from_fen(const std::string &side,
                                         const std::string &castle,
                                         const std::string &ep, uint16_t h,
                                         uint16_t f) {
  bool side_bad = false;
  if (side == "w" || side == "W")
    turn = WHITE;
  else if (side == "b" || side == "B")
    turn = BLACK;
  else {
    turn = WHITE;
    side_bad = true;
  }
  castling = 0;
  if (!side_bad && castle.find('K') != std::string::npos)
    castling |= WHITE_KINGSIDE;
  if (!side_bad && castle.find('Q') != std::string::npos)
    castling |= WHITE_QUEENSIDE;
  if (!side_bad && castle.find('k') != std::string::npos)
    castling |= BLACK_KINGSIDE;
  if (!side_bad && castle.find('q') != std::string::npos)
    castling |= BLACK_QUEENSIDE;
  if (ep == "-")
    ep_square = INVALID_SQUARE;
  else if (ep.size() == 2 && ep[0] >= 'a' && ep[0] <= 'h' && ep[1] >= '1' &&
           ep[1] <= '8')
    ep_square = (ep[0] - 'a') + (ep[1] - '1') * BOARD_SIZE;
  else
    ep_square = INVALID_SQUARE;
  if (side_bad) {
    castling = 0;
    ep_square = INVALID_SQUARE;
  }
  halfmove = h;
  fullmove = f;
}
void Position::finalize_position_setup() {
  update_occupancy();
  if (ep_square != INVALID_SQUARE) {
    const Color stm = turn;
    const Color opp = Color(1 - stm);
    const Square cs = ep_square + ((stm == WHITE) ? -BOARD_SIZE : BOARD_SIZE);
    const bool oppPawnBehind =
        (cs >= 0 && cs < NSQUARES) && (pieces[PAWN][opp] & bit(cs));
    const bool stmCanCapture =
        (get_pawn_attacks(ep_square, opp) & pieces[PAWN][stm]) != 0;
    if (!oppPawnBehind || !stmCanCapture)
      ep_square = INVALID_SQUARE;
  }
  update_hash();
  history.clear();
  history.push_back(hash);
}
void Position::from_fen(const std::string &fen) {
  if (fen.empty()) {
    reset();
    return;
  }
  std::istringstream ss(fen);
  std::string board, side, castle, ep;
  uint16_t h = 0, f = 1;
  ss >> board >> side >> castle >> ep >> h >> f;
  if (board.empty() || side.empty()) {
    reset();
    return;
  }
  clear_position();
  parse_board_from_fen(board);
  parse_game_state_from_fen(side, castle, ep, h, f);
  finalize_position_setup();
}
std::string Position::serialize_board_to_fen() const {
  std::string out;
  for (int r = BOARD_SIZE - 1; r >= 0; --r) {
    int empty = 0;
    for (int c = 0; c < BOARD_SIZE; ++c) {
      Square s = r * BOARD_SIZE + c;
      int p = piece_at(s);
      if (p == -1) {
        ++empty;
      } else {
        if (empty) {
          out += std::to_string(empty);
          empty = 0;
        }
        const char ps[] = "pnbrqk";
        char ch = ps[p / 2];
        if (p % 2 == 0)
          ch = std::toupper(static_cast<unsigned char>(ch));
        out += ch;
      }
    }
    if (empty)
      out += std::to_string(empty);
    if (r > 0)
      out += '/';
  }
  return out;
}
std::string Position::serialize_game_state_to_fen() const {
  std::string s;
  s += (turn == WHITE) ? " w " : " b ";
  std::string cr;
  if (castling & WHITE_KINGSIDE)
    cr += 'K';
  if (castling & WHITE_QUEENSIDE)
    cr += 'Q';
  if (castling & BLACK_KINGSIDE)
    cr += 'k';
  if (castling & BLACK_QUEENSIDE)
    cr += 'q';
  if (cr.empty())
    cr = "-";
  s += cr + " ";
  if (ep_square == INVALID_SQUARE)
    s += "-";
  else {
    s += char('a' + (ep_square % BOARD_SIZE));
    s += char('1' + (ep_square / BOARD_SIZE));
  }
  s += " " + std::to_string(halfmove) + " " + std::to_string(fullmove);
  return s;
}
std::string Position::to_fen() const {
  return serialize_board_to_fen() + serialize_game_state_to_fen();
}
void Position::update_hash() {
  hash = 0;
  for (int p = 0; p < NUM_PIECE_TYPES; ++p)
    for (int c = 0; c < NUM_COLORS; ++c) {
      Bitboard bb = pieces[p][c];
      while (bb) {
        Square s = lsb(bb);
        bb &= bb - 1;
        hash ^= zob_pieces[p][c][s];
      }
    }
  hash ^= zob_castle[castling];
  if (ep_square != INVALID_SQUARE)
    hash ^= zobrist_ep[ep_square];
  if (turn == BLACK)
    hash ^= zobrist_turn;
}
void Position::update_occupancy() {
  occupancy_color[WHITE] = 0;
  occupancy_color[BLACK] = 0;
  for (int p = 0; p < NUM_PIECE_TYPES; ++p) {
    occupancy_color[WHITE] |= pieces[p][WHITE];
    occupancy_color[BLACK] |= pieces[p][BLACK];
  }
  occupancy_all = occupancy_color[WHITE] | occupancy_color[BLACK];
}
Bitboard Position::occupied() const noexcept { return occupancy_all; }
Bitboard Position::occupied(Color c) const noexcept {
  return occupancy_color[c];
}
int Position::piece_at(Square s) const noexcept {
  return (s < 0 || s >= NSQUARES) ? -1 : mailbox[s];
}
void Position::add_promotion_moves(Square f, Square t, MoveList &m) const {
  static constexpr Piece promos[] = {QUEEN, ROOK, BISHOP, KNIGHT};
  for (Piece pc : promos)
    m.add(f, t, pc);
}
void Position::add_pawn_moves(Square from, MoveList &m) const {
  const int dir = (turn == WHITE) ? BOARD_SIZE : -BOARD_SIZE,
            start = (turn == WHITE) ? 1 : 6, promo = (turn == WHITE) ? 6 : 1;
  const Bitboard occ = occupied(), opp = occupied(Color(1 - turn));
  const Square one = from + dir;
  if (one >= 0 && one < NSQUARES && !(occ & bit(one))) {
    if (from / BOARD_SIZE == promo)
      add_promotion_moves(from, one, m);
    else
      m.add(from, one);
    if (from / BOARD_SIZE == start) {
      const Square two = from + dir * 2;
      if (!(occ & bit(two)))
        m.add(from, two);
    }
  }
  const Bitboard att = get_pawn_attacks(from, turn) & opp;
  for_each_attack(att, [&](Square to) {
    if (from / BOARD_SIZE == promo)
      add_promotion_moves(from, to, m);
    else
      m.add(from, to);
  });
  if (ep_square != INVALID_SQUARE &&
      (get_pawn_attacks(from, turn) & bit(ep_square)))
    m.add(from, ep_square);
}
bool Position::is_square_attacked(Square s, Color by) const {
  const Bitboard occ = occupied();
  return (get_pawn_attacks(s, Color(1 - by)) & pieces[PAWN][by]) ||
         (knight_attacks(s) & pieces[KNIGHT][by]) ||
         (bishop_attacks(s, occ) & (pieces[BISHOP][by] | pieces[QUEEN][by])) ||
         (rook_attacks(s, occ) & (pieces[ROOK][by] | pieces[QUEEN][by])) ||
         (king_attacks(s) & pieces[KING][by]);
}
void Position::generate_piece_moves(Piece pc, MoveList &m) const {
  const Bitboard us = occupied(turn), occ = occupied();
  for_each_piece(pieces[pc][turn], [&](Square from) {
    Bitboard a = 0;
    switch (pc) {
    case KNIGHT:
      a = knight_attacks(from);
      break;
    case BISHOP:
      a = bishop_attacks(from, occ);
      break;
    case ROOK:
      a = rook_attacks(from, occ);
      break;
    case QUEEN:
      a = queen_attacks(from, occ);
      break;
    case KING:
      a = king_attacks(from);
      break;
    default:
      return;
    }
    a &= ~us;
    for_each_attack(a, [&](Square to) { m.add(from, to); });
  });
}
void Position::legal_moves(MoveList &m) {
  m.clear();
  if (!pieces[KING][turn]) {
    return;
  }
  for_each_piece(pieces[PAWN][turn],
                 [&](Square from) { add_pawn_moves(from, m); });
  generate_piece_moves(KNIGHT, m);
  generate_piece_moves(BISHOP, m);
  generate_piece_moves(ROOK, m);
  generate_piece_moves(QUEEN, m);
  generate_piece_moves(KING, m);
  if (pieces[KING][turn]) {
    const Square k = lsb(pieces[KING][turn]);
    const Color e = Color(1 - turn);
    if (turn == WHITE) {
      if ((castling & WHITE_KINGSIDE) &&
          (pieces[ROOK][WHITE] & bit(WHITE_ROOK_KINGSIDE_START)) &&
          !(occupied() & WHITE_KINGSIDE_CLEAR) && k == WHITE_KING_START &&
          !is_square_attacked(WHITE_KING_START, e) &&
          !is_square_attacked(F1, e) &&
          !is_square_attacked(WHITE_KING_KINGSIDE, e))
        m.add(WHITE_KING_START, WHITE_KING_KINGSIDE);
      if ((castling & WHITE_QUEENSIDE) &&
          (pieces[ROOK][WHITE] & bit(WHITE_ROOK_QUEENSIDE_START)) &&
          !(occupied() & WHITE_QUEENSIDE_CLEAR) && k == WHITE_KING_START &&
          !is_square_attacked(WHITE_KING_START, e) &&
          !is_square_attacked(D1, e) &&
          !is_square_attacked(WHITE_KING_QUEENSIDE, e))
        m.add(WHITE_KING_START, WHITE_KING_QUEENSIDE);
    } else {
      if ((castling & BLACK_KINGSIDE) &&
          (pieces[ROOK][BLACK] & bit(BLACK_ROOK_KINGSIDE_START)) &&
          !(occupied() & BLACK_KINGSIDE_CLEAR) && k == BLACK_KING_START &&
          !is_square_attacked(BLACK_KING_START, e) &&
          !is_square_attacked(F8, e) &&
          !is_square_attacked(BLACK_KING_KINGSIDE, e))
        m.add(BLACK_KING_START, BLACK_KING_KINGSIDE);
      if ((castling & BLACK_QUEENSIDE) &&
          (pieces[ROOK][BLACK] & bit(BLACK_ROOK_QUEENSIDE_START)) &&
          !(occupied() & BLACK_QUEENSIDE_CLEAR) && k == BLACK_KING_START &&
          !is_square_attacked(BLACK_KING_START, e) &&
          !is_square_attacked(D8, e) &&
          !is_square_attacked(BLACK_KING_QUEENSIDE, e))
        m.add(BLACK_KING_START, BLACK_KING_QUEENSIDE);
    }
  }
  MoveList legal;
  for (size_t i = 0; i < m.size(); ++i) {
    const Move &mv = m[i];
    if (piece_at(mv.from()) == -1)
      continue;
    const int dst = piece_at(mv.to());
    if (dst == KING * 2 + (1 - turn))
      continue;
    MoveInfo info;
    make_move_fast(mv, info);
    bool ok = true;
    if (const Bitboard kb = pieces[KING][Color(1 - turn)]) {
      const Square ks = lsb(kb);
      ok = !is_square_attacked(ks, turn);
    }
    unmake_move_fast(mv, info);
    if (ok)
      legal.add(mv);
  }
  m = legal;
}
MoveList Position::legal_moves() {
  MoveList m;
  legal_moves(m);
  return m;
}
void Position::update_castling_rights(Square f, Square t) {
  if (f == WHITE_ROOK_QUEENSIDE_START || t == WHITE_ROOK_QUEENSIDE_START)
    castling &= ~WHITE_QUEENSIDE;
  if (f == WHITE_ROOK_KINGSIDE_START || t == WHITE_ROOK_KINGSIDE_START)
    castling &= ~WHITE_KINGSIDE;
  if (f == BLACK_ROOK_QUEENSIDE_START || t == BLACK_ROOK_QUEENSIDE_START)
    castling &= ~BLACK_QUEENSIDE;
  if (f == BLACK_ROOK_KINGSIDE_START || t == BLACK_ROOK_KINGSIDE_START)
    castling &= ~BLACK_KINGSIDE;
}
void Position::handle_castling_rook_move(Square from, Square to, Color c) {
  if (to > from) {
    pieces[ROOK][c] &= ~bit(from + 3);
    pieces[ROOK][c] |= bit(from + 1);
    mailbox[from + 3] = -1;
    mailbox[from + 1] = ROOK * 2 + c;
    hash ^= zob_pieces[ROOK][c][from + 3];
    hash ^= zob_pieces[ROOK][c][from + 1];
  } else {
    pieces[ROOK][c] &= ~bit(from - 4);
    pieces[ROOK][c] |= bit(from - 1);
    mailbox[from - 4] = -1;
    mailbox[from - 1] = ROOK * 2 + c;
    hash ^= zob_pieces[ROOK][c][from - 4];
    hash ^= zob_pieces[ROOK][c][from - 1];
  }
}
void Position::execute_move_core(const Move &move, MoveInfo *undo) {
  const Square from = move.from(), to = move.to();
  if (undo) {
    undo->captured_piece = piece_at(to);
    undo->captured_square = to;
    undo->old_ep_square = ep_square;
    undo->old_castling = castling;
    undo->old_hash = hash;
    undo->old_occupancy_all = occupancy_all;
    undo->old_occupancy_color[WHITE] = occupancy_color[WHITE];
    undo->old_occupancy_color[BLACK] = occupancy_color[BLACK];
  }
  const int p = piece_at(from);
  if (undo)
    undo->moved_piece = p;
  const Piece pt = Piece(p / 2);
  const Color c = Color(p % 2);
  const bool is_pawn_promo =
      (pt == PAWN) &&
      ((c == WHITE && to / BOARD_SIZE == 7) ||
       (c == BLACK && to / BOARD_SIZE == 0));
  const Piece promo = move.promotion();
  const bool valid_promo = (promo == QUEEN || promo == ROOK ||
                            promo == BISHOP || promo == KNIGHT);
  hash ^= zob_pieces[pt][c][from];
  if (is_pawn_promo && valid_promo)
    hash ^= zob_pieces[promo][c][to];
  else
    hash ^= zob_pieces[pt][c][to];
  const int cap = piece_at(to);
  if (cap >= 0) {
    hash ^= zob_pieces[cap / 2][cap % 2][to];
    pieces[cap / 2][cap % 2] &= ~bit(to);
  }
  pieces[pt][c] &= ~bit(from);
  mailbox[from] = -1;
  if (is_pawn_promo && valid_promo) {
    pieces[promo][c] |= bit(to);
    mailbox[to] = promo * 2 + c;
  } else {
    pieces[pt][c] |= bit(to);
    mailbox[to] = pt * 2 + c;
  }
  if (pt == KING && std::abs(to - from) == KING_CASTLE_DISTANCE)
    handle_castling_rook_move(from, to, c);
  if (pt == KING) {
    if (c == WHITE)
      castling &= ~(WHITE_KINGSIDE | WHITE_QUEENSIDE);
    else
      castling &= ~(BLACK_KINGSIDE | BLACK_QUEENSIDE);
  }
  const uint8_t old_castling = castling;
  update_castling_rights(from, to);
  if (pt == PAWN && to == ep_square && ep_square != INVALID_SQUARE) {
    const Square cs = ep_square + (turn == WHITE ? -BOARD_SIZE : BOARD_SIZE);
    if (undo) {
      undo->captured_square = cs;
      undo->captured_piece = PAWN * 2 + (1 - turn);
    }
    pieces[PAWN][1 - turn] &= ~bit(cs);
    mailbox[cs] = -1;
    hash ^= zob_pieces[PAWN][1 - turn][cs];
  }
  if (undo) {
    hash ^= zob_castle[undo->old_castling];
    hash ^= zob_castle[castling];
    if (undo->old_ep_square != INVALID_SQUARE)
      hash ^= zobrist_ep[undo->old_ep_square];
  } else {
    const Square old_ep = ep_square;
    hash ^= zob_castle[old_castling];
    hash ^= zob_castle[castling];
    if (old_ep != INVALID_SQUARE)
      hash ^= zobrist_ep[old_ep];
  }
  ep_square = INVALID_SQUARE;
  if (pt == PAWN && std::abs(to - from) == PAWN_DOUBLE_MOVE_DISTANCE) {
    ep_square = (from + to) / 2;
    if (ep_square != INVALID_SQUARE) {
      const Color stm_next = Color(1 - turn);
      const Color opp = turn;
      const Square cs = ep_square + ((stm_next == WHITE) ? -BOARD_SIZE : BOARD_SIZE);
      const bool oppPawnBehind = (cs >= 0 && cs < NSQUARES) && (pieces[PAWN][opp] & bit(cs));
      const bool stmCanCapture = (get_pawn_attacks(ep_square, opp) & pieces[PAWN][stm_next]) != 0;
      if (!oppPawnBehind || !stmCanCapture)
        ep_square = INVALID_SQUARE;
    }
  }
  if (ep_square != INVALID_SQUARE)
    hash ^= zobrist_ep[ep_square];
  turn = Color(1 - turn);
  hash ^= zobrist_turn;
  update_occupancy();
}
Result Position::make_move(const Move &m) {
  MoveList legals;
  legal_moves(legals);
  bool is_legal = false;
  for (size_t i = 0; i < legals.size(); ++i) {
    if (legals[i] == m) { is_legal = true; break; }
  }
  if (!is_legal) return ONGOING;
  const int p = piece_at(m.from());
  const Piece pt = Piece(p / 2);
  const int cap = piece_at(m.to());
  execute_move_core(m);
  history.push_back(hash);
  if (cap >= 0 || pt == PAWN)
    halfmove = 0;
  else
    ++halfmove;
  if (turn == WHITE)
    ++fullmove;
  return result();
}
void Position::make_move_fast(const Move &m, MoveInfo &info) {
  info.old_halfmove = halfmove;
  info.old_fullmove = fullmove;
  execute_move_core(m, &info);
  history.push_back(hash);
  const int moved_type = (info.moved_piece >= 0) ? (info.moved_piece / 2) : -1;
  if (info.captured_piece >= 0 || moved_type == PAWN)
    halfmove = 0;
  else
    ++halfmove;
  if (turn == WHITE)
    ++fullmove;
}
void Position::unmake_move_fast(const Move &m, const MoveInfo &info) {
  Square from = m.from(), to = m.to();
  turn = Color(1 - turn);
  int p_to = piece_at(to);
  Color c = Color(p_to % 2);
  Piece pt_to = Piece(p_to / 2);
  const bool moved_was_pawn =
      (info.moved_piece >= 0) && (info.moved_piece / 2 == PAWN);
  const bool promo_requested =
      (m.promotion() == QUEEN || m.promotion() == ROOK ||
       m.promotion() == BISHOP || m.promotion() == KNIGHT);
  const bool reached_last =
      ((c == WHITE && to / BOARD_SIZE == 7) ||
       (c == BLACK && to / BOARD_SIZE == 0));
  const bool was_promo = moved_was_pawn && promo_requested && reached_last;
  pieces[pt_to][c] &= ~bit(to);
  if (was_promo) {
    pieces[PAWN][c] |= bit(from);
    mailbox[from] = PAWN * 2 + c;
  } else {
    pieces[pt_to][c] |= bit(from);
    mailbox[from] = pt_to * 2 + c;
  }
  mailbox[to] = -1;
  if (pt_to == KING && std::abs(to - from) == KING_CASTLE_DISTANCE) {
    if (to > from) {
      pieces[ROOK][c] &= ~bit(from + 1);
      pieces[ROOK][c] |= bit(from + 3);
      mailbox[from + 1] = -1;
      mailbox[from + 3] = ROOK * 2 + c;
    } else {
      pieces[ROOK][c] &= ~bit(from - 1);
      pieces[ROOK][c] |= bit(from - 4);
      mailbox[from - 1] = -1;
      mailbox[from - 4] = ROOK * 2 + c;
    }
  }
  if (info.captured_piece >= 0) {
    pieces[info.captured_piece / 2][info.captured_piece % 2] |=
        bit(info.captured_square);
    mailbox[info.captured_square] = info.captured_piece;
  }
  ep_square = info.old_ep_square;
  castling = info.old_castling;
  hash = info.old_hash;
  occupancy_all = info.old_occupancy_all;
  occupancy_color[WHITE] = info.old_occupancy_color[WHITE];
  occupancy_color[BLACK] = info.old_occupancy_color[BLACK];
  halfmove = info.old_halfmove;
  fullmove = info.old_fullmove;
  if (!history.empty())
    history.pop_back();
}
Result Position::result() {
  if (!pieces[KING][WHITE])
    return BLACK_WIN;
  if (!pieces[KING][BLACK])
    return WHITE_WIN;
  if (halfmove >= 100)
    return DRAW;
  if (count_repetitions() >= 3)
    return DRAW;
  if (is_insufficient_material())
    return DRAW;
  MoveList m;
  legal_moves(m);
  if (!m.empty())
    return ONGOING;
  Bitboard kb = pieces[KING][turn];
  const Square ks = lsb(kb);
  const Color e = Color(1 - turn);
  const bool check = is_square_attacked(ks, e);
  return check ? (turn == WHITE ? BLACK_WIN : WHITE_WIN) : DRAW;
}
int Position::count_repetitions() const {
  int c = 0;
  for (auto it = history.rbegin(); it != history.rend(); ++it)
    if (*it == hash) {
      ++c;
      if (c >= 3)
        break;
    }
  return c;
}
bool Position::is_insufficient_material() const {
  const int wp = popcount(pieces[PAWN][WHITE]);
  const int bp = popcount(pieces[PAWN][BLACK]);
  const int wr = popcount(pieces[ROOK][WHITE]);
  const int br = popcount(pieces[ROOK][BLACK]);
  const int wq = popcount(pieces[QUEEN][WHITE]);
  const int bq = popcount(pieces[QUEEN][BLACK]);
  const int wn = popcount(pieces[KNIGHT][WHITE]);
  const int bn = popcount(pieces[KNIGHT][BLACK]);
  const int wb = popcount(pieces[BISHOP][WHITE]);
  const int bb = popcount(pieces[BISHOP][BLACK]);

  if (wp || bp || wr || br || wq || bq)
    return false;

  const int wMinors = wn + wb;
  const int bMinors = bn + bb;

  if (wMinors == 0 && bMinors == 0)
    return true;

  if ((wMinors == 1 && bMinors == 0 && (wn == 1 || wb == 1)) ||
      (bMinors == 1 && wMinors == 0 && (bn == 1 || bb == 1)))
    return true;

  if ((wMinors == 2 && wn == 2 && bMinors == 0) ||
      (bMinors == 2 && bn == 2 && wMinors == 0))
    return true;
  return false;
}
} // namespace chess
