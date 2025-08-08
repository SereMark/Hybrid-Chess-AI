#include "chess.hpp"

#include <algorithm>
#include <cctype>
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

alignas(16) const int ROOK_DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

alignas(16) const int BISHOP_DIRS[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

void init_tables() {
  static std::once_flag initialized;
  std::call_once(initialized, []() {
    std::mt19937_64 rng(0x1337BEEF);

    for (int piece = 0; piece < NUM_PIECE_TYPES; piece++) {
      for (int color = 0; color < NUM_COLORS; color++) {
        for (int square = 0; square < NSQUARES; square++) {
          zob_pieces[piece][color][square] = rng();
        }
      }
    }

    for (int i = 0; i < NCASTLING; i++) {
      zob_castle[i] = rng();
    }

    for (int i = 0; i < NSQUARES; i++) {
      zobrist_ep[i] = rng();
    }

    zobrist_turn = rng();

    for (int square = 0; square < NSQUARES; square++) {
      int row = square / BOARD_SIZE;
      int file = square % BOARD_SIZE;
      pawn_attack_table[WHITE][square] = 0;
      pawn_attack_table[BLACK][square] = 0;

      if (row < BOARD_SIZE - 1) {
        if (file > 0) {
          pawn_attack_table[WHITE][square] |= bit(square + 7);
        }
        if (file < BOARD_SIZE - 1) {
          pawn_attack_table[WHITE][square] |= bit(square + 9);
        }
      }

      if (row > 0) {
        if (file > 0) {
          pawn_attack_table[BLACK][square] |= bit(square - 9);
        }
        if (file < BOARD_SIZE - 1) {
          pawn_attack_table[BLACK][square] |= bit(square - 7);
        }
      }

      knight_attack_table[square] = 0;
      const int knight_moves[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                                      {1, -2},  {1, 2},  {2, -1},  {2, 1}};

      for (const auto &[delta_row, delta_file] : knight_moves) {
        int new_row = row + delta_row;
        int new_file = file + delta_file;
        if (new_row >= 0 && new_row < BOARD_SIZE && new_file >= 0 &&
            new_file < BOARD_SIZE) {
          knight_attack_table[square] |= bit(new_row * BOARD_SIZE + new_file);
        }
      }

      king_attack_table[square] = 0;
      for (int delta_row = -1; delta_row <= 1; delta_row++) {
        for (int delta_file = -1; delta_file <= 1; delta_file++) {
          if (delta_row == 0 && delta_file == 0) {
            continue;
          }
          int new_row = row + delta_row;
          int new_file = file + delta_file;
          if (new_row >= 0 && new_row < BOARD_SIZE && new_file >= 0 &&
              new_file < BOARD_SIZE) {
            king_attack_table[square] |= bit(new_row * BOARD_SIZE + new_file);
          }
        }
      }
    }
  });
}

Bitboard get_pawn_attacks(Square square, Color color) {
  return pawn_attack_table[color][square];
}

Bitboard knight_attacks(Square square) { return knight_attack_table[square]; }

Bitboard king_attacks(Square square) { return king_attack_table[square]; }

[[gnu::hot]]
static Bitboard sliding_attacks(Square square, Bitboard occupancy,
                                const int directions[][2], int num_directions) {
  Bitboard attacks = 0;
  const int row = square >> 3;
  const int file = square & 7;

  for (int i = 0; i < num_directions; i++) {
    const int delta_row = directions[i][0];
    const int delta_file = directions[i][1];
    int new_row = row + delta_row;
    int new_file = file + delta_file;

    while (__builtin_expect((new_row >= 0 && new_row < BOARD_SIZE &&
                             new_file >= 0 && new_file < BOARD_SIZE),
                            1)) {
      const Square target_square = (new_row << 3) | new_file;
      const Bitboard square_bit = 1ULL << target_square;
      attacks |= square_bit;

      if (__builtin_expect(!!(occupancy & square_bit), 0)) {
        break;
      }

      new_row += delta_row;
      new_file += delta_file;
    }
  }

  return attacks;
}

Bitboard bishop_attacks(Square square, Bitboard occupancy) {
  return sliding_attacks(square, occupancy, BISHOP_DIRS, 4);
}

Bitboard rook_attacks(Square square, Bitboard occupancy) {
  return sliding_attacks(square, occupancy, ROOK_DIRS, 4);
}

Bitboard queen_attacks(Square square, Bitboard occupancy) {
  return bishop_attacks(square, occupancy) | rook_attacks(square, occupancy);
}

Position::Position() {
  init_tables();
  for (int i = 0; i < NSQUARES; i++) {
    mailbox[i] = -1;
  }
  reset();
}

void Position::reset() {
  from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Position::clear_position() {
  for (int i = 0; i < NUM_PIECE_TYPES; i++) {
    for (int j = 0; j < NUM_COLORS; j++) {
      pieces[i][j] = 0;
    }
  }
  for (int i = 0; i < NSQUARES; i++) {
    mailbox[i] = -1;
  }
  history.clear();
}

void Position::parse_board_from_fen(const std::string &board_part) {
  Square square = 56;
  for (char character : board_part) {
    if (character == '/') {
      square -= 16;
    } else if (std::isdigit(character)) {
      int skip = character - '0';
      if (skip >= 1 && skip <= 8) {
        square += skip;
      }
    } else {
      const Color color = std::isupper(character) ? WHITE : BLACK;
      Piece piece;
      switch (std::tolower(character)) {
      case 'p':
        piece = PAWN;
        break;
      case 'n':
        piece = KNIGHT;
        break;
      case 'b':
        piece = BISHOP;
        break;
      case 'r':
        piece = ROOK;
        break;
      case 'q':
        piece = QUEEN;
        break;
      case 'k':
        piece = KING;
        break;
      default:
        continue;
      }
      if (square >= 0 && square < NSQUARES) {
        pieces[piece][color] |= bit(square);
        mailbox[square] = piece * 2 + color;
      }
      square++;
    }
  }
}

void Position::parse_game_state_from_fen(const std::string &side,
                                         const std::string &castle,
                                         const std::string &ep,
                                         uint16_t halfmove_count,
                                         uint16_t fullmove_count) {

  turn = (side == "w") ? WHITE : BLACK;
  castling = 0;
  if (castle.find('K') != std::string::npos) {
    castling |= WHITE_KINGSIDE;
  }
  if (castle.find('Q') != std::string::npos) {
    castling |= WHITE_QUEENSIDE;
  }
  if (castle.find('k') != std::string::npos) {
    castling |= BLACK_KINGSIDE;
  }
  if (castle.find('q') != std::string::npos) {
    castling |= BLACK_QUEENSIDE;
  }
  ep_square = (ep == "-") ? INVALID_SQUARE
                          : ((ep[0] - 'a') + (ep[1] - '1') * BOARD_SIZE);
  halfmove = halfmove_count;
  fullmove = fullmove_count;
}

void Position::finalize_position_setup() {
  update_occupancy();
  update_hash();
}

void Position::from_fen(const std::string &fen) {
  if (fen.empty()) {
    reset();
    return;
  }

  std::istringstream stream(fen);
  std::string board, side, castle, ep;
  uint16_t halfmove_count = 0, fullmove_count = 1;
  stream >> board >> side >> castle >> ep >> halfmove_count >> fullmove_count;

  if (board.empty() || side.empty()) {
    reset();
    return;
  }

  clear_position();
  parse_board_from_fen(board);
  parse_game_state_from_fen(side, castle, ep, halfmove_count, fullmove_count);
  finalize_position_setup();
}

std::string Position::serialize_board_to_fen() const {
  std::string board_fen;
  for (int row = BOARD_SIZE - 1; row >= 0; row--) {
    int empty = 0;
    for (int file = 0; file < BOARD_SIZE; file++) {
      const Square square = row * BOARD_SIZE + file;
      const int piece = piece_at(square);
      if (piece == -1) {
        empty++;
      } else {
        if (empty) {
          board_fen += std::to_string(empty);
          empty = 0;
        }
        const char pieces_string[] = "pnbrqk";
        char character = pieces_string[piece / 2];
        if (piece % 2 == 0) {
          character = std::toupper(character);
        }
        board_fen += character;
      }
    }
    if (empty) {
      board_fen += std::to_string(empty);
    }
    if (row > 0) {
      board_fen += '/';
    }
  }
  return board_fen;
}

std::string Position::serialize_game_state_to_fen() const {
  std::string state_fen;
  state_fen += (turn == WHITE) ? " w " : " b ";

  std::string castle_rights;
  if (castling & WHITE_KINGSIDE) {
    castle_rights += 'K';
  }
  if (castling & WHITE_QUEENSIDE) {
    castle_rights += 'Q';
  }
  if (castling & BLACK_KINGSIDE) {
    castle_rights += 'k';
  }
  if (castling & BLACK_QUEENSIDE) {
    castle_rights += 'q';
  }
  if (castle_rights.empty()) {
    castle_rights = "-";
  }
  state_fen += castle_rights + " ";

  if (ep_square == INVALID_SQUARE) {
    state_fen += "-";
  } else {
    state_fen += char('a' + (ep_square % BOARD_SIZE));
    state_fen += char('1' + (ep_square / BOARD_SIZE));
  }
  state_fen += " " + std::to_string(halfmove) + " " + std::to_string(fullmove);
  return state_fen;
}

std::string Position::to_fen() const {
  return serialize_board_to_fen() + serialize_game_state_to_fen();
}

void Position::update_hash() {
  hash = 0;
  for (int piece = 0; piece < NUM_PIECE_TYPES; piece++) {
    for (int color = 0; color < NUM_COLORS; color++) {
      Bitboard bitboard = pieces[piece][color];
      while (bitboard) {
        Square square = lsb(bitboard);
        bitboard &= bitboard - 1;
        hash ^= zob_pieces[piece][color][square];
      }
    }
  }
  hash ^= zob_castle[castling];
  if (ep_square != INVALID_SQUARE) {
    hash ^= zobrist_ep[ep_square];
  }
  if (turn == BLACK) {
    hash ^= zobrist_turn;
  }
}

void Position::update_occupancy() {
  occupancy_color[WHITE] = 0;
  occupancy_color[BLACK] = 0;
  for (int piece = 0; piece < NUM_PIECE_TYPES; piece++) {
    occupancy_color[WHITE] |= pieces[piece][WHITE];
    occupancy_color[BLACK] |= pieces[piece][BLACK];
  }
  occupancy_all = occupancy_color[WHITE] | occupancy_color[BLACK];
}

Bitboard Position::occupied() const { return occupancy_all; }

Bitboard Position::occupied(Color color) const {
  return occupancy_color[color];
}

int Position::piece_at(Square square) const {
  return (square & ~63) ? -1 : mailbox[square];
}

void Position::add_promotion_moves(Square from, Square to,
                                   MoveList &moves) const {
  for (Piece piece = QUEEN; piece >= KNIGHT; piece = Piece(piece - 1)) {
    moves.add(from, to, piece);
  }
}

void Position::add_pawn_moves(Square from, MoveList &moves) const {
  const int direction = (turn == WHITE) ? BOARD_SIZE : -BOARD_SIZE;
  const int start_rank = (turn == WHITE) ? 1 : 6;
  const int promo_rank = (turn == WHITE) ? 6 : 1;
  const Bitboard occupancy = occupied();
  const Bitboard enemies = occupied(Color(1 - turn));

  const Square one_step = from + direction;
  if (one_step >= 0 && one_step < NSQUARES && !(occupancy & bit(one_step))) {
    if (from / BOARD_SIZE == promo_rank) {
      add_promotion_moves(from, one_step, moves);
    } else {
      moves.add(from, one_step);
    }

    if (from / BOARD_SIZE == start_rank) {
      const Square two_step = from + direction * 2;
      if (!(occupancy & bit(two_step))) {
        moves.add(from, two_step);
      }
    }
  }

  const Bitboard attacks = get_pawn_attacks(from, turn) & enemies;
  for_each_attack(attacks, [&](Square to) {
    if (from / BOARD_SIZE == promo_rank) {
      add_promotion_moves(from, to, moves);
    } else {
      moves.add(from, to);
    }
  });

  if (ep_square != INVALID_SQUARE &&
      (get_pawn_attacks(from, turn) & bit(ep_square))) {
    moves.add(from, ep_square);
  }
}

bool Position::is_square_attacked(Square square, Color by_color) const {
  const Bitboard occupancy = occupied();
  return (get_pawn_attacks(square, Color(1 - by_color)) &
          pieces[PAWN][by_color]) ||
         (knight_attacks(square) & pieces[KNIGHT][by_color]) ||
         (bishop_attacks(square, occupancy) &
          (pieces[BISHOP][by_color] | pieces[QUEEN][by_color])) ||
         (rook_attacks(square, occupancy) &
          (pieces[ROOK][by_color] | pieces[QUEEN][by_color])) ||
         (king_attacks(square) & pieces[KING][by_color]);
}

void Position::generate_piece_moves(Piece piece, MoveList &moves) const {
  const Bitboard us = occupied(turn);
  const Bitboard occupancy = occupied();

  for_each_piece(pieces[piece][turn], [&](Square from) {
    Bitboard attacks;
    switch (piece) {
    case KNIGHT:
      attacks = knight_attacks(from);
      break;
    case BISHOP:
      attacks = bishop_attacks(from, occupancy);
      break;
    case ROOK:
      attacks = rook_attacks(from, occupancy);
      break;
    case QUEEN:
      attacks = queen_attacks(from, occupancy);
      break;
    case KING:
      attacks = king_attacks(from);
      break;
    default:
      return;
    }
    attacks &= ~us;
    for_each_attack(attacks, [&](Square to) { moves.add(from, to); });
  });
}

void Position::legal_moves(MoveList &moves) {
  moves.clear();

  for_each_piece(pieces[PAWN][turn],
                 [&](Square from) { add_pawn_moves(from, moves); });
  generate_piece_moves(KNIGHT, moves);
  generate_piece_moves(BISHOP, moves);
  generate_piece_moves(ROOK, moves);
  generate_piece_moves(QUEEN, moves);
  generate_piece_moves(KING, moves);

  if (pieces[KING][turn]) {
    const Square king_position = lsb(pieces[KING][turn]);
    const Color enemy = Color(1 - turn);

    if (turn == WHITE) {
      if ((castling & WHITE_KINGSIDE) && !(occupied() & WHITE_KINGSIDE_CLEAR) &&
          king_position == WHITE_KING_START &&
          !is_square_attacked(WHITE_KING_START, enemy) &&
          !is_square_attacked(F1, enemy) &&
          !is_square_attacked(WHITE_KING_KINGSIDE, enemy)) {
        moves.add(WHITE_KING_START, WHITE_KING_KINGSIDE);
      }
      if ((castling & WHITE_QUEENSIDE) &&
          !(occupied() & WHITE_QUEENSIDE_CLEAR) &&
          king_position == WHITE_KING_START &&
          !is_square_attacked(WHITE_KING_START, enemy) &&
          !is_square_attacked(D1, enemy) &&
          !is_square_attacked(WHITE_KING_QUEENSIDE, enemy)) {
        moves.add(WHITE_KING_START, WHITE_KING_QUEENSIDE);
      }
    } else {
      if ((castling & BLACK_KINGSIDE) && !(occupied() & BLACK_KINGSIDE_CLEAR) &&
          king_position == BLACK_KING_START &&
          !is_square_attacked(BLACK_KING_START, enemy) &&
          !is_square_attacked(F8, enemy) &&
          !is_square_attacked(BLACK_KING_KINGSIDE, enemy)) {
        moves.add(BLACK_KING_START, BLACK_KING_KINGSIDE);
      }
      if ((castling & BLACK_QUEENSIDE) &&
          !(occupied() & BLACK_QUEENSIDE_CLEAR) &&
          king_position == BLACK_KING_START &&
          !is_square_attacked(BLACK_KING_START, enemy) &&
          !is_square_attacked(D8, enemy) &&
          !is_square_attacked(BLACK_KING_QUEENSIDE, enemy)) {
        moves.add(BLACK_KING_START, BLACK_KING_QUEENSIDE);
      }
    }
  }

  MoveList legal;
  for (size_t i = 0; i < moves.count; i++) {
    const Move &move = moves[i];
    if (piece_at(move.from()) == -1) {
      continue;
    }
    MoveInfo info;
    make_move_fast(move, info);
    bool is_legal = true;
    if (const Bitboard king_bitboard = pieces[KING][Color(1 - turn)]) {
      const Square king_square = lsb(king_bitboard);
      is_legal = !is_square_attacked(king_square, turn);
    }
    unmake_move_fast(move, info);
    if (is_legal) {
      legal.add(move);
    }
  }
  moves = legal;
}

MoveList Position::legal_moves() {
  MoveList moves;
  legal_moves(moves);
  return moves;
}

void Position::update_castling_rights(Square from, Square to) {
  if (from == WHITE_ROOK_QUEENSIDE_START || to == WHITE_ROOK_QUEENSIDE_START) {
    castling &= ~WHITE_QUEENSIDE;
  }
  if (from == WHITE_ROOK_KINGSIDE_START || to == WHITE_ROOK_KINGSIDE_START) {
    castling &= ~WHITE_KINGSIDE;
  }
  if (from == BLACK_ROOK_QUEENSIDE_START || to == BLACK_ROOK_QUEENSIDE_START) {
    castling &= ~BLACK_QUEENSIDE;
  }
  if (from == BLACK_ROOK_KINGSIDE_START || to == BLACK_ROOK_KINGSIDE_START) {
    castling &= ~BLACK_KINGSIDE;
  }
}

void Position::handle_castling_rook_move(Square from, Square to, Color color) {
  if (to > from) {
    pieces[ROOK][color] &= ~bit(from + 3);
    pieces[ROOK][color] |= bit(from + 1);
    mailbox[from + 3] = -1;
    mailbox[from + 1] = ROOK * 2 + color;
    hash ^= zob_pieces[ROOK][color][from + 3];
    hash ^= zob_pieces[ROOK][color][from + 1];
  } else {
    pieces[ROOK][color] &= ~bit(from - 4);
    pieces[ROOK][color] |= bit(from - 1);
    mailbox[from - 4] = -1;
    mailbox[from - 1] = ROOK * 2 + color;
    hash ^= zob_pieces[ROOK][color][from - 4];
    hash ^= zob_pieces[ROOK][color][from - 1];
  }
}

void Position::execute_move_core(const Move &move, MoveInfo *undo_info) {
  const Square from = move.from();
  const Square to = move.to();

  if (undo_info) {
    undo_info->captured_piece = piece_at(to);
    undo_info->captured_square = to;
    undo_info->old_ep_square = ep_square;
    undo_info->old_castling = castling;
    undo_info->old_hash = hash;
    undo_info->old_occupancy_all = occupancy_all;
    undo_info->old_occupancy_color[WHITE] = occupancy_color[WHITE];
    undo_info->old_occupancy_color[BLACK] = occupancy_color[BLACK];
  }

  const int piece = piece_at(from);
  const Piece piece_type = Piece(piece / 2);
  const Color color = Color(piece % 2);

  hash ^= zob_pieces[piece_type][color][from];
  if (move.promotion() <= KING) {
    hash ^= zob_pieces[move.promotion()][color][to];
  } else {
    hash ^= zob_pieces[piece_type][color][to];
  }

  const int captured = piece_at(to);
  if (captured >= 0) {
    hash ^= zob_pieces[captured / 2][captured % 2][to];
    pieces[captured / 2][captured % 2] &= ~bit(to);
  }

  pieces[piece_type][color] &= ~bit(from);
  mailbox[from] = -1;

  if (move.promotion() <= KING) {
    pieces[move.promotion()][color] |= bit(to);
    mailbox[to] = move.promotion() * 2 + color;
  } else {
    pieces[piece_type][color] |= bit(to);
    mailbox[to] = piece_type * 2 + color;
  }

  if (piece_type == KING && abs(to - from) == 2) {
    handle_castling_rook_move(from, to, color);
  }

  if (piece_type == KING) {
    if (color == WHITE) {
      castling &= ~(WHITE_KINGSIDE | WHITE_QUEENSIDE);
    } else {
      castling &= ~(BLACK_KINGSIDE | BLACK_QUEENSIDE);
    }
  }

  update_castling_rights(from, to);

  if (piece_type == PAWN && to == ep_square && ep_square != INVALID_SQUARE) {
    const Square capture_square =
        ep_square + (turn == WHITE ? -BOARD_SIZE : BOARD_SIZE);
    if (undo_info) {
      undo_info->captured_square = capture_square;
      undo_info->captured_piece = PAWN * 2 + (1 - turn);
    }
    pieces[PAWN][1 - turn] &= ~bit(capture_square);
    mailbox[capture_square] = -1;
    hash ^= zob_pieces[PAWN][1 - turn][capture_square];
  }

  if (undo_info) {
    hash ^= zob_castle[undo_info->old_castling];
    hash ^= zob_castle[castling];
    if (undo_info->old_ep_square != INVALID_SQUARE) {
      hash ^= zobrist_ep[undo_info->old_ep_square];
    }
  } else {
    hash ^= zob_castle[castling];
    if (ep_square != INVALID_SQUARE) {
      hash ^= zobrist_ep[ep_square];
    }
  }

  ep_square = INVALID_SQUARE;
  if (piece_type == PAWN && abs(to - from) == 16) {
    ep_square = (from + to) / 2;
  }

  if (ep_square != INVALID_SQUARE) {
    hash ^= zobrist_ep[ep_square];
  }

  turn = Color(1 - turn);
  hash ^= zobrist_turn;
  update_occupancy();
}

Result Position::make_move(const Move &move) {
  if (piece_at(move.from()) == -1) {
    return ONGOING;
  }

  history.push_back(hash);
  const int piece = piece_at(move.from());
  const Piece piece_type = Piece(piece / 2);
  const int captured = piece_at(move.to());

  execute_move_core(move);

  if (captured >= 0 || piece_type == PAWN) {
    halfmove = 0;
  } else {
    halfmove++;
  }

  if (turn == WHITE) {
    fullmove++;
  }

  return result();
}

void Position::make_move_fast(const Move &move, MoveInfo &info) {
  execute_move_core(move, &info);
}

void Position::unmake_move_fast(const Move &move, const MoveInfo &info) {
  Square from = move.from();
  Square to = move.to();
  turn = Color(1 - turn);
  int piece = piece_at(to);
  Piece piece_type =
      (move.promotion() <= KING) ? move.promotion() : Piece(piece / 2);
  Color color = Color(piece % 2);

  pieces[piece_type][color] &= ~bit(to);
  if (move.promotion() <= KING) {
    pieces[PAWN][color] |= bit(from);
    mailbox[from] = PAWN * 2 + color;
  } else {
    pieces[piece_type][color] |= bit(from);
    mailbox[from] = piece_type * 2 + color;
  }

  if (info.captured_piece >= 0) {
    pieces[info.captured_piece / 2][info.captured_piece % 2] |=
        bit(info.captured_square);
    mailbox[info.captured_square] = info.captured_piece;
  }

  if (info.captured_piece == -1) {
    mailbox[to] = -1;
  }

  if (piece_type == KING && abs(to - from) == 2) {
    if (to > from) {
      pieces[ROOK][color] &= ~bit(from + 1);
      pieces[ROOK][color] |= bit(from + 3);
      mailbox[from + 1] = -1;
      mailbox[from + 3] = ROOK * 2 + color;
    } else {
      pieces[ROOK][color] &= ~bit(from - 1);
      pieces[ROOK][color] |= bit(from - 4);
      mailbox[from - 1] = -1;
      mailbox[from - 4] = ROOK * 2 + color;
    }
  }

  ep_square = info.old_ep_square;
  castling = info.old_castling;
  hash = info.old_hash;
  occupancy_all = info.old_occupancy_all;
  occupancy_color[WHITE] = info.old_occupancy_color[WHITE];
  occupancy_color[BLACK] = info.old_occupancy_color[BLACK];
}

Result Position::result() {
  if (halfmove >= 100) {
    return DRAW;
  }
  if (count_repetitions() >= 2) {
    return DRAW;
  }
  if (is_insufficient_material()) {
    return DRAW;
  }

  MoveList moves;
  legal_moves(moves);
  if (!moves.empty()) {
    return ONGOING;
  }

  Bitboard king_bitboard = pieces[KING][turn];
  if (!king_bitboard) {
    return turn == WHITE ? BLACK_WIN : WHITE_WIN;
  }

  const Square king_square = lsb(king_bitboard);
  const Color enemy = Color(1 - turn);
  const bool in_check = is_square_attacked(king_square, enemy);

  if (in_check) {
    return turn == WHITE ? BLACK_WIN : WHITE_WIN;
  } else {
    return DRAW;
  }
}

int Position::count_repetitions() const {
  int count = 0;
  for (Hash hash_value : history) {
    if (hash_value == hash) {
      count++;
    }
  }
  return count;
}

bool Position::is_insufficient_material() const {
  int white_pawns = popcount(pieces[PAWN][WHITE]);
  int white_knights = popcount(pieces[KNIGHT][WHITE]);
  int white_bishops = popcount(pieces[BISHOP][WHITE]);
  int white_rooks = popcount(pieces[ROOK][WHITE]);
  int white_queens = popcount(pieces[QUEEN][WHITE]);

  int black_pawns = popcount(pieces[PAWN][BLACK]);
  int black_knights = popcount(pieces[KNIGHT][BLACK]);
  int black_bishops = popcount(pieces[BISHOP][BLACK]);
  int black_rooks = popcount(pieces[ROOK][BLACK]);
  int black_queens = popcount(pieces[QUEEN][BLACK]);

  if (white_pawns || black_pawns || white_rooks || black_rooks ||
      white_queens || black_queens) {
    return false;
  }

  if (white_knights == 0 && white_bishops == 0 && black_knights == 0 &&
      black_bishops == 0) {
    return true;
  }

  if ((white_knights == 1 && white_bishops == 0 && black_knights == 0 &&
       black_bishops == 0) ||
      (black_knights == 1 && black_bishops == 0 && white_knights == 0 &&
       white_bishops == 0)) {
    return true;
  }

  if ((white_bishops == 1 && white_knights == 0 && black_knights == 0 &&
       black_bishops == 0) ||
      (black_bishops == 1 && black_knights == 0 && white_knights == 0 &&
       white_bishops == 0)) {
    return true;
  }

  if (white_knights == 0 && black_knights == 0 && white_bishops > 0 &&
      black_bishops > 0) {
    if (white_bishops <= 1 && black_bishops <= 1) {
      return true;
    }
  }

  return false;
}

}