#include "encoder.hpp"
#include <algorithm>

namespace encoder {
static inline void set_plane_value(float *out, int plane, int row, int col) {
  const size_t A = static_cast<size_t>(chess::BOARD_SIZE) *
                   static_cast<size_t>(chess::BOARD_SIZE);
  out[static_cast<size_t>(plane) * A +
      static_cast<size_t>(row) * static_cast<size_t>(chess::BOARD_SIZE) +
      static_cast<size_t>(col)] = 1.0f;
}

static inline void
fill_repetition_planes(const std::vector<chess::Position> &hist, float *out) {
  static constexpr size_t A = static_cast<size_t>(chess::BOARD_SIZE) *
                              static_cast<size_t>(chess::BOARD_SIZE);
  const int lim = std::min(HISTORY_LENGTH, static_cast<int>(hist.size()));
  for (int t = 0; t < lim; ++t) {
    const int base = t * PLANES_PER_POSITION,
              idx = static_cast<int>(hist.size()) - 1 - t;
    int reps = 0;
    const auto h = hist[static_cast<size_t>(idx)].get_hash();
    for (int j = 0; j <= idx; ++j)
      if (hist[static_cast<size_t>(j)].get_hash() == h)
        ++reps;
    if (reps >= 2)
      std::fill(out + static_cast<size_t>(base + 12) * A,
                out + static_cast<size_t>(base + 13) * A, 1.0f);
    if (reps >= 3)
      std::fill(out + static_cast<size_t>(base + 13) * A,
                out + static_cast<size_t>(base + 14) * A, 1.0f);
  }
}

void encode_position_into(const chess::Position &pos, float *out) {
  constexpr int ppp = PLANES_PER_POSITION, H = HISTORY_LENGTH, P = INPUT_PLANES;
  const size_t A = static_cast<size_t>(chess::BOARD_SIZE) *
                   static_cast<size_t>(chess::BOARD_SIZE),
               T = static_cast<size_t>(P) * A;
  std::fill(out, out + T, 0.0f);
  for (int t = 0; t < H; ++t) {
    const int base = t * ppp;
    for (int piece = 0; piece < 6; ++piece)
      for (int color = 0; color < 2; ++color) {
        const int plane = base + piece * 2 + color;
        chess::Bitboard bb = pos.get_pieces(static_cast<chess::Piece>(piece),
                                            static_cast<chess::Color>(color));
        while (bb) {
          const int s = chess::lsb(bb);
          chess::pop_lsb(bb);
          set_plane_value(out, plane, s >> 3, s & 7);
        }
      }
  }
  const int turn_plane = H * ppp;
  if (pos.get_turn() == chess::WHITE)
    std::fill(out + static_cast<size_t>(turn_plane) * A,
              out + static_cast<size_t>(turn_plane + 1) * A, 1.0f);
  const int fullmove_plane = turn_plane + 1;
  const float fm =
      std::min(1.0f, static_cast<float>(pos.get_fullmove()) / 100.0f);
  std::fill(out + static_cast<size_t>(fullmove_plane) * A,
            out + static_cast<size_t>(fullmove_plane + 1) * A, fm);
  const int c0 = fullmove_plane + 1;
  const uint8_t c = pos.get_castling();
  if (c & chess::WHITE_KINGSIDE)
    std::fill(out + static_cast<size_t>(c0 + 0) * A,
              out + static_cast<size_t>(c0 + 1) * A, 1.0f);
  if (c & chess::WHITE_QUEENSIDE)
    std::fill(out + static_cast<size_t>(c0 + 1) * A,
              out + static_cast<size_t>(c0 + 2) * A, 1.0f);
  if (c & chess::BLACK_KINGSIDE)
    std::fill(out + static_cast<size_t>(c0 + 2) * A,
              out + static_cast<size_t>(c0 + 3) * A, 1.0f);
  if (c & chess::BLACK_QUEENSIDE)
    std::fill(out + static_cast<size_t>(c0 + 3) * A,
              out + static_cast<size_t>(c0 + 4) * A, 1.0f);
  const int halfmove_plane = c0 + 4;
  const float hm =
      std::min(1.0f, static_cast<float>(pos.get_halfmove()) / 100.0f);
  std::fill(out + static_cast<size_t>(halfmove_plane) * A,
            out + static_cast<size_t>(halfmove_plane + 1) * A, hm);
}

void encode_position_with_history(const std::vector<chess::Position> &history,
                                  float *out) {
  const int ppp = PLANES_PER_POSITION, H = HISTORY_LENGTH;
  const size_t A = static_cast<size_t>(chess::BOARD_SIZE) *
                   static_cast<size_t>(chess::BOARD_SIZE),
               T = static_cast<size_t>(INPUT_PLANES) * A;
  std::fill(out, out + T, 0.0f);
  const int avail = std::min(H, static_cast<int>(history.size()));
  for (int t = 0; t < avail; ++t) {
    const int base = t * ppp;
    const chess::Position &pos =
        history[static_cast<size_t>(history.size() - 1 - t)];
    for (int piece = 0; piece < 6; ++piece)
      for (int color = 0; color < 2; ++color) {
        const int plane = base + piece * 2 + color;
        chess::Bitboard bb = pos.get_pieces(static_cast<chess::Piece>(piece),
                                            static_cast<chess::Color>(color));
        while (bb) {
          const int s = chess::lsb(bb);
          chess::pop_lsb(bb);
          set_plane_value(out, plane, s >> 3, s & 7);
        }
      }
  }
  const int turn_plane = H * ppp;
  if (!history.empty() && history.back().get_turn() == chess::WHITE)
    std::fill(out + static_cast<size_t>(turn_plane) * A,
              out + static_cast<size_t>(turn_plane + 1) * A, 1.0f);
  const int fullmove_plane = turn_plane + 1;
  chess::Position def;
  const chess::Position &cur = history.empty() ? def : history.back();
  const float fm =
      std::min(1.0f, static_cast<float>(cur.get_fullmove()) / 100.0f);
  std::fill(out + static_cast<size_t>(fullmove_plane) * A,
            out + static_cast<size_t>(fullmove_plane + 1) * A, fm);
  const int c0 = fullmove_plane + 1;
  const uint8_t c = cur.get_castling();
  if (c & chess::WHITE_KINGSIDE)
    std::fill(out + static_cast<size_t>(c0 + 0) * A,
              out + static_cast<size_t>(c0 + 1) * A, 1.0f);
  if (c & chess::WHITE_QUEENSIDE)
    std::fill(out + static_cast<size_t>(c0 + 1) * A,
              out + static_cast<size_t>(c0 + 2) * A, 1.0f);
  if (c & chess::BLACK_KINGSIDE)
    std::fill(out + static_cast<size_t>(c0 + 2) * A,
              out + static_cast<size_t>(c0 + 3) * A, 1.0f);
  if (c & chess::BLACK_QUEENSIDE)
    std::fill(out + static_cast<size_t>(c0 + 3) * A,
              out + static_cast<size_t>(c0 + 4) * A, 1.0f);
  const int halfmove_plane = c0 + 4;
  const float hm =
      std::min(1.0f, static_cast<float>(cur.get_halfmove()) / 100.0f);
  std::fill(out + static_cast<size_t>(halfmove_plane) * A,
            out + static_cast<size_t>(halfmove_plane + 1) * A, hm);
  fill_repetition_planes(history, out);
}
} // namespace encoder
