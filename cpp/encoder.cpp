#include "encoder.hpp"

#include <algorithm>

namespace encoder {

static inline void set_plane_value(float *out, int plane, int row, int col) {
  const size_t plane_area = 64;
  out[static_cast<size_t>(plane) * plane_area + static_cast<size_t>(row) * 8 +
      static_cast<size_t>(col)] = 1.0f;
}

static inline void
fill_repetition_planes(const std::vector<chess::Position> &hist, float *out) {
  for (int t = 0; t < static_cast<int>(hist.size()) && t < HISTORY_LENGTH;
       t++) {
    const int base = t * PLANES_PER_POSITION;
    int reps = 0;
    const auto thash = hist[static_cast<size_t>(t)].get_hash();
    for (int j = 0; j <= t; j++) {
      if (hist[static_cast<size_t>(j)].get_hash() == thash)
        reps++;
    }
    if (reps >= 2) {
      std::fill(out + static_cast<size_t>(base + 12) * 64,
                out + static_cast<size_t>(base + 13) * 64, 1.0f);
    }
    if (reps >= 3) {
      std::fill(out + static_cast<size_t>(base + 13) * 64,
                out + static_cast<size_t>(base + 14) * 64, 1.0f);
    }
  }
}

void encode_position_into(const chess::Position &pos, float *out) {
  constexpr int planes_per_position = PLANES_PER_POSITION;
  constexpr int history_length = HISTORY_LENGTH;
  constexpr int input_planes = INPUT_PLANES;
  const size_t total = static_cast<size_t>(input_planes) * 64;
  std::fill(out, out + total, 0.0f);

  for (int t = 0; t < history_length; t++) {
    const int base = t * planes_per_position;
    for (int piece = 0; piece < 6; piece++) {
      for (int color = 0; color < 2; color++) {
        const int plane = base + piece * 2 + color;
        chess::Bitboard bb = pos.get_pieces(static_cast<chess::Piece>(piece),
                                            static_cast<chess::Color>(color));
        while (bb) {
          const int square = chess::lsb(bb);
          chess::pop_lsb(bb);
          const int row = square >> 3;
          const int col = square & 7;
          set_plane_value(out, plane, row, col);
        }
      }
    }
  }

  const int turn_plane = history_length * planes_per_position;
  if (pos.get_turn() == chess::WHITE) {
    std::fill(out + static_cast<size_t>(turn_plane) * 64,
              out + static_cast<size_t>(turn_plane + 1) * 64, 1.0f);
  }

  const int fullmove_plane = turn_plane + 1;
  const float fullmove_val =
      std::min(1.0f, static_cast<float>(pos.get_fullmove()) / 100.0f);
  std::fill(out + static_cast<size_t>(fullmove_plane) * 64,
            out + static_cast<size_t>(fullmove_plane + 1) * 64, fullmove_val);

  const int castling_start_plane = fullmove_plane + 1;
  const uint8_t castling = pos.get_castling();
  if (castling & chess::WHITE_KINGSIDE) {
    std::fill(out + static_cast<size_t>(castling_start_plane + 0) * 64,
              out + static_cast<size_t>(castling_start_plane + 1) * 64, 1.0f);
  }
  if (castling & chess::WHITE_QUEENSIDE) {
    std::fill(out + static_cast<size_t>(castling_start_plane + 1) * 64,
              out + static_cast<size_t>(castling_start_plane + 2) * 64, 1.0f);
  }
  if (castling & chess::BLACK_KINGSIDE) {
    std::fill(out + static_cast<size_t>(castling_start_plane + 2) * 64,
              out + static_cast<size_t>(castling_start_plane + 3) * 64, 1.0f);
  }
  if (castling & chess::BLACK_QUEENSIDE) {
    std::fill(out + static_cast<size_t>(castling_start_plane + 3) * 64,
              out + static_cast<size_t>(castling_start_plane + 4) * 64, 1.0f);
  }

  const int halfmove_plane = castling_start_plane + 4;
  const float halfmove_val =
      std::min(1.0f, static_cast<float>(pos.get_halfmove()) / 100.0f);
  std::fill(out + static_cast<size_t>(halfmove_plane) * 64,
            out + static_cast<size_t>(halfmove_plane + 1) * 64, halfmove_val);
}

void encode_position_with_history(const std::vector<chess::Position> &history,
                                  float *out) {
  if (history.empty()) {
    chess::Position tmp;
    encode_position_into(tmp, out);
  } else {
    encode_position_into(history.back(), out);
  }
  fill_repetition_planes(history, out);
}

} // namespace encoder