#include "encoder.hpp"

#include <algorithm>

namespace encoder {

static inline void set_plane_value(float *out, int plane, int row, int col) {
  const size_t plane_area = static_cast<size_t>(chess::BOARD_SIZE) *
                            static_cast<size_t>(chess::BOARD_SIZE);
  out[static_cast<size_t>(plane) * plane_area +
      static_cast<size_t>(row) * static_cast<size_t>(chess::BOARD_SIZE) +
      static_cast<size_t>(col)] = 1.0f;
}

static inline void
fill_repetition_planes(const std::vector<chess::Position> &hist, float *out) {
  const int limit = std::min(HISTORY_LENGTH, static_cast<int>(hist.size()));
  for (int t = 0; t < limit; t++) {
    const int base = t * PLANES_PER_POSITION;
    const int idx = static_cast<int>(hist.size()) - 1 - t;
    int reps = 0;
    const auto thash = hist[static_cast<size_t>(idx)].get_hash();
    for (int j = 0; j <= idx; j++) {
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
  const size_t plane_area = static_cast<size_t>(chess::BOARD_SIZE) *
                            static_cast<size_t>(chess::BOARD_SIZE);
  const size_t total = static_cast<size_t>(input_planes) * plane_area;
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
    std::fill(out + static_cast<size_t>(turn_plane) * plane_area,
              out + static_cast<size_t>(turn_plane + 1) * plane_area, 1.0f);
  }

  const int fullmove_plane = turn_plane + 1;
  const float fullmove_val =
      std::min(1.0f, static_cast<float>(pos.get_fullmove()) / 100.0f);
  std::fill(out + static_cast<size_t>(fullmove_plane) * plane_area,
            out + static_cast<size_t>(fullmove_plane + 1) * plane_area,
            fullmove_val);

  const int castling_start_plane = fullmove_plane + 1;
  const uint8_t castling = pos.get_castling();
  if (castling & chess::WHITE_KINGSIDE) {
    std::fill(out + static_cast<size_t>(castling_start_plane + 0) * plane_area,
              out + static_cast<size_t>(castling_start_plane + 1) * plane_area,
              1.0f);
  }
  if (castling & chess::WHITE_QUEENSIDE) {
    std::fill(out + static_cast<size_t>(castling_start_plane + 1) * plane_area,
              out + static_cast<size_t>(castling_start_plane + 2) * plane_area,
              1.0f);
  }
  if (castling & chess::BLACK_KINGSIDE) {
    std::fill(out + static_cast<size_t>(castling_start_plane + 2) * plane_area,
              out + static_cast<size_t>(castling_start_plane + 3) * plane_area,
              1.0f);
  }
  if (castling & chess::BLACK_QUEENSIDE) {
    std::fill(out + static_cast<size_t>(castling_start_plane + 3) * plane_area,
              out + static_cast<size_t>(castling_start_plane + 4) * plane_area,
              1.0f);
  }

  const int halfmove_plane = castling_start_plane + 4;
  const float halfmove_val =
      std::min(1.0f, static_cast<float>(pos.get_halfmove()) / 100.0f);
  std::fill(out + static_cast<size_t>(halfmove_plane) * plane_area,
            out + static_cast<size_t>(halfmove_plane + 1) * plane_area,
            halfmove_val);
}

void encode_position_with_history(const std::vector<chess::Position> &history,
                                  float *out) {
  const int planes_per_position = PLANES_PER_POSITION;
  const int history_length = HISTORY_LENGTH;
  const size_t plane_area = static_cast<size_t>(chess::BOARD_SIZE) *
                            static_cast<size_t>(chess::BOARD_SIZE);
  const size_t total = static_cast<size_t>(INPUT_PLANES) * plane_area;
  std::fill(out, out + total, 0.0f);

  const int available =
      std::min(history_length, static_cast<int>(history.size()));
  for (int t = 0; t < available; t++) {
    const int base = t * planes_per_position;
    const chess::Position &pos =
        history[static_cast<size_t>(history.size() - 1 - t)];
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
  if (!history.empty() && history.back().get_turn() == chess::WHITE) {
    std::fill(out + static_cast<size_t>(turn_plane) * plane_area,
              out + static_cast<size_t>(turn_plane + 1) * plane_area, 1.0f);
  }

  const int fullmove_plane = turn_plane + 1;
  chess::Position default_pos;
  const chess::Position &cur = history.empty() ? default_pos : history.back();
  const float fullmove_val =
      std::min(1.0f, static_cast<float>(cur.get_fullmove()) / 100.0f);
  std::fill(out + static_cast<size_t>(fullmove_plane) * plane_area,
            out + static_cast<size_t>(fullmove_plane + 1) * plane_area,
            fullmove_val);

  const int castling_start_plane = fullmove_plane + 1;
  const uint8_t castling = cur.get_castling();
  if (castling & chess::WHITE_KINGSIDE) {
    std::fill(out + static_cast<size_t>(castling_start_plane + 0) * plane_area,
              out + static_cast<size_t>(castling_start_plane + 1) * plane_area,
              1.0f);
  }
  if (castling & chess::WHITE_QUEENSIDE) {
    std::fill(out + static_cast<size_t>(castling_start_plane + 1) * plane_area,
              out + static_cast<size_t>(castling_start_plane + 2) * plane_area,
              1.0f);
  }
  if (castling & chess::BLACK_KINGSIDE) {
    std::fill(out + static_cast<size_t>(castling_start_plane + 2) * plane_area,
              out + static_cast<size_t>(castling_start_plane + 3) * plane_area,
              1.0f);
  }
  if (castling & chess::BLACK_QUEENSIDE) {
    std::fill(out + static_cast<size_t>(castling_start_plane + 3) * plane_area,
              out + static_cast<size_t>(castling_start_plane + 4) * plane_area,
              1.0f);
  }

  const int halfmove_plane = castling_start_plane + 4;
  const float halfmove_val =
      std::min(1.0f, static_cast<float>(cur.get_halfmove()) / 100.0f);
  std::fill(out + static_cast<size_t>(halfmove_plane) * plane_area,
            out + static_cast<size_t>(halfmove_plane + 1) * plane_area,
            halfmove_val);

  fill_repetition_planes(history, out);
}

} // namespace encoder