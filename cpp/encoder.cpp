#include "encoder.hpp"

#include <algorithm>
#include <cstdint>
namespace encoder {
static inline void set_plane_value(float *out, int plane, int row, int col) {
    const size_t board_area =
        static_cast<size_t>(chess::BOARD_SIZE) * static_cast<size_t>(chess::BOARD_SIZE);
    out[static_cast<size_t>(plane) * board_area +
        static_cast<size_t>(row) * static_cast<size_t>(chess::BOARD_SIZE) +
        static_cast<size_t>(col)] = 1.0f;
}
static inline void fill_repetition_planes(const std::vector<chess::Position> &hist, float *out) {
    static constexpr size_t A =
        static_cast<size_t>(chess::BOARD_SIZE) * static_cast<size_t>(chess::BOARD_SIZE);
    const int lim = std::min(HISTORY_LENGTH, static_cast<int>(hist.size()));
    for (int t = 0; t < lim; ++t) {
        const int base  = t * PLANES_PER_POSITION;
        const int idx   = static_cast<int>(hist.size()) - 1 - t;
        const auto &pos = hist[static_cast<size_t>(idx)];
        const auto h    = pos.get_hash();
        int start       = idx - pos.get_halfmove();
        if (start < 0)
            start = 0;
        int reps = 0;
        for (int j = start; j <= idx; ++j) {
            if (hist[static_cast<size_t>(j)].get_hash() == h)
                ++reps;
        }
        if (reps >= 2)
            std::fill(out + static_cast<size_t>(base + 12) * A,
                      out + static_cast<size_t>(base + 13) * A, 1.0f);
        if (reps >= 3)
            std::fill(out + static_cast<size_t>(base + 13) * A,
                      out + static_cast<size_t>(base + 14) * A, 1.0f);
    }
}
void encode_position_into(const chess::Position &pos, float *out) {
    constexpr int ppp = PLANES_PER_POSITION;
    constexpr int H   = HISTORY_LENGTH;
    constexpr int P   = INPUT_PLANES;
    const size_t board_area =
        static_cast<size_t>(chess::BOARD_SIZE) * static_cast<size_t>(chess::BOARD_SIZE);
    const size_t total_elems = static_cast<size_t>(P) * board_area;
    std::fill(out, out + total_elems, 0.0f);
    {
        const int base = 0;
        for (int piece = 0; piece < 6; ++piece) {
            for (int color = 0; color < 2; ++color) {
                const int plane    = base + piece * 2 + color;
                chess::Bitboard bb = pos.get_pieces(static_cast<chess::Piece>(piece),
                                                    static_cast<chess::Color>(color));
                while (bb) {
                    const int s = chess::lsb(bb);
                    chess::pop_lsb(bb);
                    set_plane_value(out, plane, s >> 3, s & 7);
                }
            }
        }
        const int reps = pos.repetition_count();
        if (reps >= 2)
            std::fill(out + static_cast<size_t>(base + 12) * board_area,
                      out + static_cast<size_t>(base + 13) * board_area, 1.0f);
        if (reps >= 3)
            std::fill(out + static_cast<size_t>(base + 13) * board_area,
                      out + static_cast<size_t>(base + 14) * board_area, 1.0f);
    }
    const int turn_plane = H * ppp;
    if (pos.get_turn() == chess::WHITE)
        std::fill(out + static_cast<size_t>(turn_plane) * board_area,
                  out + static_cast<size_t>(turn_plane + 1) * board_area, 1.0f);
    const int fullmove_plane = turn_plane + 1;
    const float fm           = std::min(1.0f, static_cast<float>(pos.get_fullmove()) / 100.0f);
    std::fill(out + static_cast<size_t>(fullmove_plane) * board_area,
              out + static_cast<size_t>(fullmove_plane + 1) * board_area, fm);
    const int c0    = fullmove_plane + 1;
    const uint8_t c = pos.get_castling();
    if (c & chess::WHITE_KINGSIDE)
        std::fill(out + static_cast<size_t>(c0 + 0) * board_area,
                  out + static_cast<size_t>(c0 + 1) * board_area, 1.0f);
    if (c & chess::WHITE_QUEENSIDE)
        std::fill(out + static_cast<size_t>(c0 + 1) * board_area,
                  out + static_cast<size_t>(c0 + 2) * board_area, 1.0f);
    if (c & chess::BLACK_KINGSIDE)
        std::fill(out + static_cast<size_t>(c0 + 2) * board_area,
                  out + static_cast<size_t>(c0 + 3) * board_area, 1.0f);
    if (c & chess::BLACK_QUEENSIDE)
        std::fill(out + static_cast<size_t>(c0 + 3) * board_area,
                  out + static_cast<size_t>(c0 + 4) * board_area, 1.0f);
    const int halfmove_plane = c0 + 4;
    const float hm           = std::min(1.0f, static_cast<float>(pos.get_halfmove()) / 100.0f);
    std::fill(out + static_cast<size_t>(halfmove_plane) * board_area,
              out + static_cast<size_t>(halfmove_plane + 1) * board_area, hm);
}
void encode_position_with_history(const std::vector<chess::Position> &history, float *out) {
    constexpr int ppp = PLANES_PER_POSITION;
    constexpr int H   = HISTORY_LENGTH;
    const size_t board_area =
        static_cast<size_t>(chess::BOARD_SIZE) * static_cast<size_t>(chess::BOARD_SIZE);
    const size_t total_elems = static_cast<size_t>(INPUT_PLANES) * board_area;
    std::fill(out, out + total_elems, 0.0f);
    const int avail = std::min(H, static_cast<int>(history.size()));
    for (int t = 0; t < avail; ++t) {
        const int base             = t * ppp;
        const chess::Position &pos = history[static_cast<size_t>(history.size() - 1 - t)];
        for (int piece = 0; piece < 6; ++piece) {
            for (int color = 0; color < 2; ++color) {
                const int plane    = base + piece * 2 + color;
                chess::Bitboard bb = pos.get_pieces(static_cast<chess::Piece>(piece),
                                                    static_cast<chess::Color>(color));
                while (bb) {
                    const int s = chess::lsb(bb);
                    chess::pop_lsb(bb);
                    set_plane_value(out, plane, s >> 3, s & 7);
                }
            }
        }
    }
    if (!history.empty()) {
        const chess::Position &cur = history.back();
        const int turn_plane       = H * ppp;
        if (cur.get_turn() == chess::WHITE)
            std::fill(out + static_cast<size_t>(turn_plane) * board_area,
                      out + static_cast<size_t>(turn_plane + 1) * board_area, 1.0f);
        const int fullmove_plane = turn_plane + 1;
        const float fm           = std::min(1.0f, static_cast<float>(cur.get_fullmove()) / 100.0f);
        std::fill(out + static_cast<size_t>(fullmove_plane) * board_area,
                  out + static_cast<size_t>(fullmove_plane + 1) * board_area, fm);
        const int c0    = fullmove_plane + 1;
        const uint8_t c = cur.get_castling();
        if (c & chess::WHITE_KINGSIDE)
            std::fill(out + static_cast<size_t>(c0 + 0) * board_area,
                      out + static_cast<size_t>(c0 + 1) * board_area, 1.0f);
        if (c & chess::WHITE_QUEENSIDE)
            std::fill(out + static_cast<size_t>(c0 + 1) * board_area,
                      out + static_cast<size_t>(c0 + 2) * board_area, 1.0f);
        if (c & chess::BLACK_KINGSIDE)
            std::fill(out + static_cast<size_t>(c0 + 2) * board_area,
                      out + static_cast<size_t>(c0 + 3) * board_area, 1.0f);
        if (c & chess::BLACK_QUEENSIDE)
            std::fill(out + static_cast<size_t>(c0 + 3) * board_area,
                      out + static_cast<size_t>(c0 + 4) * board_area, 1.0f);
        const int halfmove_plane = c0 + 4;
        const float hm           = std::min(1.0f, static_cast<float>(cur.get_halfmove()) / 100.0f);
        std::fill(out + static_cast<size_t>(halfmove_plane) * board_area,
                  out + static_cast<size_t>(halfmove_plane + 1) * board_area, hm);
    }
    fill_repetition_planes(history, out);
}
} // namespace encoder
