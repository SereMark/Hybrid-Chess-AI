#pragma once
#include "chess.hpp"
namespace encoder {
constexpr int INPUT_PLANES = 119;
constexpr int HISTORY_LENGTH = 8;
constexpr int PLANES_PER_POSITION = 14;
void encode_position_into(const chess::Position &pos, float *out);
void encode_position_with_history(const std::vector<chess::Position> &history,
                                  float *out);
} // namespace encoder
