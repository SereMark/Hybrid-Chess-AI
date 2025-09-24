#pragma once
#include <cstdint>
#include <random>
#include <vector>

namespace replaybuf {

class ReplayBuffer {
public:
  ReplayBuffer(size_t capacity, int planes, int height, int width);
  void   seed(uint64_t s);
  void   clear();
  void   set_capacity(size_t cap);
  size_t capacity() const {
    return capacity_;
  }
  size_t size() const {
    return size_;
  }

  void push(const uint8_t* state, [[maybe_unused]] size_t state_bytes, const int32_t* idx, size_t idx_len, const uint16_t* cnt,
            size_t cnt_len, int8_t v);

  void sample(size_t batch_size, double recent_ratio, double recent_window_frac, std::vector<uint8_t>& out_states,
              std::vector<size_t>& out_offsets, std::vector<int32_t>& out_idx, std::vector<size_t>& out_idx_off,
              std::vector<uint16_t>& out_cnt, std::vector<size_t>& out_cnt_off, std::vector<int8_t>& out_val) const;

  int planes() const {
    return planes_;
  }
  int height() const {
    return height_;
  }
  int width() const {
    return width_;
  }
  size_t state_stride() const {
    return state_stride_;
  }

private:
  struct EntryMeta {
    size_t idx_offset = 0;
    size_t idx_len    = 0;
    size_t cnt_offset = 0;
    size_t cnt_len    = 0;
    int8_t value      = 0;
  };

  size_t capacity_;
  size_t size_ = 0;
  size_t head_ = 0;

  int    planes_;
  int    height_;
  int    width_;
  size_t state_stride_;

  std::vector<uint8_t>   states_;
  std::vector<EntryMeta> meta_;

  std::vector<int32_t>  idx_store_;
  std::vector<uint16_t> cnt_store_;

  mutable std::mt19937 rng_;

  void maybe_compact_ragged();
};

} // namespace replaybuf
