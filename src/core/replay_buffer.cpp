#include "replay_buffer.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace replaybuf {

ReplayBuffer::ReplayBuffer(size_t capacity, int planes, int height, int width)
    : capacity_(std::max<size_t>(1, capacity)), planes_(planes), height_(height), width_(width) {
  state_stride_ = static_cast<size_t>(planes_) * static_cast<size_t>(height_) * static_cast<size_t>(width_);
  states_.resize(capacity_ * state_stride_);
  meta_.resize(capacity_);
  rng_.seed(0xC0FFEEu);
}

void ReplayBuffer::seed(uint64_t s) {
  rng_.seed(static_cast<unsigned long>(s));
}

void ReplayBuffer::clear() {
  size_ = 0;
  head_ = 0;
  std::fill(meta_.begin(), meta_.end(), EntryMeta{});
}

void ReplayBuffer::set_capacity(size_t cap) {
  cap = std::max<size_t>(1, cap);
  if (cap == capacity_)
    return;
  const size_t           keep = std::min(size_, cap);
  std::vector<uint8_t>   new_states(cap * state_stride_);
  std::vector<EntryMeta> new_meta(cap);
  size_t                 start = (size_ >= keep) ? ((head_ + capacity_ - keep) % capacity_) : 0;
  std::vector<int32_t>   new_idx;
  std::vector<uint16_t>  new_cnt;
  new_idx.reserve(idx_store_.size());
  new_cnt.reserve(cnt_store_.size());
  for (size_t i = 0; i < keep; ++i) {
    size_t           old_pos = (start + i) % capacity_;
    const EntryMeta& m       = meta_[old_pos];
    std::memcpy(&new_states[i * state_stride_], &states_[old_pos * state_stride_], state_stride_);
    size_t idx_off = new_idx.size();
    size_t cnt_off = new_cnt.size();
    for (size_t k = 0; k < m.idx_len; ++k)
      new_idx.push_back(idx_store_[m.idx_offset + k]);
    for (size_t k = 0; k < m.cnt_len; ++k)
      new_cnt.push_back(cnt_store_[m.cnt_offset + k]);
    new_meta[i] = EntryMeta{idx_off, m.idx_len, cnt_off, m.cnt_len, m.value};
  }
  capacity_ = cap;
  size_     = keep;
  head_     = keep % capacity_;
  states_.swap(new_states);
  meta_.swap(new_meta);
  idx_store_.swap(new_idx);
  cnt_store_.swap(new_cnt);
}

void ReplayBuffer::maybe_compact_ragged() {
  size_t live_idx = 0, live_cnt = 0;
  for (size_t i = 0; i < size_; ++i) {
    size_t pos = (head_ + capacity_ - size_ + i) % capacity_;
    live_idx += meta_[pos].idx_len;
    live_cnt += meta_[pos].cnt_len;
  }
  if (idx_store_.size() <= std::max<size_t>(1, live_idx * 4) && cnt_store_.size() <= std::max<size_t>(1, live_cnt * 4))
    return;
  std::vector<int32_t>  new_idx;
  std::vector<uint16_t> new_cnt;
  new_idx.reserve(live_idx);
  new_cnt.reserve(live_cnt);
  for (size_t i = 0; i < size_; ++i) {
    size_t     pos         = (head_ + capacity_ - size_ + i) % capacity_;
    EntryMeta& m           = meta_[pos];
    size_t     new_idx_off = new_idx.size();
    size_t     new_cnt_off = new_cnt.size();
    for (size_t k = 0; k < m.idx_len; ++k)
      new_idx.push_back(idx_store_[m.idx_offset + k]);
    for (size_t k = 0; k < m.cnt_len; ++k)
      new_cnt.push_back(cnt_store_[m.cnt_offset + k]);
    m.idx_offset = new_idx_off;
    m.cnt_offset = new_cnt_off;
  }
  idx_store_.swap(new_idx);
  cnt_store_.swap(new_cnt);
}

void ReplayBuffer::push(const uint8_t* state, [[maybe_unused]] size_t state_bytes, const int32_t* idx, size_t idx_len,
                        const uint16_t* cnt, size_t cnt_len, int8_t v) {
  assert(state_bytes == state_stride_);
  if (idx_len != cnt_len)
    idx_len = std::min(idx_len, cnt_len);
  std::memcpy(&states_[head_ * state_stride_], state, state_stride_);
  size_t idx_off = idx_store_.size();
  size_t cnt_off = cnt_store_.size();
  idx_store_.insert(idx_store_.end(), idx, idx + idx_len);
  cnt_store_.insert(cnt_store_.end(), cnt, cnt + cnt_len);
  meta_[head_] = EntryMeta{idx_off, idx_len, cnt_off, cnt_len, v};
  head_        = (head_ + 1) % capacity_;
  if (size_ < capacity_)
    ++size_;
  else
    maybe_compact_ragged();
}

void ReplayBuffer::sample(size_t batch_size, double recent_ratio, double recent_window_frac,
                          std::vector<uint8_t>& out_states, std::vector<size_t>& out_offsets,
                          std::vector<int32_t>& out_idx, std::vector<size_t>& out_idx_off,
                          std::vector<uint16_t>& out_cnt, std::vector<size_t>& out_cnt_off,
                          std::vector<int8_t>& out_val) const {
  batch_size = std::min(batch_size, size_);
  if (batch_size == 0)
    return;
  const size_t recent_window =
      std::max<size_t>(1, static_cast<size_t>(static_cast<double>(size_) * recent_window_frac));
  const size_t recent_samples = static_cast<size_t>(std::round(static_cast<double>(batch_size) * recent_ratio));
  const size_t old_samples    = batch_size - recent_samples;

  auto rand_index = [&](size_t lo, size_t hi) {
    std::uniform_int_distribution<size_t> dist(lo, hi - 1);
    return dist(rng_);
  };
  std::vector<size_t> picked;
  picked.reserve(batch_size);
  const size_t base      = (head_ + capacity_ - size_) % capacity_;
  const size_t recent_lo = (size_ >= recent_window) ? (size_ - recent_window) : 0;
  for (size_t i = 0; i < recent_samples; ++i) {
    size_t off = rand_index(recent_lo, size_);
    picked.push_back((base + off) % capacity_);
  }
  const size_t old_hi = (size_ > recent_window) ? (size_ - recent_window) : 1;
  for (size_t i = 0; i < old_samples; ++i) {
    size_t off = rand_index(0, old_hi);
    picked.push_back((base + off) % capacity_);
  }

  out_states.resize(batch_size * state_stride_);
  out_offsets.resize(batch_size + 1);
  out_idx_off.resize(batch_size + 1);
  out_cnt_off.resize(batch_size + 1);
  out_val.resize(batch_size);
  out_offsets[0]   = 0;
  out_idx_off[0]   = 0;
  out_cnt_off[0]   = 0;
  size_t idx_total = 0, cnt_total = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    const EntryMeta& m = meta_[picked[i]];
    std::memcpy(&out_states[i * state_stride_], &states_[picked[i] * state_stride_], state_stride_);
    for (size_t k = 0; k < m.idx_len; ++k)
      out_idx.push_back(idx_store_[m.idx_offset + k]);
    for (size_t k = 0; k < m.cnt_len; ++k)
      out_cnt.push_back(cnt_store_[m.cnt_offset + k]);
    idx_total += m.idx_len;
    cnt_total += m.cnt_len;
    out_val[i]         = m.value;
    out_offsets[i + 1] = (i + 1) * state_stride_;
    out_idx_off[i + 1] = idx_total;
    out_cnt_off[i + 1] = cnt_total;
  }
}

} // namespace replaybuf
