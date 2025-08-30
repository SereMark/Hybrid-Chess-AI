#pragma once
#include "chess.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

namespace mcts {
// Small numerical epsilons and fixed policy size (73 planes x 64 squares)
constexpr float POLICY_EPSILON = 1e-8f;
constexpr float DIRICHLET_EPSILON = 1e-10f;
constexpr int POLICY_SIZE = 73 * 64;

int encode_move_index(const chess::Move &move);

struct alignas(32) Node {
  // Children are stored in a contiguous pool; this is the index of the first
  // child
  uint32_t first_child_index = 0;
  uint16_t child_count = 0;
  chess::Move move;
  float val_sum = 0.0f;
  float prior = 0.0f;
  int visits = 0;

  [[gnu::hot, gnu::always_inline]] float ucb(float c_puct, float sqrt_visits,
                                             float parent_q, float fpu) const;
  [[gnu::always_inline]] void update(float v);
};

class NodePool {
  std::vector<Node> nodes_;
  size_t used_ = 0;
  static constexpr size_t kDefaultCapacity = 500000;

public:
  NodePool();
  void reset();
  [[nodiscard]] Node *get_root();
  Node *allocate(size_t count);
  [[nodiscard]] Node *get_node(uint32_t index);
  [[nodiscard]] uint32_t get_index(Node *node);
};

class MCTS {
public:
  MCTS(int simulations = 800, float c_puct = 1.0f, float dirichlet_alpha = 0.3f,
       float dirichlet_weight = 0.25f);

  // Evaluator signature: given positions, fill softmax policy logits and scalar
  // value per item
  using EvalBatchFn = std::function<void(const std::vector<chess::Position> &,
                                         std::vector<std::vector<float>> &,
                                         std::vector<float> &)>;

  std::vector<int> search_batched(const chess::Position &position,
                                  EvalBatchFn eval_fn, int max_batch = 64);

  void set_simulations(int s) { simulations_ = s; }
  void set_c_puct(float c) { c_puct_ = c; }
  void set_dirichlet_params(float a, float w) {
    dirichlet_alpha_ = a;
    dirichlet_weight_ = w;
  }
  void set_c_puct_params(float base, float init);
  void set_fpu_reduction(float f) { fpu_reduction_ = f; }
  void seed(uint64_t s) { rng_.seed(static_cast<unsigned long long>(s)); }

private:
  int simulations_;
  float c_puct_;
  float c_puct_base_ = 19652.0f;
  float c_puct_init_ = 1.25f;
  float dirichlet_alpha_;
  float dirichlet_weight_;
  float fpu_reduction_ = 0.10f;

  NodePool node_pool_;
  mutable chess::Position working_pos_;
  mutable std::array<chess::Position::MoveInfo, 1024> undo_stack_;
  std::array<uint32_t, 1024> path_buffer_;
  static constexpr float VIRTUAL_LOSS = 1.0f;
  std::mt19937 rng_;

  [[gnu::hot]] Node *select_child(Node *parent);
  [[gnu::hot]] void expand_node(Node *node, const chess::MoveList &moves,
                                const std::vector<float> &policy);
  [[gnu::hot]] void add_dirichlet_noise(Node *node);
  [[gnu::always_inline]] static float
  get_policy_value(const chess::Move &m, const std::vector<float> &p) {
    const int idx = encode_move_index(m);
    return (idx >= 0 && idx < static_cast<int>(p.size()))
               ? p[static_cast<size_t>(idx)]
               : POLICY_EPSILON;
  }
};
} // namespace mcts
