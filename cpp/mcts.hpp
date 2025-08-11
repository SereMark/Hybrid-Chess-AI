#pragma once

#include "chess.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace mcts {

constexpr float UCB_UNVISITED_BONUS = 1e6f;
constexpr float POLICY_EPSILON = 1e-8f;
constexpr float DIRICHLET_EPSILON = 1e-10f;
constexpr int POLICY_SIZE = 73 * 64;

int encode_move_index(const chess::Move &move);

struct alignas(32) Node {
  uint32_t child_idx = 0;
  uint16_t nchildren = 0;
  chess::Move move;
  float val_sum = 0.0f;
  float prior = 0.0f;
  int visits = 0;

  [[gnu::hot, gnu::always_inline]]
  float ucb(float c_puct, float sqrt_visits) const;

  [[gnu::always_inline]]
  void update(float value);
};

class NodePool {
private:
  std::vector<Node> nodes;
  size_t used = 0;
  static constexpr size_t DEFAULT_CAPACITY = 500000;

public:
  NodePool();

  void reset();

  Node *get_root();

  Node *allocate(size_t count);

  Node *get_node(uint32_t index);

  uint32_t get_index(Node *node);
};

class MCTS {
public:
  MCTS(int simulations = 800, float c_puct = 1.0f, float dirichlet_alpha = 0.3f,
       float dirichlet_weight = 0.25f);

  using EvalBatchFn = std::function<void(const std::vector<chess::Position> &,
                                         std::vector<std::vector<float>> &,
                                         std::vector<float> &)>;

  std::vector<int> search_batched(const chess::Position &position,
                                  EvalBatchFn eval_fn, int max_batch = 64);

  void set_simulations(int simulations) { simulations_ = simulations; }

  void set_c_puct(float c_puct) { c_puct_ = c_puct; }

  void set_dirichlet_params(float alpha, float weight) {
    dirichlet_alpha_ = alpha;
    dirichlet_weight_ = weight;
  }

  void set_c_puct_params(float c_puct_base, float c_puct_init);

private:
  int simulations_;
  float c_puct_;
  float c_puct_base_ = 19652.0f;
  float c_puct_init_ = 1.25f;
  float dirichlet_alpha_;
  float dirichlet_weight_;
  NodePool node_pool_;
  mutable chess::Position working_pos_;
  mutable std::array<chess::Position::MoveInfo, 512> undo_stack_;
  std::array<uint32_t, 1024> path_buffer_;
  static constexpr float VIRTUAL_LOSS = 1.0f;

  [[gnu::hot]] Node *select_child(Node *parent);
  [[gnu::hot]] void expand_node(Node *node, const chess::MoveList &moves,
                                const std::vector<float> &policy);
  [[gnu::hot]] void add_dirichlet_noise(Node *node);

  [[gnu::always_inline]]
  static float get_policy_value(const chess::Move &move,
                                const std::vector<float> &policy) {
    const int index = encode_move_index(move);
    return (index >= 0 && index < static_cast<int>(policy.size()))
               ? policy[static_cast<size_t>(index)]
               : POLICY_EPSILON;
  }
};

} // namespace mcts