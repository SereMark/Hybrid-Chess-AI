#pragma once

#include "chess.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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
  float ucb(float c_puct, float sqrt_visits) const {
    if (visits == 0) [[unlikely]] {
      return UCB_UNVISITED_BONUS + prior;
    }
    const float q = val_sum / visits;
    const float u = c_puct * prior * sqrt_visits / (1 + visits);
    return q + u;
  }

  [[gnu::always_inline]]
  void update(float value) {
    visits++;
    val_sum += value;
  }
};

class NodePool {
private:
  std::vector<Node> nodes;
  size_t used = 0;
  static constexpr size_t DEFAULT_CAPACITY = 500000;

public:
  NodePool() {
    nodes.reserve(DEFAULT_CAPACITY);
    nodes.resize(DEFAULT_CAPACITY);
  }

  void reset() {
    if (nodes.empty()) {
      nodes.resize(DEFAULT_CAPACITY);
    }
    if (used > 0) {
      std::memset(nodes.data(), 0, used * sizeof(Node));
    }
    used = 1;
  }

  Node *get_root() { return &nodes[0]; }

  Node *allocate(size_t count) {
    if (used + count > nodes.size()) {
      size_t old_size = nodes.size();
      nodes.resize(nodes.size() * 2);
      std::memset(&nodes[old_size], 0,
                  (nodes.size() - old_size) * sizeof(Node));
    }
    Node *ptr = &nodes[used];
    used += count;
    return ptr;
  }

  Node *get_node(uint32_t index) { return &nodes[index]; }

  uint32_t get_index(Node *node) {
    return static_cast<uint32_t>(node - nodes.data());
  }
};

class MCTS {
public:
  MCTS(int simulations = 800, float c_puct = 1.0f, float dirichlet_alpha = 0.3f,
       float dirichlet_weight = 0.25f)
      : simulations_(simulations), c_puct_(c_puct),
        dirichlet_alpha_(dirichlet_alpha), dirichlet_weight_(dirichlet_weight) {
    node_pool_.reset();
  }

  std::vector<int> search(const chess::Position &position,
                          const std::vector<float> &policy, float value);

  void set_simulations(int simulations) { simulations_ = simulations; }

  void set_c_puct(float c_puct) { c_puct_ = c_puct; }

  void set_dirichlet_params(float alpha, float weight) {
    dirichlet_alpha_ = alpha;
    dirichlet_weight_ = weight;
  }

private:
  int simulations_;
  float c_puct_;
  float dirichlet_alpha_;
  float dirichlet_weight_;
  NodePool node_pool_;
  mutable chess::Position working_pos_;
  mutable std::array<chess::Position::MoveInfo, 512> undo_stack_;
  std::array<float, 512> prior_buffer_;
  std::array<uint32_t, 1024> path_buffer_;

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

}
