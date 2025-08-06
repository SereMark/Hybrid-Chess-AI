#pragma once
#include "chess.hpp"
#include <cmath>
#include <memory>
#include <vector>

namespace mcts {

struct alignas(64) Node {
  chess::Move move;
  Node *parent = nullptr;
  std::vector<std::unique_ptr<Node>> children;

  float value_sum = 0.0f;
  float prior = 0.0f;
  int visits = 0;

  [[gnu::hot, gnu::always_inline]] float ucb(float c_puct) const {
    if (visits == 0) [[unlikely]]
      return 1e6f;

    const float q = value_sum / visits;
    const float sqrt_parent = sqrtf(parent ? parent->visits : 1);
    const float u = c_puct * prior * sqrt_parent / (1 + visits);
    return q + u;
  }

  void expand(const std::vector<chess::Move> &moves,
              const std::vector<float> &priors) {
    children.reserve(moves.size());
    for (size_t i = 0; i < moves.size(); i++) {
      auto child = std::make_unique<Node>();
      child->move = moves[i];
      child->parent = this;
      child->prior = priors[i];
      children.push_back(std::move(child));
    }
  }

  [[gnu::always_inline]] void update(float value) {
    visits++;
    value_sum += value;
  }
};

class MCTS {
public:
  MCTS(int simulations = 800, float c_puct = 1.0f, float dirichlet_alpha = 0.3f,
       float dirichlet_weight = 0.25f)
      : simulations_(simulations), c_puct_(c_puct),
        dirichlet_alpha_(dirichlet_alpha), dirichlet_weight_(dirichlet_weight) {
  }

  std::vector<int> search(const chess::Position &position,
                          const std::vector<float> &policy, float value);

  void set_simulations(int sims) { simulations_ = sims; }
  void set_c_puct(float c) { c_puct_ = c; }
  void set_dirichlet_params(float alpha, float weight) {
    dirichlet_alpha_ = alpha;
    dirichlet_weight_ = weight;
  }

private:
  int simulations_;
  float c_puct_;
  float dirichlet_alpha_;
  float dirichlet_weight_;

  Node *select(Node *node);
  void backup(Node *node, float value);
  void add_dirichlet_noise(Node *root, float alpha = 0.3f,
                           float weight = 0.25f);
};

}