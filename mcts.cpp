#include "mcts.hpp"
#include <algorithm>
#include <cstring>
#include <random>

namespace mcts {

[[gnu::hot]] std::vector<int> MCTS::search(const chess::Position &position,
                                           const std::vector<float> &policy,
                                           float value) {
  auto root = std::make_unique<Node>();
  auto moves = position.legal_moves();

  if (moves.empty())
    return {};

  std::vector<float> priors;
  priors.reserve(moves.size());
  priors.resize(moves.size());
  float sum = 0;

  for (size_t i = 0; i < moves.size(); i++) {
    int idx = moves[i].from() + moves[i].to() * 64;
    priors[i] = (idx < policy.size()) ? policy[idx] : 1e-8f;
    sum += priors[i];
  }

  if (sum > 0) {
    for (float &p : priors)
      p /= sum;
  } else {
    std::fill(priors.begin(), priors.end(), 1.0f / priors.size());
  }

  root->expand(moves, priors);
  add_dirichlet_noise(root.get(), dirichlet_alpha_, dirichlet_weight_);

  std::vector<Node *> path;
  path.reserve(512);

  for (int sim = 0; sim < simulations_; sim++) {
    Node *node = root.get();
    chess::Position pos = position;
    path.clear();

    while (!node->children.empty()) {
      node = select(node);
      path.push_back(node);
      pos.make_move(node->move);
    }

    float eval_value = value;

    auto result = pos.result();
    if (result != chess::ONGOING) {
      if (result == chess::DRAW) {
        eval_value = 0;
      } else {
        eval_value = (result == chess::WHITE_WIN) ? 1.0f : -1.0f;
        if (pos.turn == chess::BLACK)
          eval_value = -eval_value;
      }
    }

    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      backup(*it, -eval_value);
      eval_value = -eval_value;
    }
    backup(root.get(), eval_value);
  }

  std::vector<int> visits;
  visits.reserve(root->children.size());
  for (const auto &child : root->children) {
    visits.push_back(child->visits);
  }
  return visits;
}

[[gnu::hot]] inline Node *MCTS::select(Node *node) {
  Node *best = node->children[0].get();
  float best_score = best->ucb(c_puct_);

  for (size_t i = 1; i < node->children.size(); i++) {
    const float score = node->children[i]->ucb(c_puct_);
    if (score > best_score) {
      best_score = score;
      best = node->children[i].get();
    }
  }

  return best;
}

void MCTS::backup(Node *node, float value) { node->update(value); }

[[gnu::hot]] void MCTS::add_dirichlet_noise(Node *root, float alpha,
                                            float weight) {
  thread_local static std::mt19937 gen(std::random_device{}());
  std::gamma_distribution<float> gamma(alpha, 1.0f);

  const size_t n_children = root->children.size();
  std::vector<float> noise;
  noise.reserve(n_children);

  float sum = 0;
  for (size_t i = 0; i < n_children; i++) {
    const float n = gamma(gen);
    noise.push_back(n);
    sum += n;
  }

  if (sum > 1e-10f) {
    const float inv_sum = 1.0f / sum;
    const float w = weight;
    const float one_minus_w = 1.0f - w;

    for (size_t i = 0; i < n_children; i++) {
      root->children[i]->prior =
          one_minus_w * root->children[i]->prior + w * noise[i] * inv_sum;
    }
  }
}

}