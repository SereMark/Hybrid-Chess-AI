#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace mcts {
static int encode_move_73x64(const chess::Move &move) {
  const int from = move.from();
  const int to = move.to();
  const int from_row = from >> 3;
  const int from_col = from & 7;
  const int to_row = to >> 3;
  const int to_col = to & 7;
  const int row_diff = to_row - from_row;
  const int col_diff = to_col - from_col;

  const int promo = static_cast<int>(move.promotion());
  if (promo == chess::BISHOP || promo == chess::ROOK || promo == chess::QUEEN) {
    const int promotion_piece = promo - 2;
    int action_plane;
    if (col_diff == 0) {
      action_plane = 64 + promotion_piece * 3 + 0;
    } else if (col_diff == -1) {
      action_plane = 64 + promotion_piece * 3 + 1;
    } else if (col_diff == 1) {
      action_plane = 64 + promotion_piece * 3 + 2;
    } else {
      return -1;
    }
    return action_plane * 64 + from;
  }

  static constexpr int knight_moves[8][2] = {
      {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};
  for (int i = 0; i < 8; i++) {
    if (row_diff == knight_moves[i][0] && col_diff == knight_moves[i][1]) {
      const int action_plane = 56 + i;
      return action_plane * 64 + from;
    }
  }

  static constexpr int directions[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                           {0, 1},   {1, -1}, {1, 0},  {1, 1}};
  if (!(row_diff == 0 && col_diff == 0)) {
    const int distance = std::max(std::abs(row_diff), std::abs(col_diff));
    if (distance >= 1 && distance <= 7) {
      for (int dir_idx = 0; dir_idx < 8; dir_idx++) {
        const int expected_row_diff = directions[dir_idx][0] * distance;
        const int expected_col_diff = directions[dir_idx][1] * distance;
        if (row_diff == expected_row_diff && col_diff == expected_col_diff) {
          const int action_plane = dir_idx * 7 + (distance - 1);
          return action_plane * 64 + from;
        }
      }
    }
  }
  return -1;
}

int encode_move_index(const chess::Move &move) {
  return encode_move_73x64(move);
}

class FastRandom {
private:
  uint64_t state;
  bool has_spare = false;
  float spare = 0.0f;

public:
  FastRandom() : state(std::random_device{}()) {}

  uint32_t next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<uint32_t>(state >> 32);
  }

  float uniform() { return next() * (1.0f / 4294967296.0f); }

  float normal() {
    if (has_spare) {
      has_spare = false;
      return spare;
    }
    has_spare = true;
    const float u = uniform();
    const float v = uniform();
    const float mag = sqrtf(-2.0f * logf(u + DIRICHLET_EPSILON));
    constexpr float TWO_PI = 6.28318530718f;
    spare = mag * cosf(TWO_PI * v);
    return mag * sinf(TWO_PI * v);
  }
};

thread_local FastRandom fast_rng;

[[gnu::hot]]
std::vector<int> MCTS::search(const chess::Position &position,
                              const std::vector<float> &policy, float value) {
  node_pool_.reset();
  Node *root = node_pool_.get_root();
  root->child_idx = 0;
  root->nchildren = 0;
  root->val_sum = 0.0f;
  root->prior = 0.0f;
  root->visits = 0;
  working_pos_ = position;
  chess::MoveList moves;
  working_pos_.legal_moves(moves);
  if (moves.empty()) {
    return {};
  }

  expand_node(root, moves, policy);
  add_dirichlet_noise(root);

  for (int simulation = 0; simulation < simulations_; simulation++) {
    Node *node = root;
    int path_length = 0;

    while (node->nchildren > 0) {
      if (path_length >= static_cast<int>(undo_stack_.size())) {
        break;
      }
      node = select_child(node);
      path_buffer_[path_length] = node_pool_.get_index(node);
      working_pos_.make_move_fast(node->move, undo_stack_[path_length]);
      path_length++;
    }

    float evaluation_value = value;
    auto result = working_pos_.result();
    if (result != chess::ONGOING) {
      if (result == chess::DRAW) {
        evaluation_value = 0;
      } else {
        evaluation_value = (result == chess::WHITE_WIN) ? 1.0f : -1.0f;
        if (working_pos_.turn == chess::BLACK) {
          evaluation_value = -evaluation_value;
        }
      }
    } else {
      chess::MoveList child_moves;
      working_pos_.legal_moves(child_moves);
      if (!child_moves.empty()) {
        std::vector<float> uniform_policy(POLICY_SIZE,
                                          1.0f / child_moves.size());
        expand_node(node, child_moves, uniform_policy);
      }
    }

    for (int i = path_length - 1; i >= 0; i--) {
      node_pool_.get_node(path_buffer_[i])->update(-evaluation_value);
      evaluation_value = -evaluation_value;
    }
    root->update(evaluation_value);

    for (int i = path_length - 1; i >= 0; i--) {
      working_pos_.unmake_move_fast(node_pool_.get_node(path_buffer_[i])->move,
                                    undo_stack_[i]);
    }
  }

  std::vector<int> visits;
  visits.reserve(root->nchildren);
  Node *first_child = node_pool_.get_node(root->child_idx);
  for (uint16_t i = 0; i < root->nchildren; i++) {
    visits.push_back(first_child[i].visits);
  }
  return visits;
}

[[gnu::hot]]
Node *MCTS::select_child(Node *parent) {
  Node *children = node_pool_.get_node(parent->child_idx);
  const float sqrt_parent = sqrtf(static_cast<float>(parent->visits));
  Node *best = &children[0];
  float best_score = best->ucb(c_puct_, sqrt_parent);

  for (uint16_t i = 1; i < parent->nchildren; i++) {
    const float score = children[i].ucb(c_puct_, sqrt_parent);
    if (score > best_score) {
      best_score = score;
      best = &children[i];
    }
  }
  return best;
}

[[gnu::hot]]
void MCTS::expand_node(Node *node, const chess::MoveList &moves,
                       const std::vector<float> &policy) {
  const size_t nchildren = moves.size();
  if (nchildren == 0) {
    return;
  }

  Node *children = node_pool_.allocate(nchildren);
  node->child_idx = node_pool_.get_index(children);
  node->nchildren = static_cast<uint16_t>(nchildren);

  std::vector<float> local_priors(nchildren, 0.0f);
  float sum = 0.0f;
  for (size_t i = 0; i < nchildren; i++) {
    local_priors[i] = get_policy_value(moves[i], policy);
    sum += local_priors[i];
  }

  if (sum > DIRICHLET_EPSILON) {
    const float inverse_sum = 1.0f / sum;
    for (size_t i = 0; i < nchildren; i++) {
      local_priors[i] *= inverse_sum;
    }
  } else {
    const float uniform = 1.0f / nchildren;
    for (size_t i = 0; i < nchildren; i++) {
      local_priors[i] = uniform;
    }
  }

  for (size_t i = 0; i < nchildren; i++) {
    children[i].move = moves[i];
    children[i].prior = local_priors[i];
  }
}

[[gnu::hot]]
void MCTS::add_dirichlet_noise(Node *node) {
  if (node->nchildren == 0) {
    return;
  }

  Node *children = node_pool_.get_node(node->child_idx);
  const size_t nchildren = node->nchildren;
  float noise[256];
  float sum = 0;

  const size_t maxn = std::min(nchildren, sizeof(noise) / sizeof(noise[0]));
  for (size_t i = 0; i < maxn; i++) {
    float g =
        dirichlet_alpha_ + sqrtf(2.0f * dirichlet_alpha_) * fast_rng.normal();
    g = std::max(0.0f, g);
    noise[i] = g;
    sum += g;
  }

  if (sum > DIRICHLET_EPSILON) {
    const float inverse_sum = 1.0f / sum;
    const float weight = dirichlet_weight_;
    const float one_minus_weight = 1.0f - weight;
    for (size_t i = 0; i < nchildren; i++) {
      const float n = (i < maxn) ? noise[i] : (sum / maxn);
      children[i].prior =
          one_minus_weight * children[i].prior + weight * n * inverse_sum;
    }
  }
}

}
