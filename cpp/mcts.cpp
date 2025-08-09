#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace mcts {

[[gnu::hot, gnu::always_inline]]
inline float Node::ucb(float c_puct, float sqrt_visits) const {
  if (visits == 0) [[unlikely]] {
    return UCB_UNVISITED_BONUS + prior;
  }
  const float q = val_sum / visits;
  const float u = c_puct * prior * sqrt_visits / (1 + visits);
  return q + u;
}

[[gnu::always_inline]]
inline void Node::update(float value) {
  visits++;
  val_sum += value;
}

NodePool::NodePool() {
  nodes.reserve(DEFAULT_CAPACITY);
  nodes.resize(DEFAULT_CAPACITY);
}

void NodePool::reset() {
  if (nodes.empty()) {
    nodes.resize(DEFAULT_CAPACITY);
  }
  used = 1;
  nodes[0] = Node{};
}

Node *NodePool::get_root() {
  return &nodes[0];
}

Node *NodePool::allocate(size_t count) {
  if (used + count > nodes.size()) {
    size_t new_size = nodes.size() * 2;
    if (new_size < used + count)
      new_size = used + count;
    nodes.resize(new_size);
  }
  Node *ptr = &nodes[used];
  used += count;
  for (size_t i = 0; i < count; i++) {
    ptr[i] = Node{};
  }
  return ptr;
}

Node *NodePool::get_node(uint32_t index) {
  return &nodes[index];
}

uint32_t NodePool::get_index(Node *node) {
  return static_cast<uint32_t>(node - nodes.data());
}
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

  static constexpr int knight_moves[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                                             {1, -2},  {1, 2},  {2, -1},  {2, 1}};
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

  float uniform() {
    return next() * (1.0f / 4294967296.0f);
  }

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
Node *MCTS::select_child(Node *parent) {
  Node *children = node_pool_.get_node(parent->child_idx);
  const float sqrt_parent = sqrtf(static_cast<float>(parent->visits));
  float c = c_puct_;
  if (parent->visits > 0) {
    c = logf((parent->visits + c_puct_base_ + 1.0f) / c_puct_base_) + c_puct_init_;
  }
  Node *best = &children[0];
  float best_score = best->ucb(c, sqrt_parent);

  for (uint16_t i = 1; i < parent->nchildren; i++) {
    const float score = children[i].ucb(c, sqrt_parent);
    if (score > best_score) {
      best_score = score;
      best = &children[i];
    }
  }
  return best;
}

[[gnu::hot]]
void MCTS::expand_node(Node *node, const chess::MoveList &moves, const std::vector<float> &policy) {
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

  thread_local std::mt19937 rng(std::random_device{}());
  std::gamma_distribution<float> gamma_dist(dirichlet_alpha_, 1.0f);
  std::vector<float> noise(nchildren);
  float sum = 0.0f;
  for (size_t i = 0; i < nchildren; i++) {
    float g = gamma_dist(rng);
    if (g < 0.0f)
      g = 0.0f;
    noise[i] = g;
    sum += g;
  }

  if (sum > DIRICHLET_EPSILON) {
    const float inv_sum = 1.0f / sum;
    const float w = dirichlet_weight_;
    const float one_minus_w = 1.0f - w;
    for (size_t i = 0; i < nchildren; i++) {
      const float n = noise[i] * inv_sum;
      children[i].prior = one_minus_w * children[i].prior + w * n;
    }
  }
}

std::vector<int> MCTS::search_batched(const chess::Position &position, EvalBatchFn eval_fn,
                                      int max_batch) {
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

  std::vector<chess::Position> to_eval;
  std::vector<uint32_t> eval_path_offsets;
  std::vector<std::vector<uint32_t>> eval_paths;
  std::vector<uint32_t> pending_nodes;
  to_eval.reserve(static_cast<size_t>(max_batch));
  eval_path_offsets.reserve(static_cast<size_t>(max_batch));
  eval_paths.reserve(static_cast<size_t>(max_batch));
  pending_nodes.reserve(static_cast<size_t>(max_batch));

  std::vector<float> root_policy(POLICY_SIZE, 0.0f);
  std::vector<float> tmp_policy;
  std::vector<float> tmp_value;
  tmp_policy.reserve(static_cast<size_t>(POLICY_SIZE));

  auto flush_and_expand = [&]() {
    if (to_eval.empty())
      return;
    for (uint32_t idx : pending_nodes) {
      Node *n = node_pool_.get_node(idx);
      if (n->visits > 0)
        n->visits -= 1;
    }
    pending_nodes.clear();
    std::vector<std::vector<float>> policies(to_eval.size());
    std::vector<float> values(to_eval.size(), 0.0f);
    eval_fn(to_eval, policies, values);
    for (size_t i = 0; i < to_eval.size(); i++) {
      const uint32_t node_index = eval_path_offsets[i];
      Node *node = node_pool_.get_node(node_index);
      chess::MoveList child_moves;
      {
        working_pos_ = to_eval[i];
        working_pos_.legal_moves(child_moves);
      }
      if (!child_moves.empty()) {
        expand_node(node, child_moves, policies[i]);
      }
      float v = values[i];
      node->val_sum += VIRTUAL_LOSS;
      const auto &path = eval_paths[i];
      for (auto it = path.rbegin(); it != path.rend(); ++it) {
        node_pool_.get_node(*it)->update(-v);
        v = -v;
      }
      root->update(v);
    }
    to_eval.clear();
    eval_path_offsets.clear();
    eval_paths.clear();
  };

  {
    std::vector<chess::Position> root_vec{position};
    std::vector<std::vector<float>> policies(1);
    std::vector<float> values(1, 0.0f);
    eval_fn(root_vec, policies, values);
    root_policy = policies[0];
    expand_node(root, moves, root_policy);
    add_dirichlet_noise(root);
  }

  for (int simulation = 0; simulation < simulations_; simulation++) {
    Node *node = root;
    int path_length = 0;

    working_pos_ = position;
    while (node->nchildren > 0) {
      if (path_length >= static_cast<int>(undo_stack_.size())) {
        break;
      }
      node = select_child(node);
      path_buffer_[path_length] = node_pool_.get_index(node);
      working_pos_.make_move_fast(node->move, undo_stack_[path_length]);
      path_length++;
    }

    auto result = working_pos_.result();
    if (result != chess::ONGOING) {
      float evaluation_value;
      if (result == chess::DRAW) {
        evaluation_value = 0.0f;
      } else {
        evaluation_value = (result == chess::WHITE_WIN) ? 1.0f : -1.0f;
        if (working_pos_.turn == chess::BLACK) {
          evaluation_value = -evaluation_value;
        }
      }
      for (int i = path_length - 1; i >= 0; i--) {
        node_pool_.get_node(path_buffer_[i])->update(-evaluation_value);
        evaluation_value = -evaluation_value;
      }
      root->update(evaluation_value);
    } else {
      const uint32_t node_index = node_pool_.get_index(node);
      eval_path_offsets.push_back(node_index);
      to_eval.push_back(working_pos_);
      std::vector<uint32_t> path;
      path.reserve(static_cast<size_t>(path_length));
      for (int d = 0; d < path_length; d++) {
        path.push_back(path_buffer_[d]);
      }
      eval_paths.push_back(std::move(path));
      Node *pending = node_pool_.get_node(node_index);
      pending->visits += 1;
      pending->val_sum -= VIRTUAL_LOSS;
      pending_nodes.push_back(node_index);
      if (static_cast<int>(to_eval.size()) >= max_batch) {
        flush_and_expand();
      }
    }

    for (int i = path_length - 1; i >= 0; i--) {
      working_pos_.unmake_move_fast(node_pool_.get_node(path_buffer_[i])->move, undo_stack_[i]);
    }
  }

  flush_and_expand();

  std::vector<int> visits;
  visits.reserve(root->nchildren);
  Node *first_child = node_pool_.get_node(root->child_idx);
  for (uint16_t i = 0; i < root->nchildren; i++) {
    visits.push_back(first_child[i].visits);
  }
  return visits;
}

MCTS::MCTS(int simulations, float c_puct, float dirichlet_alpha, float dirichlet_weight)
    : simulations_(simulations), c_puct_(c_puct), dirichlet_alpha_(dirichlet_alpha),
      dirichlet_weight_(dirichlet_weight) {
  node_pool_.reset();
}

void MCTS::set_c_puct_params(float c_puct_base, float c_puct_init) {
  c_puct_base_ = c_puct_base;
  c_puct_init_ = c_puct_init;
}

} // namespace mcts