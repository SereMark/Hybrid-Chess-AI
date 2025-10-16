#include "mcts.hpp"

// Hybrid Chess AI - Monte Carlo Tree Search implementation

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace mcts {

namespace {

int encode_move_73x64(const chess::Move& move) {
  const int from = move.from();
  const int to = move.to();
  const int fr = from >> 3;
  const int fc = from & 7;
  const int tr = to >> 3;
  const int tc = to & 7;
  const int dr = tr - fr;
  const int dc = tc - fc;
  const int promo = static_cast<int>(move.promotion());

  if (promo == chess::KNIGHT || promo == chess::ROOK || promo == chess::BISHOP) {
    const int promo_plane = (promo == chess::KNIGHT) ? 0 : ((promo == chess::ROOK) ? 1 : 2);
    int plane_index;
    if (dc == 0) {
      plane_index = 64 + promo_plane * 3 + 0;
    } else if (dc == -1) {
      plane_index = 64 + promo_plane * 3 + 1;
    } else if (dc == 1) {
      plane_index = 64 + promo_plane * 3 + 2;
    } else {
      return -1;
    }
    return plane_index * 64 + from;
  }

  static constexpr int kKnightOffsets[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                                               {1, -2},  {1, 2},  {2, -1},  {2, 1}};
  for (int i = 0; i < 8; ++i) {
    if (dr == kKnightOffsets[i][0] && dc == kKnightOffsets[i][1]) {
      return (56 + i) * 64 + from;
    }
  }

  static constexpr int kDir8[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                      {0, 1},  {1, -1}, {1, 0},  {1, 1}};
  if (dr != 0 || dc != 0) {
    const int dist = std::max(std::abs(dr), std::abs(dc));
    if (dist >= 1 && dist <= 7) {
      for (int d = 0; d < 8; ++d) {
        if (dr == kDir8[d][0] * dist && dc == kDir8[d][1] * dist) {
          return (d * 7 + (dist - 1)) * 64 + from;
        }
      }
    }
  }
  return -1;
}

void normalize_priors_inplace(std::vector<float>& priors) {
  float sum = 0.0f;
  bool any_valid = false;
  for (float& value : priors) {
    if (std::isfinite(value) && value > 0.0f) {
      sum += value;
      any_valid = true;
    } else {
      value = 0.0f;
    }
  }

  const float inv = (any_valid && sum > POLICY_EPSILON) ? (1.0f / sum) : 0.0f;
  if (inv == 0.0f) {
    const float uniform = 1.0f / static_cast<float>(priors.empty() ? 1 : priors.size());
    for (float& value : priors) {
      value = uniform;
    }
  } else {
    for (float& value : priors) {
      value *= inv;
    }
  }
}

}  // namespace

float Node::ucb(float c_puct, float sqrt_visits, float parent_q, float fpu) const {
  const float q = (visits > 0) ? (val_sum / visits) : (parent_q - fpu);
  const float u = c_puct * prior * sqrt_visits / (1.0f + visits);
  return q + u;
}

void Node::update(float v) {
  ++visits;
  val_sum += v;
}

NodePool::NodePool() {
  nodes_.reserve(kDefaultCapacity);
  nodes_.resize(kDefaultCapacity);
}

void NodePool::reset() {
  if (nodes_.empty()) {
    nodes_.resize(kDefaultCapacity);
  }
  used_ = 1;
  nodes_[0] = Node{};
}

Node* NodePool::get_root() { return &nodes_[0]; }

Node* NodePool::allocate(size_t count) {
  if (used_ + count > nodes_.size()) {
    size_t new_size = nodes_.size() * 2;
    if (new_size < used_ + count) {
      new_size = used_ + count;
    }
    nodes_.resize(new_size);
  }
  Node* ptr = &nodes_[used_];
  used_ += count;
  std::memset(static_cast<void*>(ptr), 0, sizeof(Node) * count);
  return ptr;
}

Node* NodePool::get_node(uint32_t index) { return &nodes_[index]; }

uint32_t NodePool::get_index(Node* node) { return static_cast<uint32_t>(node - nodes_.data()); }

int encode_move_index(const chess::Move& move) { return encode_move_73x64(move); }

Node* MCTS::select_child(Node* parent) {
  Node* children = node_pool_.get_node(parent->first_child_index);
  const float sqrt_visits = std::sqrt(1.0f + static_cast<float>(parent->visits));

  float adjusted_c = c_puct_;
  if (parent->visits > 0) {
    adjusted_c = c_puct_ * (std::log((parent->visits + c_puct_base_ + 1.0f) / c_puct_base_) + c_puct_init_);
  }
  const float parent_q = (parent->visits > 0) ? (parent->val_sum / parent->visits) : 0.0f;

  Node* best_child = &children[0];
  float best_score = best_child->ucb(adjusted_c, sqrt_visits, parent_q, fpu_reduction_);
  for (uint16_t i = 1; i < parent->child_count; ++i) {
    const float score = children[i].ucb(adjusted_c, sqrt_visits, parent_q, fpu_reduction_);
    if (score > best_score) {
      best_score = score;
      best_child = &children[i];
    }
  }
  return best_child;
}

void MCTS::expand_node_with_priors(Node* node,
                                   const std::vector<chess::Move>& moves,
                                   const std::vector<float>& priors_in) {
  if (moves.empty()) {
    return;
  }

  std::vector<float> priors = priors_in;
  priors.resize(moves.size(), 0.0f);
  normalize_priors_inplace(priors);

  node->first_child_index = node_pool_.get_index(node_pool_.allocate(moves.size()));
  node->child_count = static_cast<uint16_t>(moves.size());
  Node* children = node_pool_.get_node(node->first_child_index);

  for (size_t i = 0; i < moves.size(); ++i) {
    children[i] = Node{};
    children[i].move = moves[i];
    children[i].prior = priors[i];
  }
}

void MCTS::add_dirichlet_noise(Node* node) {
  if (node->child_count == 0 || dirichlet_weight_ <= 0.0f) {
    return;
  }

  std::gamma_distribution<float> gamma(dirichlet_alpha_, 1.0f);
  std::vector<float> noise(node->child_count, 0.0f);
  float sum = 0.0f;
  for (float& n : noise) {
    n = gamma(rng_);
    sum += n;
  }
  if (sum <= 0.0f) {
    return;
  }
  for (float& n : noise) {
    n /= sum;
  }

  Node* children = node_pool_.get_node(node->first_child_index);
  for (uint16_t i = 0; i < node->child_count; ++i) {
    children[i].prior = children[i].prior * (1.0f - dirichlet_weight_) + noise[i] * dirichlet_weight_;
  }
}

bool MCTS::ensure_root(const chess::Position& position) {
  const uint64_t hash = position.get_hash();
  if (!root_initialized_ || hash != root_hash_) {
    node_pool_.reset();
    root_index_ = 0;
    root_hash_ = hash;
    root_initialized_ = true;

    Node* root = node_pool_.get_node(root_index_);
    root->first_child_index = 0;
    root->child_count = 0;
    root->val_sum = 0.0f;
    root->prior = 0.0f;
    root->visits = 0;
    return false;
  }
  return true;
}

std::vector<int> MCTS::search_batched_legal(const chess::Position& position,
                                            const EvalLegalBatchFn& eval_fn,
                                            int max_batch) {
  if (max_batch <= 0) {
    throw std::invalid_argument("max_batch must be > 0");
  }

  ensure_root(position);
  working_pos_ = position;

  std::vector<chess::Position> to_eval;
  std::vector<std::vector<chess::Move>> pending_moves;
  std::vector<std::vector<float>> eval_priors;
  std::vector<float> eval_values;
  std::vector<uint32_t> eval_path_offsets;
  std::vector<std::vector<uint32_t>> eval_paths;

  const int batch_limit = std::max(1, std::min(max_batch, std::max(1, simulations_)));

  auto flush_and_expand = [&]() {
    if (to_eval.empty()) {
      return;
    }
    eval_priors.clear();
    eval_values.clear();
    eval_fn(to_eval, pending_moves, eval_priors, eval_values);
    if (eval_priors.size() != pending_moves.size() || eval_values.size() != pending_moves.size()) {
      throw std::runtime_error("evaluation callback returned mismatched sizes");
    }
    for (size_t i = 0; i < to_eval.size(); ++i) {
      Node* node = node_pool_.get_node(eval_path_offsets[i]);
      node->val_sum += eval_values[i];
      expand_node_with_priors(node, pending_moves[i], eval_priors[i]);
      add_dirichlet_noise(node);

      float value = eval_values[i];
      for (auto it = eval_paths[i].rbegin(); it != eval_paths[i].rend(); ++it) {
        node_pool_.get_node(*it)->update(value);
        value = -value;
      }
      node_pool_.get_node(root_index_)->update(value);
    }
    to_eval.clear();
    pending_moves.clear();
    eval_path_offsets.clear();
    eval_paths.clear();
  };

  for (int sim = 0; sim < simulations_; ++sim) {
    Node* node = node_pool_.get_node(root_index_);
    int depth = 0;
    while (node->child_count > 0) {
      Node* child = select_child(node);
      const uint32_t child_index = node_pool_.get_index(child);
      working_pos_.make_move_fast(child->move, undo_stack_[depth]);
      path_buffer_[depth] = child_index;
      child->val_sum -= VIRTUAL_LOSS;
      node = child;
      ++depth;
    }

    const chess::Result terminal = working_pos_.result();
    if (terminal != chess::ONGOING) {
      float v;
      if (terminal == chess::DRAW) {
        v = 0.0f;
      } else {
        v = (terminal == chess::WHITE_WIN) ? 1.0f : -1.0f;
        if (working_pos_.get_turn() == chess::BLACK) {
          v = -v;
        }
      }
      for (int i = depth - 1; i >= 0; --i) {
        node_pool_.get_node(path_buffer_[i])->update(v);
        v = -v;
      }
      node_pool_.get_node(root_index_)->update(v);
    } else {
      chess::MoveList legal_moves;
      working_pos_.legal_moves(legal_moves);

      std::vector<chess::Move> moves;
      moves.reserve(legal_moves.size());
      for (size_t i = 0; i < legal_moves.size(); ++i) {
        moves.push_back(legal_moves[i]);
      }

      eval_path_offsets.push_back(node_pool_.get_index(node));
      to_eval.push_back(working_pos_);
      pending_moves.push_back(std::move(moves));

      std::vector<uint32_t> path;
      path.reserve(static_cast<size_t>(depth));
      for (int d = 0; d < depth; ++d) {
        path.push_back(path_buffer_[d]);
      }
      eval_paths.push_back(std::move(path));

      node->val_sum -= VIRTUAL_LOSS;
      if (static_cast<int>(to_eval.size()) >= batch_limit) {
        flush_and_expand();
      }
    }

    for (int i = depth - 1; i >= 0; --i) {
      working_pos_.unmake_move_fast(node_pool_.get_node(path_buffer_[i])->move, undo_stack_[i]);
    }

    if (!to_eval.empty()) {
      flush_and_expand();
    }
  }

  flush_and_expand();

  std::vector<int> visits;
  Node* root = node_pool_.get_node(root_index_);
  visits.reserve(root->child_count);
  Node* children = node_pool_.get_node(root->first_child_index);
  for (uint16_t i = 0; i < root->child_count; ++i) {
    visits.push_back(children[i].visits);
  }
  return visits;
}

MCTS::MCTS(int simulations, float c_puct, float alpha, float weight)
    : simulations_(simulations),
      c_puct_(c_puct),
      c_puct_base_(19652.0f),
      c_puct_init_(1.25f),
      dirichlet_alpha_(alpha),
      dirichlet_weight_(weight) {
  node_pool_.reset();
  rng_.seed(1337u);
  root_initialized_ = false;
}

void MCTS::set_c_puct_params(float base, float init) {
  c_puct_base_ = base;
  c_puct_init_ = init;
}


void MCTS::reset_tree() {
  node_pool_.reset();
  root_index_ = 0;
  root_hash_ = 0;
  root_initialized_ = false;
}

void MCTS::advance_root(const chess::Position& new_position, const chess::Move& played_move) {
  const uint64_t new_hash = new_position.get_hash();
  if (!root_initialized_) {
    node_pool_.reset();
    root_index_ = 0;
    root_hash_ = new_hash;
    root_initialized_ = true;
    return;
  }

  Node* root = node_pool_.get_node(root_index_);
  bool matched_child = false;
  if (root->child_count > 0) {
    Node* children = node_pool_.get_node(root->first_child_index);
    for (uint16_t i = 0; i < root->child_count; ++i) {
      if (children[i].move == played_move) {
        root_index_ = static_cast<uint32_t>(root->first_child_index + i);
        matched_child = true;
        break;
      }
    }
  }
  if (!matched_child) {
    node_pool_.reset();
    root_index_ = 0;
  }
  root_hash_ = new_hash;
  root_initialized_ = true;
}

}  // namespace mcts
