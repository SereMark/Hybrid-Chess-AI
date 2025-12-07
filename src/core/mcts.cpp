#include "mcts.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
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

}

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
  std::fill(ptr, ptr + count, Node{});
  return ptr;
}

Node* NodePool::get_node(uint32_t index) { return &nodes_[index]; }

uint32_t NodePool::get_index(Node* node) { return static_cast<uint32_t>(node - nodes_.data()); }

int encode_move_index(const chess::Move& move, bool flip) {
  if (!flip) {
    return encode_move_73x64(move);
  }
  const int from = move.from() ^ 56;
  const int to = move.to() ^ 56;
  const chess::Move flipped(from, to, move.promotion());
  return encode_move_73x64(flipped);
}

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

  if (moves.size() > std::numeric_limits<uint16_t>::max()) {
    throw std::overflow_error("túl sok gyerek");
  }

  const uint32_t node_idx = node_pool_.get_index(node);
  const uint32_t first_child = node_pool_.get_index(node_pool_.allocate(moves.size()));

  Node* updated = node_pool_.get_node(node_idx);
  updated->first_child_index = first_child;
  updated->child_count = static_cast<uint16_t>(moves.size());

  Node* children = node_pool_.get_node(first_child);

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
    root_noise_applied_ = false;

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

void MCTS::backpropagate(const std::vector<uint32_t>& path, float value) {
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    Node* path_node = node_pool_.get_node(*it);
    path_node->visits--;
    path_node->val_sum += VIRTUAL_LOSS;
    path_node->update(value);
    value = -value;
  }
  value = -value;

  Node* root_node = node_pool_.get_node(root_index_);
  root_node->visits--;
  root_node->val_sum += VIRTUAL_LOSS;
  root_node->update(value);
}

std::vector<int> MCTS::search_batched_legal(const chess::Position& position,
                                            const EvalLegalBatchFn& eval_fn,
                                            int max_batch) {
  if (max_batch <= 0) {
    throw std::invalid_argument("max_batch > 0 legyen");
  }

  ensure_root(position);
  working_pos_ = position;

  if (!root_noise_applied_) {
    Node* root = node_pool_.get_node(root_index_);
    if (root->child_count > 0) {
      add_dirichlet_noise(root);
      root_noise_applied_ = true;
    }
  }

  std::vector<chess::Position> to_eval;
  std::vector<std::vector<chess::Move>> pending_moves;
  std::vector<int32_t> pending_encoded;
  std::vector<int> pending_counts;
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
    
    eval_fn(to_eval, pending_encoded, pending_counts, eval_priors, eval_values);
    
    if (eval_priors.size() != pending_moves.size() || eval_values.size() != pending_moves.size()) {
      throw std::runtime_error("az értékelési callback visszatérési méretei nem egyeznek");
    }
    for (size_t i = 0; i < to_eval.size(); ++i) {
      const uint32_t node_idx = eval_path_offsets[i];
      Node* node = node_pool_.get_node(node_idx);
      expand_node_with_priors(node, pending_moves[i], eval_priors[i]);

      node = node_pool_.get_node(node_idx);

      if (node_idx == root_index_ && !root_noise_applied_) {
        add_dirichlet_noise(node);
        root_noise_applied_ = true;
      }

      float value = eval_values[i];
      value = -value;
      
      backpropagate(eval_paths[i], value);
    }
    to_eval.clear();
    pending_moves.clear();
    pending_encoded.clear();
    pending_counts.clear();
    eval_path_offsets.clear();
    eval_paths.clear();
  };

  for (int sim = 0; sim < simulations_; ++sim) {
    Node* node = node_pool_.get_node(root_index_);
    node->visits++;
    node->val_sum -= VIRTUAL_LOSS;

    int depth = 0;
    uint32_t current_node_idx = root_index_;

    while (true) {
      node = node_pool_.get_node(current_node_idx);

      if (node->child_count == 0) {
        bool pending = false;
        for (uint32_t idx : eval_path_offsets) {
          if (idx == current_node_idx) {
            pending = true;
            break;
          }
        }

        if (pending) {
          flush_and_expand();
          node = node_pool_.get_node(current_node_idx);
          if (node->child_count > 0) {
            continue;
          }
          break;
        }
        break;
      }

      if (depth >= static_cast<int>(path_buffer_.size())) {
        size_t new_size = path_buffer_.size() * 2;
        path_buffer_.resize(new_size);
        undo_stack_.resize(new_size);
      }
      Node* child = select_child(node);
      
      child->visits++;
      child->val_sum -= VIRTUAL_LOSS;

      const uint32_t child_index = node_pool_.get_index(child);
      working_pos_.make_move_fast(child->move, undo_stack_[depth]);
      path_buffer_[depth] = child_index;
      
      current_node_idx = child_index;
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
      v = -v;

      std::vector<uint32_t> path;
      path.reserve(depth);
      for(int i = 0; i < depth; ++i) {
        path.push_back(path_buffer_[i]);
      }
      backpropagate(path, v);

    } else {
      chess::MoveList legal_moves;
      working_pos_.legal_moves(legal_moves);

      std::vector<chess::Move> moves;
      moves.reserve(legal_moves.size());
      
      const bool flip = (working_pos_.get_turn() == chess::BLACK);
      for (size_t i = 0; i < legal_moves.size(); ++i) {
        moves.push_back(legal_moves[i]);
        const int idx = encode_move_index(legal_moves[i], flip);
        if (idx < 0) {
            pending_encoded.push_back(-1); 
        } else {
            pending_encoded.push_back(static_cast<int32_t>(idx));
        }
      }
      pending_counts.push_back(static_cast<int>(legal_moves.size()));

      eval_path_offsets.push_back(node_pool_.get_index(node));
      to_eval.push_back(working_pos_);
      pending_moves.push_back(std::move(moves));

      std::vector<uint32_t> path;
      path.reserve(static_cast<size_t>(depth));
      for (int d = 0; d < depth; ++d) {
        path.push_back(path_buffer_[d]);
      }
      eval_paths.push_back(std::move(path));

      if (static_cast<int>(to_eval.size()) >= batch_limit) {
        flush_and_expand();
      }
    }

    for (int i = depth - 1; i >= 0; --i) {
      if (i >= static_cast<int>(path_buffer_.size())) {
         break; 
      }
      working_pos_.unmake_move_fast(node_pool_.get_node(path_buffer_[i])->move, undo_stack_[i]);
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
  undo_stack_.resize(1024);
  path_buffer_.resize(1024);
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
  root_noise_applied_ = false;
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
  root_noise_applied_ = false;
}

}
