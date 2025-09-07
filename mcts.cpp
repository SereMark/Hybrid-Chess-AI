#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace mcts {

  [[gnu::hot, gnu::always_inline]] inline float
  Node::ucb(float c_puct, float sqrt_visits, float parent_q, float fpu) const {
    const float q = (visits > 0) ? (val_sum / visits) : (parent_q - fpu);
    const float u = c_puct * prior * sqrt_visits / (1.0f + visits);
    return q + u;
  }

  [[gnu::always_inline]] inline void Node::update(float v) {
    ++visits;
    val_sum += v;
  }

  NodePool::NodePool() {
    nodes_.reserve(kDefaultCapacity);
    nodes_.resize(kDefaultCapacity);
  }

  void NodePool::reset() {
    if (nodes_.empty())
      nodes_.resize(kDefaultCapacity);
    used_     = 1;
    nodes_[0] = Node{};
  }

  Node* NodePool::get_root() {
    return &nodes_[0];
  }

  Node* NodePool::allocate(size_t count) {
    if (used_ + count > nodes_.size()) {
      size_t new_size = nodes_.size() * 2;
      if (new_size < used_ + count)
        new_size = used_ + count;
      nodes_.resize(new_size);
    }
    Node* ptr = &nodes_[used_];
    used_ += count;
    for (size_t i = 0; i < count; ++i)
      ptr[i] = Node{};
    return ptr;
  }

  Node* NodePool::get_node(uint32_t i) {
    return &nodes_[i];
  }

  uint32_t NodePool::get_index(Node* n) {
    return static_cast<uint32_t>(n - nodes_.data());
  }

  static int encode_move_73x64(const chess::Move& move) {
    const int from = move.from(), to = move.to();
    const int fr = from >> 3, fc = from & 7, tr = to >> 3, tc = to & 7, dr = tr - fr, dc = tc - fc;
    const int promo = static_cast<int>(move.promotion());
    if (promo == chess::KNIGHT || promo == chess::ROOK || promo == chess::BISHOP) {
      int pp = (promo == chess::KNIGHT) ? 0 : ((promo == chess::ROOK) ? 1 : 2);
      int plane;
      if (dc == 0)
        plane = 64 + pp * 3 + 0;
      else if (dc == -1)
        plane = 64 + pp * 3 + 1;
      else if (dc == 1)
        plane = 64 + pp * 3 + 2;
      else
        return -1;
      return plane * 64 + from;
    }
    static constexpr int KNIGHT_OFFSETS[8][2] =
        {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};
    for (int i = 0; i < 8; ++i)
      if (dr == KNIGHT_OFFSETS[i][0] && dc == KNIGHT_OFFSETS[i][1])
        return (56 + i) * 64 + from;
    static constexpr int DIR8[8][2] =
        {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
    if (dr || dc) {
      const int dist = std::max(std::abs(dr), std::abs(dc));
      if (dist >= 1 && dist <= 7) {
        for (int d = 0; d < 8; ++d)
          if (dr == DIR8[d][0] * dist && dc == DIR8[d][1] * dist)
            return (d * 7 + (dist - 1)) * 64 + from;
      }
    }
    return -1;
  }

  int encode_move_index(const chess::Move& m) {
    return encode_move_73x64(m);
  }

  [[gnu::hot]] Node* MCTS::select_child(Node* parent) {
    Node*       children    = node_pool_.get_node(parent->first_child_index);
    const float sqrt_visits = sqrtf(1.0f + static_cast<float>(parent->visits));
    float       c           = c_puct_;
    if (parent->visits > 0)
      c = c_puct_ * (logf((parent->visits + c_puct_base_ + 1.0f) / c_puct_base_) + c_puct_init_);
    const float parent_q = (parent->visits > 0) ? (parent->val_sum / parent->visits) : 0.0f;

    Node* best       = &children[0];
    float best_score = best->ucb(c, sqrt_visits, parent_q, fpu_reduction_);
    for (uint16_t i = 1; i < parent->child_count; ++i) {
      float sc = children[i].ucb(c, sqrt_visits, parent_q, fpu_reduction_);
      if (sc > best_score) {
        best_score = sc;
        best       = &children[i];
      }
    }
    return best;
  }

  [[gnu::hot]] void
  MCTS::expand_node(Node* node, const chess::MoveList& moves, const std::vector<float>& policy) {
    const size_t n = moves.size();
    if (!n)
      return;
    Node* children          = node_pool_.allocate(n);
    node->first_child_index = node_pool_.get_index(children);
    node->child_count       = static_cast<uint16_t>(n);

    std::vector<float> pri(n, 0.0f); // normalized child priors
    float              sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      pri[i] = get_policy_value(moves[i], policy);
      sum += pri[i];
    }
    if (sum > DIRICHLET_EPSILON) {
      const float inv = 1.0f / sum;
      for (size_t i = 0; i < n; ++i)
        pri[i] *= inv;
    } else {
      const float u = 1.0f / static_cast<float>(n);
      for (size_t i = 0; i < n; ++i)
        pri[i] = u;
    }
    for (size_t i = 0; i < n; ++i) {
      children[i].move  = moves[i];
      children[i].prior = pri[i];
    }
  }

  [[gnu::hot]] void MCTS::add_dirichlet_noise(Node* node) {
    if (node->child_count == 0)
      return;
    Node*        children = node_pool_.get_node(node->first_child_index);
    const size_t n        = node->child_count;

    std::gamma_distribution<float> g(dirichlet_alpha_, 1.0f);
    std::vector<float>             noise(n);
    float                          sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      float x = g(rng_);
      if (x < 0.0f)
        x = 0.0f;
      noise[i] = x;
      sum += x;
    }
    if (sum > DIRICHLET_EPSILON) {
      const float inv = 1.0f / sum, w = dirichlet_weight_, omw = 1.0f - w;
      for (size_t i = 0; i < n; ++i) {
        const float nrm   = noise[i] * inv;
        children[i].prior = omw * children[i].prior + w * nrm;
      }
    }
  }

  std::vector<int>
  MCTS::search_batched(const chess::Position& position, EvalBatchFn eval_fn, int max_batch) {
    if (max_batch < 1)
      max_batch = 1;
    node_pool_.reset();
    {
      Node* root              = node_pool_.get_root();
      root->first_child_index = 0;
      root->child_count       = 0;
      root->val_sum           = 0.0f;
      root->prior             = 0.0f;
      root->visits            = 0;
    }

    working_pos_ = position;

    chess::MoveList root_moves;
    working_pos_.legal_moves(root_moves);
    if (root_moves.empty())
      return {};

    std::vector<chess::Position>       to_eval;
    std::vector<uint32_t>              eval_path_offsets;
    std::vector<std::vector<uint32_t>> eval_paths;
    std::vector<uint32_t>              pending_nodes;
    to_eval.reserve(static_cast<size_t>(max_batch));
    eval_path_offsets.reserve(static_cast<size_t>(max_batch));
    eval_paths.reserve(static_cast<size_t>(max_batch));
    pending_nodes.reserve(static_cast<size_t>(max_batch));

    std::vector<float> root_policy(POLICY_SIZE, 0.0f);

    auto flush_and_expand = [&]() {
      if (to_eval.empty())
        return;
      for (uint32_t idx : pending_nodes) {
        Node* node_ptr = node_pool_.get_node(idx);
        if (node_ptr->visits > 0)
          node_ptr->visits -= 1;
      }
      pending_nodes.clear();

      std::vector<std::vector<float>> policy_batch(to_eval.size());
      std::vector<float>              value_batch(to_eval.size(), 0.0f);
      eval_fn(to_eval, policy_batch, value_batch);

      if (policy_batch.size() != to_eval.size() || value_batch.size() != to_eval.size()) {
        throw std::runtime_error("MCTS evaluator returned mismatched batch size");
      }
      for (size_t i = 0; i < to_eval.size(); ++i) {
        const uint32_t idx  = eval_path_offsets[i];
        Node*          node = node_pool_.get_node(idx);

        chess::MoveList child_moves;
        {
          working_pos_ = to_eval[i];
          working_pos_.legal_moves(child_moves);
        }
        if (!child_moves.empty())
          expand_node(node, child_moves, policy_batch[i]);

        node->val_sum += VIRTUAL_LOSS;
        float       v    = value_batch[i];
        const auto& path = eval_paths[i];
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
          node_pool_.get_node(*it)->update(-v);
          v = -v;
        }
        node_pool_.get_node(0)->update(v);
      }

      to_eval.clear();
      eval_path_offsets.clear();
      eval_paths.clear();
    };

    {
      std::vector<chess::Position>    root_vec{position};
      std::vector<std::vector<float>> policy_batch(1);
      std::vector<float>              value_batch(1, 0.0f);
      eval_fn(root_vec, policy_batch, value_batch);
      if (policy_batch.size() != 1 || value_batch.size() != 1) {
        throw std::runtime_error("MCTS root evaluator returned wrong batch size");
      }
      Node* root  = node_pool_.get_root();
      root_policy = policy_batch[0];
      expand_node(root, root_moves, root_policy);
      root = node_pool_.get_root();
      add_dirichlet_noise(root);
    }

    for (int sim = 0; sim < simulations_; ++sim) {
      Node* node          = node_pool_.get_node(0);
      int   depth         = 0;
      working_pos_        = position;
      const int max_depth = static_cast<int>(std::min(undo_stack_.size(), path_buffer_.size()));

      while (node->child_count > 0) {
        if (depth >= max_depth)
          break;
        node                = select_child(node);
        path_buffer_[depth] = node_pool_.get_index(node);
        working_pos_.make_move_fast(node->move, undo_stack_[depth]);
        ++depth;
      }

      auto res = working_pos_.result();
      if (res != chess::ONGOING) {
        float v;
        if (res == chess::DRAW)
          v = 0.0f;
        else {
          v = (res == chess::WHITE_WIN) ? 1.0f : -1.0f;
          if (working_pos_.turn == chess::BLACK)
            v = -v;
        }
        for (int i = depth - 1; i >= 0; --i) {
          node_pool_.get_node(path_buffer_[i])->update(-v);
          v = -v;
        }
        node_pool_.get_node(0)->update(v);
      } else {
        const uint32_t idx = node_pool_.get_index(node);
        eval_path_offsets.push_back(idx);
        to_eval.push_back(working_pos_);
        std::vector<uint32_t> path;
        path.reserve(static_cast<size_t>(depth));
        for (int d = 0; d < depth; ++d)
          path.push_back(path_buffer_[d]);
        eval_paths.push_back(std::move(path));
        Node* pend = node_pool_.get_node(idx);
        pend->visits += 1;
        pend->val_sum -= VIRTUAL_LOSS;
        pending_nodes.push_back(idx);
        if (static_cast<int>(to_eval.size()) >= max_batch)
          flush_and_expand();
      }

      for (int i = depth - 1; i >= 0; --i)
        working_pos_.unmake_move_fast(node_pool_.get_node(path_buffer_[i])->move, undo_stack_[i]);
    }

    flush_and_expand();

    std::vector<int> visits;
    {
      Node* root = node_pool_.get_node(0);
      visits.reserve(root->child_count);
      Node* children = node_pool_.get_node(root->first_child_index);
      for (uint16_t i = 0; i < root->child_count; ++i)
        visits.push_back(children[i].visits);
    }
    return visits;
  }

  MCTS::MCTS(int sims, float c, float alpha, float w)
      : simulations_(sims),
        c_puct_(c),
        c_puct_base_(19652.0f),
        c_puct_init_(1.25f),
        dirichlet_alpha_(alpha),
        dirichlet_weight_(w) {
    node_pool_.reset();
    rng_.seed(1337u);
  }

  void MCTS::set_c_puct_params(float base, float init) {
    c_puct_base_ = base;
    c_puct_init_ = init;
  }

}
