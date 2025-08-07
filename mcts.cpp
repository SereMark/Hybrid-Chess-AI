#include "mcts.hpp"

#include <algorithm>
#include <cstring>
#include <random>

namespace mcts {

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
std::vector<int> MCTS::search(const chess::Position &position,
                              const std::vector<float> &policy,
                              float value) {
    node_pool_.reset();
    Node *root = node_pool_.get_root();
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
                std::vector<float> uniform_policy(4096, 1.0f / child_moves.size());
                expand_node(node, child_moves, uniform_policy);
            }
        }

        for (int i = path_length - 1; i >= 0; i--) {
            node_pool_.get_node(path_buffer_[i])->update(-evaluation_value);
            evaluation_value = -evaluation_value;
        }
        root->update(evaluation_value);

        for (int i = path_length - 1; i >= 0; i--) {
            working_pos_.unmake_move_fast(
                node_pool_.get_node(path_buffer_[i])->move,
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
void MCTS::expand_node(Node *node,
                       const chess::MoveList &moves,
                       const std::vector<float> &policy) {
    const size_t nchildren = moves.size();
    if (nchildren == 0) {
        return;
    }

    Node *children = node_pool_.allocate(nchildren);
    node->child_idx = node_pool_.get_index(children);
    node->nchildren = static_cast<uint16_t>(nchildren);

    float sum = 0;
    for (size_t i = 0; i < nchildren; i++) {
        prior_buffer_[i] = get_policy_value(moves[i], policy);
        sum += prior_buffer_[i];
    }

    if (sum > DIRICHLET_EPSILON) {
        const float inverse_sum = 1.0f / sum;
        for (size_t i = 0; i < nchildren; i++) {
            prior_buffer_[i] *= inverse_sum;
        }
    } else {
        const float uniform = 1.0f / nchildren;
        for (size_t i = 0; i < nchildren; i++) {
            prior_buffer_[i] = uniform;
        }
    }

    for (size_t i = 0; i < nchildren; i++) {
        children[i].move = moves[i];
        children[i].prior = prior_buffer_[i];
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

    for (size_t i = 0; i < nchildren; i++) {
        float g = dirichlet_alpha_ +
                  sqrtf(2.0f * dirichlet_alpha_) * fast_rng.normal();
        g = std::max(0.0f, g);
        noise[i] = g;
        sum += g;
    }

    if (sum > DIRICHLET_EPSILON) {
        const float inverse_sum = 1.0f / sum;
        const float weight = dirichlet_weight_;
        const float one_minus_weight = 1.0f - weight;
        for (size_t i = 0; i < nchildren; i++) {
            children[i].prior =
                one_minus_weight * children[i].prior + weight * noise[i] * inverse_sum;
        }
    }
}

}  // namespace mcts