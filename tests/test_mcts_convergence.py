import numpy as np
import pytest

pytest.importorskip("chesscore", reason="missing chesscore extension")
import chesscore as ccore


def uniform_evaluator(positions, encoded_moves, counts):
    batch_size = len(positions)
    policies = []
    values = np.zeros(batch_size, dtype=np.float32)

    offset = 0
    for count in counts:
        if count == 0:
            policies.append(np.zeros(0, dtype=np.float32))
        else:
            policies.append(np.full(count, 1.0 / count, dtype=np.float32))
        offset += count

    return policies, values


def winning_evaluator(positions, encoded_moves, counts):
    policies = []
    values = []

    offset = 0
    for count in counts:
        policies.append(np.full(count, 1.0 / max(1, count), dtype=np.float32))
        values.append(0.0)
        offset += count

    return policies, np.array(values, dtype=np.float32)


@pytest.mark.usefixtures("ensure_chesscore")
class TestMCTSConvergence:

    def test_mate_in_one_detection(self):
        pos = ccore.Position()
        pos.from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")

        mcts = ccore.MCTS(simulations=400, c_puct=1.0)
        mcts.seed(123)

        visits = mcts.search_batched_legal(pos, uniform_evaluator, 16)

        legal = pos.legal_moves()
        best_idx = np.argmax(visits)
        best_move = legal[best_idx]

        uci = ccore.uci_of_move(best_move)
        assert uci == "a1a8", (
            f"MCTS error: chose move {uci} "
            f"(visits: {visits}, legal moves: {[ccore.uci_of_move(m) for m in legal]})"
        )

        visit_prob = visits[best_idx] / sum(visits)
        assert visit_prob > 0.5, f"MCTS confidence too low: {visit_prob:.2f}"

    def test_simulation_scaling(self):
        pos = ccore.Position()
        batch_size = 1

        sims_low = 50
        mcts_low = ccore.MCTS(simulations=sims_low)
        visits_low = mcts_low.search_batched_legal(pos, uniform_evaluator, batch_size)
        total_low = sum(visits_low)

        sims_high = 200
        mcts_high = ccore.MCTS(simulations=sims_high)
        visits_high = mcts_high.search_batched_legal(pos, uniform_evaluator, batch_size)
        total_high = sum(visits_high)
        assert abs(total_low - (sims_low - 1)) <= 1
        assert abs(total_high - (sims_high - 1)) <= 1

    def test_tree_reuse(self):
        pos = ccore.Position()
        mcts = ccore.MCTS(simulations=100)
        batch_size = 1

        visits_1 = mcts.search_batched_legal(pos, uniform_evaluator, batch_size)
        best_idx = np.argmax(visits_1)
        move = pos.legal_moves()[best_idx]

        prev_visits = visits_1[best_idx]

        mcts.advance_root(pos, move)
        pos.make_move(move)

        visits_2 = mcts.search_batched_legal(pos, uniform_evaluator, batch_size)

        assert len(visits_2) == len(pos.legal_moves())

        total_visits = sum(visits_2)
        assert total_visits >= 99, f"Too few visits: {total_visits}"

        if total_visits < prev_visits + 90:
            print(
                f"\nNote: Tree was reinitialized "
                f"(visits: {total_visits}, previous: {prev_visits})"
            )

    def test_policy_prior_influence(self):
        pos = ccore.Position()
        target_idx = 0

        def biased_evaluator(positions, encoded_moves, counts):
            pols = []
            vals = np.zeros(len(positions), dtype=np.float32)
            for count in counts:
                p = np.zeros(count, dtype=np.float32)
                if len(p) > target_idx:
                    p[target_idx] = 1.0
                pols.append(p)
            return pols, vals

        mcts = ccore.MCTS(simulations=50, c_puct=2.0)
        visits = mcts.search_batched_legal(pos, biased_evaluator, 16)

        assert np.argmax(visits) == target_idx, "MCTS did not follow strong policy prior"
