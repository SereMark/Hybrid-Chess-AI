#include "chess.hpp"
#include "mcts.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace detail {

constexpr char kModuleDoc[] = "Python bindings for the C++ chess core.";

std::string square_to_string(int square) {
  if (square < 0 || square >= chess::NSQUARES) {
    return "--";
  }
  const char file = static_cast<char>('a' + (square % chess::BOARD_SIZE));
  const char rank = static_cast<char>('1' + (square / chess::BOARD_SIZE));
  return std::string{file, rank};
}

std::string move_to_uci(const chess::Move& move) {
  std::ostringstream oss;
  oss << square_to_string(move.from()) << square_to_string(move.to());
  switch (move.promotion()) {
    case chess::KNIGHT: oss << 'n'; break;
    case chess::BISHOP: oss << 'b'; break;
    case chess::ROOK:   oss << 'r'; break;
    case chess::QUEEN:  oss << 'q'; break;
    default: break;
  }
  return oss.str();
}

std::vector<chess::Move> copy_moves(const chess::MoveList& moves) {
  return {moves.begin(), moves.end()};
}

py::tuple piece_bitboards_as_tuple(const chess::Position& position) {
  py::tuple outer(6);
  for (int p = 0; p < 6; ++p) {
    const auto piece = static_cast<chess::Piece>(p);
    outer[p] = py::make_tuple(position.get_pieces(piece, chess::WHITE),
                              position.get_pieces(piece, chess::BLACK));
  }
  return outer;
}

class PythonEvaluator {
public:
  explicit PythonEvaluator(py::object fn) : fn_(std::move(fn)) {}

  void operator()(const std::vector<chess::Position>& positions,
                  const std::vector<int32_t>& encoded_moves,
                  const std::vector<int>& counts,
                  std::vector<std::vector<float>>& policies,
                  std::vector<float>& values) const {

    py::gil_scoped_acquire acquire;

    py::array_t<int32_t> flat_arr(static_cast<py::ssize_t>(encoded_moves.size()));
    std::memcpy(flat_arr.mutable_data(), encoded_moves.data(), encoded_moves.size() * sizeof(int32_t));

    py::list counts_py;
    for (int c : counts) {
      counts_py.append(c);
    }

    const py::object result = fn_(positions, flat_arr, counts_py);
    const auto tuple = result.cast<py::tuple>();
    if (tuple.size() != 2) {
      throw py::value_error("evaluator must return a (policies, values) tuple");
    }

    const auto batch_size = static_cast<py::ssize_t>(positions.size());
    const py::sequence pol_seq = tuple[0].cast<py::sequence>();
    const auto val_arr =
        tuple[1].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

    if (py::len(pol_seq) != batch_size) {
      throw py::value_error("policies sequence must start with batch dimension");
    }

    const auto val_info = val_arr.request();
    if (val_info.ndim != 1 || val_info.shape[0] != batch_size) {
      throw py::value_error("values array must be 1D and match batch size");
    }
    const auto* value_ptr = static_cast<const float*>(val_info.ptr);
    values.assign(value_ptr, value_ptr + static_cast<size_t>(batch_size));

    policies.clear();
    policies.reserve(static_cast<size_t>(batch_size));
    for (py::ssize_t i = 0; i < batch_size; ++i) {
      const auto arr =
          pol_seq[i].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
      const auto arr_info = arr.request();
      if (arr_info.ndim != 1) {
        throw py::value_error("each policy vector must be 1D");
      }

      const size_t count = static_cast<size_t>(arr_info.shape[0]);
      const size_t expected = static_cast<size_t>(counts[static_cast<size_t>(i)]);
      if (count != expected) {
        throw py::value_error(
            "policy length must match the number of legal moves from the position");
      }

      const auto* policy_ptr = static_cast<const float*>(arr_info.ptr);
      policies.emplace_back(policy_ptr, policy_ptr + count);
    }
  }

private:
  py::object fn_;
};

void bind_enums(py::module_& m) {
  py::enum_<chess::Piece>(m, "Piece")
      .value("PAWN", chess::PAWN)
      .value("KNIGHT", chess::KNIGHT)
      .value("BISHOP", chess::BISHOP)
      .value("ROOK", chess::ROOK)
      .value("QUEEN", chess::QUEEN)
      .value("KING", chess::KING)
      .value("NONE", chess::PIECE_NONE);

  py::enum_<chess::Color>(m, "Color")
      .value("WHITE", chess::WHITE)
      .value("BLACK", chess::BLACK);

  py::enum_<chess::Result>(m, "Result")
      .value("ONGOING", chess::ONGOING)
      .value("WHITE_WIN", chess::WHITE_WIN)
      .value("BLACK_WIN", chess::BLACK_WIN)
      .value("DRAW", chess::DRAW);
}

void bind_move(py::module_& m) {
  py::class_<chess::Move>(m, "Move", "Compact chess move representation.")
      .def(py::init<>())
      .def(py::init<chess::Square, chess::Square, chess::Piece>(),
           py::arg("from_square"),
           py::arg("to_square"),
           py::arg("promotion") = chess::PIECE_NONE)
      .def_property_readonly("from_square", &chess::Move::from)
      .def_property_readonly("to_square", &chess::Move::to)
      .def_property_readonly("promotion", &chess::Move::promotion)
      .def("__repr__", [](const chess::Move& move) {
        std::string repr = "Move(" + square_to_string(move.from()) + "->" + square_to_string(move.to());
        if (move.promotion() != chess::PIECE_NONE) {
          repr += ", promo=" + std::to_string(static_cast<int>(move.promotion()));
        }
        repr += ")";
        return repr;
      })
      .def("__str__", &move_to_uci)
      .def("__int__", [](const chess::Move& move) { return static_cast<int>(move.data); })
      .def("__hash__", [](const chess::Move& move) { return py::hash(py::int_(move.data)); })
      .def("__eq__", [](const chess::Move& lhs, const chess::Move& rhs) { return lhs == rhs; });
}

void bind_position(py::module_& m) {
  py::class_<chess::Position>(m, "Position", "Mutable chess position with fast move generation.")
      .def(py::init<>())
      .def(py::init<const chess::Position&>(), py::arg("other"))
      .def("reset", &chess::Position::reset,
           "Resets the position to the initial board state.")
      .def("from_fen", &chess::Position::from_fen, py::arg("fen"),
           "Loads position from a FEN string.")
      .def("to_fen", &chess::Position::to_fen,
           "Converts the position to a FEN string.")
      .def("legal_moves",
           [](chess::Position& position) { return copy_moves(position.legal_moves()); },
           "Returns a list of legal moves from the current position.")
      .def("make_move", &chess::Position::make_move, py::arg("move"),
           "Applies the move and returns the result.")
      .def("result", &chess::Position::result,
           "Returns the current game result.")
      .def("in_check", &chess::Position::in_check,
           "Checks if the side to move is in check.")
      .def("count_repetitions", &chess::Position::repetition_count,
           "Returns the number of times this position has been repeated.")
      .def_property_readonly(
          "pieces", &piece_bitboards_as_tuple,
          "Bitboards (white, black) for each piece type.")
      .def_property_readonly("turn", &chess::Position::get_turn)
      .def_property_readonly("castling", &chess::Position::get_castling)
      .def_property_readonly("ep_square", &chess::Position::get_ep_square)
      .def_property_readonly("halfmove", &chess::Position::get_halfmove)
      .def_property_readonly("fullmove", &chess::Position::get_fullmove)
      .def_property_readonly("hash", &chess::Position::get_hash)
      .def("__repr__", [](const chess::Position& position) {
        return "Position(" + position.to_fen() + ")";
      })
      .def("__str__", &chess::Position::to_fen)
      .def(py::pickle(
          [](const chess::Position& position) { return position.to_fen(); },
          [](const std::string& fen) {
            chess::Position position;
            position.from_fen(fen);
            return position;
          }));
}

void bind_mcts(py::module_& m) {
  py::class_<mcts::MCTS>(m, "MCTS", "Monte Carlo Tree Search (MCTS) engine.")
      .def(py::init<int, float, float, float>(),
           py::arg("simulations") = 800,
           py::arg("c_puct") = 1.0f,
           py::arg("dirichlet_alpha") = 0.3f,
           py::arg("dirichlet_weight") = 0.25f)
      .def("seed", &mcts::MCTS::seed, py::arg("seed"),
           "Initializes the internal random number generator.")
      .def(
          "search_batched_legal",
          [](mcts::MCTS& engine, const chess::Position& position, py::object evaluator, int max_batch) {
            const PythonEvaluator adapter(std::move(evaluator));
            py::gil_scoped_release release;
            return engine.search_batched_legal(position, adapter, max_batch);
          },
          py::arg("position"),
          py::arg("evaluator"),
          py::arg("max_batch") = 64,
          "Runs batched MCTS search with the provided Python evaluator callback.")
      .def("set_simulations", &mcts::MCTS::set_simulations, py::arg("simulations"))
      .def("set_c_puct", &mcts::MCTS::set_c_puct, py::arg("c_puct"))
      .def("set_dirichlet_params", &mcts::MCTS::set_dirichlet_params, py::arg("alpha"), py::arg("weight"))
      .def("set_c_puct_params", &mcts::MCTS::set_c_puct_params, py::arg("c_puct_base"), py::arg("c_puct_init"))
      .def("set_fpu_reduction", &mcts::MCTS::set_fpu_reduction, py::arg("fpu_reduction"))
      .def("reset_tree", &mcts::MCTS::reset_tree)
      .def("advance_root", &mcts::MCTS::advance_root, py::arg("new_position"), py::arg("played_move"));
}

void bind_utilities(py::module_& m) {
  m.def("encode_move_index", &mcts::encode_move_index, py::arg("move"), py::arg("flip") = false,
        "Encodes a move into an index compatible with network policy outputs.");

  m.def(
      "encode_move_indices_batch",
      [](const std::vector<std::vector<chess::Move>>& moves_lists, const std::vector<int>& turns) {
        if (moves_lists.size() != turns.size()) {
          throw std::invalid_argument(
              "moves_lists and turns parameters must have the same size");
        }
        py::list result;
        for (size_t k = 0; k < moves_lists.size(); ++k) {
          const auto& moves = moves_lists[k];
          const bool flip = (turns[k] == static_cast<int>(chess::BLACK));
          py::array_t<int32_t> encoded(static_cast<py::ssize_t>(moves.size()));
          const auto info = encoded.request();
          auto* data = static_cast<int32_t*>(info.ptr);
          for (size_t i = 0; i < moves.size(); ++i) {
            const int idx = mcts::encode_move_index(moves[i], flip);
            if (idx < 0) {
              throw std::runtime_error("encode_move_index: move cannot be encoded");
            }
            data[i] = idx;
          }
          result.append(std::move(encoded));
        }
        return result;
      },
      py::arg("moves_lists"), py::arg("turns"),
      "Vectorized utility that encodes a batch of legal move lists and flips moves for Black's turn.");

  m.def("uci_of_move", &move_to_uci, py::arg("move"),
        "Converts a move to UCI notation.");

  m.attr("POLICY_SIZE") = mcts::POLICY_SIZE;
  m.attr("BOARD_SIZE") = chess::BOARD_SIZE;
  m.attr("NSQUARES") = chess::NSQUARES;

  m.attr("WHITE") = chess::WHITE;
  m.attr("BLACK") = chess::BLACK;
  m.attr("ONGOING") = chess::ONGOING;
  m.attr("WHITE_WIN") = chess::WHITE_WIN;
  m.attr("BLACK_WIN") = chess::BLACK_WIN;
  m.attr("DRAW") = chess::DRAW;
}

}

PYBIND11_MODULE(chesscore, m) {
  m.doc() = detail::kModuleDoc;
  chess::init_tables();

  detail::bind_enums(m);
  detail::bind_move(m);
  detail::bind_position(m);
  detail::bind_mcts(m);
  detail::bind_utilities(m);
}
