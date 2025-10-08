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

namespace {

constexpr char kModuleDoc[] =
    "Bindings for the high-performance C++ chess core used by Hybrid Chess AI.";

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
  case chess::KNIGHT:
    oss << 'n';
    break;
  case chess::BISHOP:
    oss << 'b';
    break;
  case chess::ROOK:
    oss << 'r';
    break;
  case chess::QUEEN:
    oss << 'q';
    break;
  default:
    break;
  }
  return oss.str();
}

std::vector<chess::Move> copy_moves(const chess::MoveList& moves) {
  return std::vector<chess::Move>(moves.begin(), moves.end());
}

py::tuple piece_bitboards_as_tuple(const chess::Position& position) {
  py::tuple outer(6);
  for (int p = 0; p < 6; ++p) {
    const auto piece = static_cast<chess::Piece>(p);
    outer[p]         = py::make_tuple(position.get_pieces(piece, chess::WHITE),
                                      position.get_pieces(piece, chess::BLACK));
  }
  return outer;
}

class PyEvaluator {
public:
  explicit PyEvaluator(py::object fn) : fn_(std::move(fn)) {}

  void operator()(const std::vector<chess::Position>& positions,
                  const std::vector<std::vector<chess::Move>>& moves,
                  std::vector<std::vector<float>>& policies,
                  std::vector<float>& values) const {
    py::gil_scoped_acquire guard;

    py::list moves_py;
    for (const auto& mv : moves) {
      moves_py.append(py::cast(mv));
    }

    py::object result = fn_(positions, moves_py);
    auto       tuple  = result.cast<py::tuple>();
    if (tuple.size() != 2) {
      throw py::value_error("evaluator must return a tuple of (policies, values)");
    }

    const auto      batch_size = static_cast<py::ssize_t>(positions.size());
    py::sequence    pol_seq    = tuple[0].cast<py::sequence>();
    py::array_t<float, py::array::c_style | py::array::forcecast> val_arr =
        tuple[1].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

    if (py::len(pol_seq) != batch_size) {
      throw py::value_error("policies must be a sequence with batch dimension first");
    }

    auto val_info = val_arr.request();
    if (val_info.ndim != 1 || val_info.shape[0] != batch_size) {
      throw py::value_error("values must be a 1D array matching the batch size");
    }

    const float* value_ptr = static_cast<const float*>(val_info.ptr);
    values.assign(value_ptr, value_ptr + static_cast<size_t>(batch_size));

    policies.clear();
    policies.reserve(static_cast<size_t>(batch_size));
    for (py::ssize_t i = 0; i < batch_size; ++i) {
      auto arr =
          pol_seq[i].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
      auto arr_info = arr.request();
      if (arr_info.ndim != 1) {
        throw py::value_error("each policy vector must be one-dimensional");
      }
      const float* policy_ptr = static_cast<const float*>(arr_info.ptr);
      const auto   count      = static_cast<size_t>(arr_info.shape[0]);
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

  py::enum_<chess::Color>(m, "Color").value("WHITE", chess::WHITE).value("BLACK", chess::BLACK);

  py::enum_<chess::Result>(m, "Result")
      .value("ONGOING", chess::ONGOING)
      .value("WHITE_WIN", chess::WHITE_WIN)
      .value("BLACK_WIN", chess::BLACK_WIN)
      .value("DRAW", chess::DRAW);
}

void bind_move(py::module_& m) {
  py::class_<chess::Move>(m, "Move", "Compact chess move representation.")
      .def(py::init<>())
      .def(py::init<int, int>(), py::arg("from_square"), py::arg("to_square"))
      .def(py::init<int, int, chess::Piece>(), py::arg("from_square"), py::arg("to_square"),
           py::arg("promotion"))
      .def_property_readonly("from_square", &chess::Move::from)
      .def_property_readonly("to_square", &chess::Move::to)
      .def_property_readonly("promotion", &chess::Move::promotion)
      .def("__repr__",
           [](const chess::Move& move) {
             std::string repr = "Move(" + square_to_string(move.from()) + "->" +
                                square_to_string(move.to());
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
      .def("reset", &chess::Position::reset, "Reset the position to the initial board state.")
      .def("from_fen", &chess::Position::from_fen, py::arg("fen"),
           "Load a position from a FEN string.")
      .def("to_fen", &chess::Position::to_fen, "Serialize the position to FEN.")
      .def("legal_moves",
           [](const chess::Position& position) { return copy_moves(position.legal_moves()); },
           "Return a list of legal moves from the current position.")
      .def("make_move", &chess::Position::make_move, py::arg("move"),
           "Apply a move and return the resulting game result.")
      .def("result", &chess::Position::result, "Return the current game result.")
      .def("count_repetitions", &chess::Position::repetition_count,
           "Return how many times this position has repeated.")
      .def_property_readonly("pieces", &piece_bitboards_as_tuple,
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
      .def(py::pickle([](const chess::Position& position) { return position.to_fen(); },
                      [](const std::string& fen) {
                        chess::Position position;
                        position.from_fen(fen);
                        return position;
                      }));
}

void bind_mcts(py::module_& m) {
  py::class_<mcts::MCTS>(m, "MCTS", "Monte Carlo Tree Search engine.")
      .def(py::init<int, float, float, float>(), py::arg("simulations") = 800,
           py::arg("c_puct") = 1.0f, py::arg("dirichlet_alpha") = 0.3f,
           py::arg("dirichlet_weight") = 0.25f)
      .def("seed", &mcts::MCTS::seed, py::arg("seed"), "Seed the internal RNG.")
      .def(
          "search_batched_legal",
          [](mcts::MCTS& engine, const chess::Position& position, py::object evaluator, int max_batch) {
            const PyEvaluator adapter(std::move(evaluator));
            py::gil_scoped_release release;
            return engine.search_batched_legal(position, adapter, max_batch);
          },
          py::arg("position"), py::arg("evaluator"), py::arg("max_batch") = 64,
          "Run batched MCTS search given a Python evaluator callback.")
      .def("set_simulations", &mcts::MCTS::set_simulations, py::arg("simulations"))
      .def("set_c_puct", &mcts::MCTS::set_c_puct, py::arg("c_puct"))
      .def("set_dirichlet_params", &mcts::MCTS::set_dirichlet_params, py::arg("alpha"), py::arg("weight"))
      .def("set_c_puct_params", &mcts::MCTS::set_c_puct_params, py::arg("c_puct_base"),
           py::arg("c_puct_init"))
      .def("set_fpu_reduction", &mcts::MCTS::set_fpu_reduction, py::arg("fpu_reduction"))
      .def("reset_tree", &mcts::MCTS::reset_tree)
      .def("advance_root", &mcts::MCTS::advance_root, py::arg("new_position"), py::arg("played_move"));
}

void bind_utilities(py::module_& m) {
  m.def(
      "encode_move_index", &mcts::encode_move_index, py::arg("move"),
      "Encode a move into an index compatible with network policy outputs.");

  m.def(
      "encode_move_indices_batch",
      [](const std::vector<std::vector<chess::Move>>& moves_lists) {
        py::list result;
        for (const auto& moves : moves_lists) {
          py::array_t<int32_t> encoded(static_cast<py::ssize_t>(moves.size()));
          auto                 info = encoded.request();
          auto*                data = static_cast<int32_t*>(info.ptr);
          for (size_t i = 0; i < moves.size(); ++i) {
            data[i] = mcts::encode_move_index(moves[i]);
          }
          result.append(std::move(encoded));
        }
        return result;
      },
      py::arg("moves_lists"), "Vectorised helper that encodes a batch of legal move lists.");

  m.def("uci_of_move", &move_to_uci, py::arg("move"), "Convert a move to UCI notation.");

  m.attr("POLICY_SIZE") = mcts::POLICY_SIZE;
  m.attr("WHITE")       = chess::WHITE;
  m.attr("BLACK")       = chess::BLACK;
  m.attr("ONGOING")     = chess::ONGOING;
  m.attr("WHITE_WIN")   = chess::WHITE_WIN;
  m.attr("BLACK_WIN")   = chess::BLACK_WIN;
  m.attr("DRAW")        = chess::DRAW;
}

} // namespace

PYBIND11_MODULE(chesscore, m) {
  m.doc() = kModuleDoc;
  chess::init_tables();

  bind_enums(m);
  bind_move(m);
  bind_position(m);
  bind_mcts(m);
  bind_utilities(m);
}
