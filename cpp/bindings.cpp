#include "chess.hpp"
#include "encoder.hpp"
#include "mcts.hpp"

#include <array>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {
inline std::string sq_to_str(int sq) {
  if (sq < 0 || sq >= chess::NSQUARES)
    return "--";
  char file = static_cast<char>('a' + (sq % chess::BOARD_SIZE));
  char rank = static_cast<char>('1' + (sq / chess::BOARD_SIZE));
  return std::string() + file + rank;
}
inline std::string move_to_uci(const chess::Move& m) {
  std::ostringstream oss;
  oss << sq_to_str(m.from()) << sq_to_str(m.to());
  switch (m.promotion()) {
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
inline void check_2d_float32(const py::buffer_info& info, const char* name, ssize_t expected_cols) {
  if (info.ndim != 2)
    throw py::value_error(std::string(name) + " must be 2D [B,N]");
  if (expected_cols >= 0 && info.shape[1] != expected_cols) {
    std::ostringstream oss;
    oss << name << " has wrong size: expected [B," << expected_cols << "], got [B," << info.shape[1] << "]";
    throw py::value_error(oss.str());
  }
  if (info.itemsize != static_cast<ssize_t>(sizeof(float)))
    throw py::value_error(std::string(name) + " must be float32");
}
inline void check_value_shape(const py::buffer_info& info, ssize_t batch) {
  if (info.ndim == 1) {
    if (info.shape[0] != batch)
      throw py::value_error("value length mismatch");
  } else if (info.ndim == 2) {
    if (info.shape[0] != batch || info.shape[1] != 1)
      throw py::value_error("value must be [B,1] if 2D");
  } else {
    throw py::value_error("value must be [B] or [B,1]");
  }
  if (info.itemsize != static_cast<ssize_t>(sizeof(float)))
    throw py::value_error("value must be float32");
}
} // namespace

PYBIND11_MODULE(chesscore, m) {
  chess::init_tables();

  py::enum_<chess::Piece>(m, "Piece")
      .value("PAWN", chess::PAWN)
      .value("KNIGHT", chess::KNIGHT)
      .value("BISHOP", chess::BISHOP)
      .value("ROOK", chess::ROOK)
      .value("QUEEN", chess::QUEEN)
      .value("KING", chess::KING);

  py::enum_<chess::Color>(m, "Color").value("WHITE", chess::WHITE).value("BLACK", chess::BLACK);

  py::enum_<chess::Result>(m, "Result")
      .value("ONGOING", chess::ONGOING)
      .value("WHITE_WIN", chess::WHITE_WIN)
      .value("BLACK_WIN", chess::BLACK_WIN)
      .value("DRAW", chess::DRAW);

  py::class_<chess::Move>(m, "Move")
      .def(py::init<>())
      .def(py::init<int, int>(), py::arg("from_sq"), py::arg("to_sq"))
      .def(py::init<int, int, chess::Piece>(), py::arg("from_sq"), py::arg("to_sq"), py::arg("promotion"))
      .def_property_readonly("from_square", &chess::Move::from)
      .def_property_readonly("to_square", &chess::Move::to)
      .def_property_readonly("promotion", &chess::Move::promotion)
      .def("__repr__",
           [](const chess::Move& mv) {
             return std::string("Move(") + sq_to_str(mv.from()) + "->" + sq_to_str(mv.to()) +
                    (mv.promotion() != chess::PIECE_NONE ? ", promo" : "") + ")";
           })
      .def("__str__", [](const chess::Move& mv) { return move_to_uci(mv); })
      .def("__int__", [](const chess::Move& mv) { return static_cast<int>(mv.data); })
      .def("__hash__", [](const chess::Move& mv) { return py::hash(py::int_(mv.data)); })
      .def("__eq__", [](const chess::Move& a, const chess::Move& b) { return a == b; });

  py::class_<chess::Position>(m, "Position")
      .def(py::init<>())
      .def(py::init<const chess::Position&>(), py::arg("other"))
      .def("reset", &chess::Position::reset)
      .def("from_fen", &chess::Position::from_fen, py::arg("fen"))
      .def("to_fen", &chess::Position::to_fen)
      .def("legal_moves",
           [](chess::Position& pos) {
             chess::MoveList          moves = pos.legal_moves();
             std::vector<chess::Move> out;
             out.reserve(moves.size());
             for (size_t i = 0; i < moves.size(); ++i)
               out.push_back(moves[i]);
             return out;
           })
      .def("make_move", &chess::Position::make_move, py::arg("move"))
      .def("result", &chess::Position::result)
      .def("count_repetitions", &chess::Position::repetition_count)
      .def_property_readonly("pieces",
                             [](const chess::Position& pos) {
                               py::list outer;
                               for (int p = 0; p < 6; ++p) {
                                 py::list inner;
                                 inner.append(pos.get_pieces(static_cast<chess::Piece>(p), chess::WHITE));
                                 inner.append(pos.get_pieces(static_cast<chess::Piece>(p), chess::BLACK));
                                 outer.append(std::move(inner));
                               }
                               return outer;
                             })
      .def_property_readonly("turn", &chess::Position::get_turn)
      .def_property_readonly("castling", &chess::Position::get_castling)
      .def_property_readonly("ep_square", &chess::Position::get_ep_square)
      .def_property_readonly("halfmove", &chess::Position::get_halfmove)
      .def_property_readonly("fullmove", &chess::Position::get_fullmove)
      .def_property_readonly("hash", &chess::Position::get_hash)
      .def("__repr__", [](const chess::Position& p) { return std::string("Position(") + p.to_fen() + ")"; })
      .def("__str__", [](const chess::Position& p) { return p.to_fen(); })
      .def(py::pickle([](const chess::Position& p) { return p.to_fen(); },
                      [](const std::string& fen) {
                        chess::Position p;
                        p.from_fen(fen);
                        return p;
                      }));

  py::class_<mcts::MCTS>(m, "MCTS")
      .def(py::init<int, float, float, float>(), py::arg("simulations") = 800, py::arg("c_puct") = 1.0f,
           py::arg("dirichlet_alpha") = 0.3f, py::arg("dirichlet_weight") = 0.25f)
      .def(
          "search_batched",
          [](mcts::MCTS& engine, const chess::Position& pos, py::object evaluator, int max_batch) {
            auto eval_fn = [&evaluator](const std::vector<chess::Position>& positions,
                                        std::vector<std::vector<float>>& policies, std::vector<float>& values) {
              py::gil_scoped_acquire acq;
              py::object             out = evaluator(positions);
              if (!py::isinstance<py::tuple>(out))
                throw py::value_error("evaluator() must return (policy, value)");
              auto tup = out.cast<py::tuple>();
              if (tup.size() != 2)
                throw py::value_error("evaluator() must return (policy, value)");
              using Arr  = py::array_t<float, py::array::c_style | py::array::forcecast>;
              Arr  pol   = tup[0].cast<Arr>();
              Arr  val   = tup[1].cast<Arr>();
              auto pinfo = pol.request();
              auto vinfo = val.request();
              check_2d_float32(pinfo, "policy", mcts::POLICY_SIZE);
              const ssize_t B = pinfo.shape[0];
              check_value_shape(vinfo, B);
              const float*  pbase = static_cast<const float*>(pinfo.ptr);
              const float*  vbase = static_cast<const float*>(vinfo.ptr);
              const ssize_t P     = pinfo.shape[1];
              const bool    v2d   = (vinfo.ndim == 2);
              const ssize_t vcol  = v2d ? vinfo.shape[1] : 1;
              policies.resize(static_cast<size_t>(B));
              values.resize(static_cast<size_t>(B));
              for (ssize_t i = 0; i < B; ++i) {
                policies[static_cast<size_t>(i)].assign(pbase + i * P, pbase + (i + 1) * P);
                values[static_cast<size_t>(i)] = v2d ? vbase[i * vcol] : vbase[i];
              }
            };
            py::gil_scoped_release rel;
            return engine.search_batched(pos, eval_fn, max_batch);
          },
          py::arg("position"), py::arg("evaluator"), py::arg("max_batch") = 64)
      .def("set_simulations", &mcts::MCTS::set_simulations, py::arg("sims"))
      .def("set_c_puct", &mcts::MCTS::set_c_puct, py::arg("c_puct"))
      .def("set_dirichlet_params", &mcts::MCTS::set_dirichlet_params, py::arg("alpha"), py::arg("weight"))
      .def("set_c_puct_params", &mcts::MCTS::set_c_puct_params, py::arg("c_puct_base"), py::arg("c_puct_init"))
      .def("set_fpu_reduction", &mcts::MCTS::set_fpu_reduction, py::arg("fpu"));

  m.def("encode_move_index", &mcts::encode_move_index);

  m.def(
      "encode_position",
      [](const chess::Position& pos) {
        constexpr int      planes = encoder::INPUT_PLANES;
        constexpr int      H = chess::BOARD_SIZE, W = chess::BOARD_SIZE;
        py::array_t<float> a({planes, H, W});
        auto               info = a.request();
        auto*              ptr  = static_cast<float*>(info.ptr);
        {
          py::gil_scoped_release rel;
          encoder::encode_position_into(pos, ptr);
        }
        return a;
      },
      py::arg("position"));

  m.def(
      "encode_batch",
      [](const std::vector<chess::Position>& positions) {
        constexpr int      planes = encoder::INPUT_PLANES;
        constexpr int      H = chess::BOARD_SIZE, W = chess::BOARD_SIZE;
        const ssize_t      B = static_cast<ssize_t>(positions.size());
        py::array_t<float> a({B, static_cast<ssize_t>(planes), static_cast<ssize_t>(H), static_cast<ssize_t>(W)});
        auto               info   = a.request();
        auto*              ptr    = static_cast<float*>(info.ptr);
        const size_t       stride = static_cast<size_t>(planes * H * W);
        {
          py::gil_scoped_release rel;
          for (ssize_t i = 0; i < B; ++i)
            encoder::encode_position_into(positions[static_cast<size_t>(i)], ptr + stride * static_cast<size_t>(i));
        }
        return a;
      },
      py::arg("positions"));

  m.def(
      "encode_batch",
      [](const std::vector<std::vector<chess::Position>>& histories) {
        constexpr int      planes = encoder::INPUT_PLANES;
        constexpr int      H = chess::BOARD_SIZE, W = chess::BOARD_SIZE;
        const ssize_t      B = static_cast<ssize_t>(histories.size());
        py::array_t<float> a({B, static_cast<ssize_t>(planes), static_cast<ssize_t>(H), static_cast<ssize_t>(W)});
        auto               info   = a.request();
        auto*              ptr    = static_cast<float*>(info.ptr);
        const size_t       stride = static_cast<size_t>(planes * H * W);
        {
          py::gil_scoped_release rel;
          for (ssize_t i = 0; i < B; ++i)
            encoder::encode_position_with_history(histories[static_cast<size_t>(i)],
                                                  ptr + stride * static_cast<size_t>(i));
        }
        return a;
      },
      py::arg("histories"));

  m.attr("INPUT_PLANES")        = encoder::INPUT_PLANES;
  m.attr("HISTORY_LENGTH")      = encoder::HISTORY_LENGTH;
  m.attr("PLANES_PER_POSITION") = encoder::PLANES_PER_POSITION;
  m.attr("POLICY_SIZE")         = mcts::POLICY_SIZE;

  m.attr("WHITE")     = chess::WHITE;
  m.attr("BLACK")     = chess::BLACK;
  m.attr("ONGOING")   = chess::ONGOING;
  m.attr("WHITE_WIN") = chess::WHITE_WIN;
  m.attr("BLACK_WIN") = chess::BLACK_WIN;
  m.attr("DRAW")      = chess::DRAW;

  m.def("square_str", [](int sq) { return sq_to_str(sq); }, py::arg("square"));
  m.def("uci_of_move", [](const chess::Move& mv) { return move_to_uci(mv); }, py::arg("move"));
}
