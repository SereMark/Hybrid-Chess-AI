#include "chess.hpp"
#include "encoder.hpp"
#include "mcts.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(chessai, m) {
  chess::init_tables();

  py::enum_<chess::Piece>(m, "Piece")
      .value("PAWN", chess::PAWN)
      .value("KNIGHT", chess::KNIGHT)
      .value("BISHOP", chess::BISHOP)
      .value("ROOK", chess::ROOK)
      .value("QUEEN", chess::QUEEN)
      .value("KING", chess::KING);

  py::enum_<chess::Color>(m, "Color")
      .value("WHITE", chess::WHITE)
      .value("BLACK", chess::BLACK);

  py::enum_<chess::Result>(m, "Result")
      .value("ONGOING", chess::ONGOING)
      .value("WHITE_WIN", chess::WHITE_WIN)
      .value("BLACK_WIN", chess::BLACK_WIN)
      .value("DRAW", chess::DRAW);

  py::class_<chess::Move>(m, "Move")
      .def(py::init<>(), "Create null move")
      .def(py::init<int, int>(), "Create move from-to", py::arg("from_sq"),
           py::arg("to_sq"))
      .def(py::init<int, int, chess::Piece>(), "Create promotion move",
           py::arg("from_sq"), py::arg("to_sq"), py::arg("promotion"))
      .def("from_square", &chess::Move::from, "Source square")
      .def("to_square", &chess::Move::to, "Target square")
      .def("promotion", &chess::Move::promotion, "Promotion piece");

  py::class_<chess::Position>(m, "Position")
      .def(py::init<>(), "Create starting position")
      .def(py::init<const chess::Position &>(), "Copy position")
      .def("reset", &chess::Position::reset, "Reset to starting position")
      .def("from_fen", &chess::Position::from_fen, "Load from FEN string",
           py::arg("fen"))
      .def("to_fen", &chess::Position::to_fen, "Export to FEN string")
      .def(
          "legal_moves",
          [](chess::Position &pos) {
            chess::MoveList moves = pos.legal_moves();
            std::vector<chess::Move> result;
            result.reserve(moves.size());
            for (size_t i = 0; i < moves.size(); i++) {
              result.push_back(moves[i]);
            }
            return result;
          },
          "Generate legal moves")
      .def("make_move", &chess::Position::make_move,
           "Make move and return result", py::arg("move"))
      .def("result", &chess::Position::result, "Get game result")
      .def_property_readonly(
          "pieces",
          [](const chess::Position &pos) {
            py::list outer;
            for (int p = 0; p < 6; p++) {
              py::list inner;
              inner.append(
                  pos.get_pieces(static_cast<chess::Piece>(p), chess::WHITE));
              inner.append(
                  pos.get_pieces(static_cast<chess::Piece>(p), chess::BLACK));
              outer.append(std::move(inner));
            }
            return outer;
          },
          "Piece bitboards [piece][color]")
      .def_property_readonly("turn", &chess::Position::get_turn, "Side to move")
      .def_property_readonly("castling", &chess::Position::get_castling,
                             "Castling rights")
      .def_property_readonly("ep_square", &chess::Position::get_ep_square,
                             "En passant square")
      .def_property_readonly("halfmove", &chess::Position::get_halfmove,
                             "Halfmove clock")
      .def_property_readonly("fullmove", &chess::Position::get_fullmove,
                             "Fullmove number")
      .def_property_readonly("hash", &chess::Position::get_hash,
                             "Zobrist hash");

  py::class_<mcts::MCTS>(m, "MCTS")
      .def(py::init<int, float, float, float>(), py::arg("simulations") = 800,
           py::arg("c_puct") = 1.0f, py::arg("dirichlet_alpha") = 0.3f,
           py::arg("dirichlet_weight") = 0.25f,
           "Create MCTS engine with simulation count, exploration constant, "
           "and Dirichlet parameters")
      .def("search", &mcts::MCTS::search,
           py::arg("position"), py::arg("policy"), py::arg("value"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "search",
          [](mcts::MCTS &engine, const chess::Position &pos,
             const py::array_t<float, py::array::c_style | py::array::forcecast>
                 &policy,
             float value) {
            py::buffer_info info = policy.request();
            const float *ptr = static_cast<const float *>(info.ptr);
            std::vector<float> vec(ptr, ptr + info.size);
            py::gil_scoped_release release;
            return engine.search(pos, vec, value);
          },
          py::arg("position"), py::arg("policy"), py::arg("value"))
      .def("set_simulations", &mcts::MCTS::set_simulations,
           "Set simulation count", py::arg("sims"))
      .def("set_c_puct", &mcts::MCTS::set_c_puct, "Set exploration constant",
           py::arg("c"))
      .def("set_dirichlet_params", &mcts::MCTS::set_dirichlet_params,
           "Set Dirichlet parameters", py::arg("alpha"), py::arg("weight"));

  m.def("encode_move_index", &mcts::encode_move_index,
        "Encode a move into the 73x64 policy index");

  m.def(
      "encode_position",
      [](const chess::Position &pos) {
        constexpr int planes = 119;
        constexpr int H = 8;
        constexpr int W = 8;
        py::array_t<float> result({planes, H, W});
        py::buffer_info info = result.request();
        auto *ptr = static_cast<float *>(info.ptr);
        {
          py::gil_scoped_release release;
          encoder::encode_position_into(pos, ptr);
        }
        return result;
      },
      "Encode a single position into planes x 8 x 8 float32 array");

  m.def(
      "encode_batch",
      [](const std::vector<chess::Position> &positions) {
        constexpr int planes = 119;
        constexpr int H = 8;
        constexpr int W = 8;
        const ssize_t B = static_cast<ssize_t>(positions.size());
        py::array_t<float> result({B, static_cast<ssize_t>(planes),
                                   static_cast<ssize_t>(H),
                                   static_cast<ssize_t>(W)});
        py::buffer_info info = result.request();
        auto *ptr = static_cast<float *>(info.ptr);
        const size_t stride = static_cast<size_t>(planes * H * W);
        {
          py::gil_scoped_release release;
          for (ssize_t i = 0; i < B; i++) {
            encoder::encode_position_into(positions[static_cast<size_t>(i)],
                                          ptr + stride * static_cast<size_t>(i));
          }
        }
        return result;
      },
      "Encode a batch of positions into B x planes x 8 x 8 float32 array");

  m.def(
      "encode_batch",
      [](const std::vector<std::vector<chess::Position>> &histories) {
        constexpr int planes = 119;
        constexpr int H = 8;
        constexpr int W = 8;
        const ssize_t B = static_cast<ssize_t>(histories.size());
        py::array_t<float> result({B, static_cast<ssize_t>(planes),
                                   static_cast<ssize_t>(H),
                                   static_cast<ssize_t>(W)});
        py::buffer_info info = result.request();
        auto *ptr = static_cast<float *>(info.ptr);
        const size_t stride = static_cast<size_t>(planes * H * W);
        {
          py::gil_scoped_release release;
          for (ssize_t i = 0; i < B; i++) {
            encoder::encode_position_with_history(
                histories[static_cast<size_t>(i)],
                ptr + stride * static_cast<size_t>(i));
          }
        }
        return result;
      },
      "Encode a batch with per-item histories into B x planes x 8 x 8 float32 "
      "array");

  m.attr("INPUT_PLANES") = encoder::INPUT_PLANES;
  m.attr("HISTORY_LENGTH") = encoder::HISTORY_LENGTH;
  m.attr("PLANES_PER_POSITION") = encoder::PLANES_PER_POSITION;

  m.attr("WHITE") = chess::WHITE;
  m.attr("BLACK") = chess::BLACK;
  m.attr("ONGOING") = chess::ONGOING;
  m.attr("WHITE_WIN") = chess::WHITE_WIN;
  m.attr("BLACK_WIN") = chess::BLACK_WIN;
  m.attr("DRAW") = chess::DRAW;
  m.attr("POLICY_SIZE") = mcts::POLICY_SIZE;
}
