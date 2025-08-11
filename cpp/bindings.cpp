#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "chess.hpp"
#include "encoder.hpp"
#include "mcts.hpp"

namespace py = pybind11;

PYBIND11_MODULE(chesscore, m) {
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
      .def(py::init<>())
      .def(py::init<int, int>(), py::arg("from_sq"), py::arg("to_sq"))
      .def(py::init<int, int, chess::Piece>(), py::arg("from_sq"),
           py::arg("to_sq"), py::arg("promotion"))
      .def_property_readonly("from_square", &chess::Move::from)
      .def_property_readonly("to_square", &chess::Move::to)
      .def_property_readonly("promotion", &chess::Move::promotion);

  py::class_<chess::Position>(m, "Position")
      .def(py::init<>())
      .def(py::init<const chess::Position &>())
      .def("reset", &chess::Position::reset)
      .def("from_fen", &chess::Position::from_fen, py::arg("fen"))
      .def("to_fen", &chess::Position::to_fen)
      .def("legal_moves",
           [](chess::Position &pos) {
             chess::MoveList moves = pos.legal_moves();
             std::vector<chess::Move> result;
             result.reserve(moves.size());
             for (size_t i = 0; i < moves.size(); i++) {
               result.push_back(moves[i]);
             }
             return result;
           })
      .def("make_move", &chess::Position::make_move, py::arg("move"))
      .def("result", &chess::Position::result)
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
          })
      .def_property_readonly("turn", &chess::Position::get_turn)
      .def_property_readonly("castling", &chess::Position::get_castling)
      .def_property_readonly("ep_square", &chess::Position::get_ep_square)
      .def_property_readonly("halfmove", &chess::Position::get_halfmove)
      .def_property_readonly("fullmove", &chess::Position::get_fullmove)
      .def_property_readonly("hash", &chess::Position::get_hash);

  py::class_<mcts::MCTS>(m, "MCTS")
      .def(py::init<int, float, float, float>(), py::arg("simulations") = 800,
           py::arg("c_puct") = 1.0f, py::arg("dirichlet_alpha") = 0.3f,
           py::arg("dirichlet_weight") = 0.25f)
      .def(
          "search_batched",
          [](mcts::MCTS &engine, const chess::Position &pos,
             py::object evaluator, int max_batch) {
            auto eval_fn = [&evaluator](
                               const std::vector<chess::Position> &positions,
                               std::vector<std::vector<float>> &policies,
                               std::vector<float> &values) {
              py::gil_scoped_acquire acquire;
              py::object out = evaluator(positions);
              auto tuple_out = out.cast<py::tuple>();
              py::array_t<float> pol = tuple_out[0].cast<py::array_t<float>>();
              py::array_t<float> val = tuple_out[1].cast<py::array_t<float>>();
              py::buffer_info pinfo = pol.request();
              py::buffer_info vinfo = val.request();
              const float *pbase = static_cast<const float *>(pinfo.ptr);
              const float *vbase = static_cast<const float *>(vinfo.ptr);
              ssize_t B = pinfo.shape[0];
              ssize_t P = pinfo.shape[1];
              policies.resize(static_cast<size_t>(B));
              values.resize(static_cast<size_t>(B));
              for (ssize_t i = 0; i < B; i++) {
                policies[static_cast<size_t>(i)].assign(pbase + i * P,
                                                        pbase + (i + 1) * P);
                values[static_cast<size_t>(i)] = vbase[i];
              }
            };
            py::gil_scoped_release release;
            return engine.search_batched(pos, eval_fn, max_batch);
          },
          py::arg("position"), py::arg("evaluator"), py::arg("max_batch") = 64)
      .def("set_simulations", &mcts::MCTS::set_simulations, py::arg("sims"))
      .def("set_c_puct", &mcts::MCTS::set_c_puct, py::arg("c_puct"))
      .def("set_dirichlet_params", &mcts::MCTS::set_dirichlet_params,
           py::arg("alpha"), py::arg("weight"))
      .def("set_c_puct_params", &mcts::MCTS::set_c_puct_params,
           py::arg("c_puct_base"), py::arg("c_puct_init"));

  m.def("encode_move_index", &mcts::encode_move_index);

  m.def(
      "encode_position",
      [](const chess::Position &pos) {
        constexpr int planes = encoder::INPUT_PLANES;
        constexpr int H = chess::BOARD_SIZE;
        constexpr int W = chess::BOARD_SIZE;
        py::array_t<float> result({planes, H, W});
        py::buffer_info info = result.request();
        auto *ptr = static_cast<float *>(info.ptr);
        {
          py::gil_scoped_release release;
          encoder::encode_position_into(pos, ptr);
        }
        return result;
      },
      py::arg("position"));

  m.def(
      "encode_batch",
      [](const std::vector<chess::Position> &positions) {
        constexpr int planes = encoder::INPUT_PLANES;
        constexpr int H = chess::BOARD_SIZE;
        constexpr int W = chess::BOARD_SIZE;
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
                                          ptr +
                                              stride * static_cast<size_t>(i));
          }
        }
        return result;
      },
      py::arg("positions"));

  m.def(
      "encode_batch",
      [](const std::vector<std::vector<chess::Position>> &histories) {
        constexpr int planes = encoder::INPUT_PLANES;
        constexpr int H = chess::BOARD_SIZE;
        constexpr int W = chess::BOARD_SIZE;
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
      py::arg("histories"));

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
