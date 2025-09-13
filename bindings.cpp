#include "chess.hpp"
#include "encoder.hpp"
#include "mcts.hpp"
#include "replay_buffer.hpp"

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
      .def("seed", &mcts::MCTS::seed, py::arg("seed"))
      .def(
          "search_batched_legal",
          [](mcts::MCTS& engine, const chess::Position& pos, py::object evaluator, int max_batch) {
            py::object                   eval_obj = evaluator;
            mcts::MCTS::EvalLegalBatchFn eval_fn  = [eval_obj](const std::vector<chess::Position>&          positions,
                                                              const std::vector<std::vector<chess::Move>>& moves,
                                                              std::vector<std::vector<float>>& policies_legal,
                                                              std::vector<float>&              values) {
              py::gil_scoped_acquire acq;
              py::list               moves_py;
              for (const auto& mvlist : moves) {
                py::list one;
                for (const auto& mv : mvlist)
                  one.append(mv);
                moves_py.append(std::move(one));
              }
              py::object out = eval_obj(positions, moves_py);
              if (!py::isinstance<py::tuple>(out))
                throw py::value_error("evaluator() must return (policies_legal, value)");
              auto tup = out.cast<py::tuple>();
              if (tup.size() != 2)
                throw py::value_error("evaluator() must return (policies_legal, value)");
              using PolArr        = py::list;
              PolArr pol_list     = tup[0].cast<PolArr>();
              using ValArr        = py::array_t<float, py::array::c_style | py::array::forcecast>;
              ValArr        val   = tup[1].cast<ValArr>();
              auto          vinfo = val.request();
              const ssize_t B     = static_cast<ssize_t>(positions.size());
              if (vinfo.ndim != 1 || vinfo.shape[0] != B)
                throw py::value_error("value must be [B]");
              const float* vbase = static_cast<const float*>(vinfo.ptr);
              policies_legal.clear();
              policies_legal.reserve(static_cast<size_t>(B));
              values.resize(static_cast<size_t>(B));
              for (ssize_t i = 0; i < B; ++i) {
                py::object item = pol_list[static_cast<size_t>(i)];
                using Arr1D     = py::array_t<float, py::array::c_style | py::array::forcecast>;
                Arr1D arr       = item.cast<Arr1D>();
                auto  ainfo     = arr.request();
                if (ainfo.ndim != 1)
                  throw py::value_error("each policy must be 1D");
                const float*  aptr = static_cast<const float*>(ainfo.ptr);
                const ssize_t n    = ainfo.shape[0];
                policies_legal.emplace_back();
                policies_legal.back().assign(aptr, aptr + n);
                values[static_cast<size_t>(i)] = vbase[i];
              }
            };
            py::gil_scoped_release rel;
            return engine.search_batched_legal(pos, eval_fn, max_batch);
          },
          py::arg("position"), py::arg("evaluator"), py::arg("max_batch") = 64)
      .def("set_simulations", &mcts::MCTS::set_simulations, py::arg("sims"))
      .def("set_c_puct", &mcts::MCTS::set_c_puct, py::arg("c_puct"))
      .def("set_dirichlet_params", &mcts::MCTS::set_dirichlet_params, py::arg("alpha"), py::arg("weight"))
      .def("set_c_puct_params", &mcts::MCTS::set_c_puct_params, py::arg("c_puct_base"), py::arg("c_puct_init"))
      .def("set_fpu_reduction", &mcts::MCTS::set_fpu_reduction, py::arg("fpu"))
      .def("reset_tree", &mcts::MCTS::reset_tree)
      .def("advance_root", &mcts::MCTS::advance_root, py::arg("new_position"), py::arg("played_move"));

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

  m.def(
      "encode_move_indices_batch",
      [](const std::vector<std::vector<chess::Move>>& moves_lists) {
        py::list out;
        for (const auto& lst : moves_lists) {
          py::array_t<int32_t> a(lst.size());
          auto                 info = a.request();
          auto*                ptr  = static_cast<int32_t*>(info.ptr);
          for (size_t i = 0; i < lst.size(); ++i)
            ptr[i] = mcts::encode_move_index(lst[i]);
          out.append(std::move(a));
        }
        return out;
      },
      py::arg("moves_lists"));

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

  py::class_<replaybuf::ReplayBuffer>(m, "ReplayBuffer")
      .def(py::init<size_t, int, int, int>(), py::arg("capacity"), py::arg("planes"), py::arg("height"),
           py::arg("width"))
      .def("seed", &replaybuf::ReplayBuffer::seed, py::arg("seed"))
      .def("clear", &replaybuf::ReplayBuffer::clear)
      .def("set_capacity", &replaybuf::ReplayBuffer::set_capacity, py::arg("capacity"))
      .def_property_readonly("capacity", &replaybuf::ReplayBuffer::capacity)
      .def_property_readonly("size", &replaybuf::ReplayBuffer::size)
      .def(
          "push",
          [](replaybuf::ReplayBuffer& buf, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> state,
             py::array_t<int32_t, py::array::c_style | py::array::forcecast>  idx,
             py::array_t<uint16_t, py::array::c_style | py::array::forcecast> cnt, int8_t value) {
            auto sinfo = state.request();
            if (sinfo.ndim != 3)
              throw py::value_error("state must be [P,H,W]");
            const size_t state_bytes = static_cast<size_t>(sinfo.shape[0] * sinfo.shape[1] * sinfo.shape[2]);
            auto         iinfo       = idx.request();
            auto         cinfo       = cnt.request();
            if (iinfo.ndim != 1 || cinfo.ndim != 1)
              throw py::value_error("idx/cnt must be 1D");
            if (cinfo.shape[0] != iinfo.shape[0])
              throw py::value_error("idx and cnt length mismatch");
            const uint8_t*  sp = static_cast<const uint8_t*>(sinfo.ptr);
            const int32_t*  ip = static_cast<const int32_t*>(iinfo.ptr);
            const uint16_t* cp = static_cast<const uint16_t*>(cinfo.ptr);
            buf.push(sp, state_bytes, ip, static_cast<size_t>(iinfo.shape[0]), cp, static_cast<size_t>(cinfo.shape[0]),
                     value);
          },
          py::arg("state"), py::arg("indices"), py::arg("counts"), py::arg("value"))
      .def(
          "sample",
          [](replaybuf::ReplayBuffer& buf, size_t batch, double recent_ratio, double recent_window_frac) {
            std::vector<uint8_t>  states;
            std::vector<size_t>   state_off;
            std::vector<int32_t>  idx;
            std::vector<size_t>   idx_off;
            std::vector<uint16_t> cnt;
            std::vector<size_t>   cnt_off;
            std::vector<int8_t>   val;
            buf.sample(batch, recent_ratio, recent_window_frac, states, state_off, idx, idx_off, cnt, cnt_off, val);
            const ssize_t        B = static_cast<ssize_t>(val.size());
            const ssize_t        P = static_cast<ssize_t>(buf.planes());
            const ssize_t        H = static_cast<ssize_t>(buf.height());
            const ssize_t        W = static_cast<ssize_t>(buf.width());
            py::array_t<uint8_t> a_states({B, P, H, W});
            std::memcpy(a_states.request().ptr, states.data(), states.size());
            py::list idx_list;
            py::list cnt_list;
            for (ssize_t i = 0; i < B; ++i) {
              const ssize_t         i0 = static_cast<ssize_t>(idx_off[static_cast<size_t>(i)]);
              const ssize_t         i1 = static_cast<ssize_t>(idx_off[static_cast<size_t>(i + 1)]);
              const ssize_t         c0 = static_cast<ssize_t>(cnt_off[static_cast<size_t>(i)]);
              const ssize_t         c1 = static_cast<ssize_t>(cnt_off[static_cast<size_t>(i + 1)]);
              py::array_t<int32_t>  ai(i1 - i0);
              py::array_t<uint16_t> ac(c1 - c0);
              if (i1 > i0)
                std::memcpy(ai.request().ptr, idx.data() + i0, static_cast<size_t>(i1 - i0) * sizeof(int32_t));
              if (c1 > c0)
                std::memcpy(ac.request().ptr, cnt.data() + c0, static_cast<size_t>(c1 - c0) * sizeof(uint16_t));
              idx_list.append(std::move(ai));
              cnt_list.append(std::move(ac));
            }
            py::array_t<int8_t> a_val(B);
            std::memcpy(a_val.request().ptr, val.data(), val.size());
            return py::make_tuple(std::move(a_states), std::move(idx_list), std::move(cnt_list), std::move(a_val));
          },
          py::arg("batch_size"), py::arg("recent_ratio"), py::arg("recent_window_frac"));
}
