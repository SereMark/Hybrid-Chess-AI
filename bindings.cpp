#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "chess.hpp"
#include "mcts.hpp"

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
        .def(py::init<int, int>(), "Create move from-to", py::arg("from_sq"), py::arg("to_sq"))
        .def(py::init<int, int, chess::Piece>(), "Create promotion move", 
             py::arg("from_sq"), py::arg("to_sq"), py::arg("promotion"))
        .def("from_square", &chess::Move::from, "Source square")
        .def("to_square", &chess::Move::to, "Target square")
        .def("promotion", &chess::Move::promotion, "Promotion piece");
    
    py::class_<chess::Position>(m, "Position")
        .def(py::init<>(), "Create starting position")
        .def(py::init<const chess::Position&>(), "Copy position")
        .def("reset", &chess::Position::reset, "Reset to starting position")
        .def("from_fen", &chess::Position::from_fen, "Load from FEN string", py::arg("fen"))
        .def("to_fen", &chess::Position::to_fen, "Export to FEN string")
        .def("legal_moves", &chess::Position::legal_moves, "Generate legal moves")
        .def("make_move", &chess::Position::make_move, "Make move and return result", py::arg("move"))
        .def("result", &chess::Position::result, "Get game result")
        .def_property_readonly("pieces", [](const chess::Position& pos) {
            py::list outer;
            for (int p = 0; p < 6; p++) {
                py::list inner;
                inner.append(pos.pieces[p][0]);
                inner.append(pos.pieces[p][1]);
                outer.append(std::move(inner));
            }
            return outer;
        }, "Piece bitboards [piece][color]")
        .def_readonly("turn", &chess::Position::turn, "Side to move")
        .def_readonly("castling", &chess::Position::castling, "Castling rights")
        .def_readonly("ep_square", &chess::Position::ep_square, "En passant square")
        .def_readonly("halfmove", &chess::Position::halfmove, "Halfmove clock")
        .def_readonly("fullmove", &chess::Position::fullmove, "Fullmove number")
        .def_readonly("hash", &chess::Position::hash, "Zobrist hash");
    
  
    py::class_<mcts::MCTS>(m, "MCTS")
        .def(py::init<int, float, float, float>(), 
             py::arg("simulations") = 800, 
             py::arg("c_puct") = 1.0f,
             py::arg("dirichlet_alpha") = 0.3f,
             py::arg("dirichlet_weight") = 0.25f,
             "Create MCTS engine with simulation count, exploration constant, and Dirichlet parameters")
        .def("search", &mcts::MCTS::search, 
             "Run MCTS search and return visit counts",
             py::arg("position"), py::arg("policy"), py::arg("value"))
        .def("set_simulations", &mcts::MCTS::set_simulations, "Set simulation count", py::arg("sims"))
        .def("set_c_puct", &mcts::MCTS::set_c_puct, "Set exploration constant", py::arg("c"))
        .def("set_dirichlet_params", &mcts::MCTS::set_dirichlet_params, "Set Dirichlet parameters", py::arg("alpha"), py::arg("weight"));
    
    m.def("lsb", &chess::lsb, "Find least significant bit (fastest)", py::arg("bb"));
    m.def("popcount", &chess::popcount, "Count set bits (hardware accelerated)", py::arg("bb"));
    m.def("bit", &chess::bit, "Create single-bit bitboard", py::arg("square"));
    
    m.attr("WHITE") = chess::WHITE;
    m.attr("BLACK") = chess::BLACK;
    m.attr("ONGOING") = chess::ONGOING;
    m.attr("WHITE_WIN") = chess::WHITE_WIN;
    m.attr("BLACK_WIN") = chess::BLACK_WIN;
    m.attr("DRAW") = chess::DRAW;
}