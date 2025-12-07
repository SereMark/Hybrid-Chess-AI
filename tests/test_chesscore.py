from __future__ import annotations

import pytest

pytest.importorskip("chesscore", reason="hiányzik a chesscore kiterjesztés")

import chesscore as ccore


@pytest.mark.usefixtures("ensure_chesscore")
class TestChessLogicScientific:

    def test_scholar_mate_sequence(self):
        pos = ccore.Position()
        moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]

        for uci in moves:
            assert pos.result() == ccore.ONGOING
            found = False
            for m in pos.legal_moves():
                if ccore.uci_of_move(m) == uci:
                    pos.make_move(m)
                    found = True
                    break
            assert found, f"A(z) {uci} lépés nem szabályos a(z) {pos.to_fen()} állásban"

        assert pos.result() == ccore.WHITE_WIN
        assert len(pos.legal_moves()) == 0, "Matt helyzetben 0 szabályos lépésnek kell lennie"

    def test_stalemate_detection(self):
        pos = ccore.Position()
        pos.from_fen("4k3/4P3/4K3/8/8/8/8/8 b - - 0 1")

        assert len(pos.legal_moves()) == 0
        assert pos.result() == ccore.DRAW

    def test_insufficient_material(self):
        pos = ccore.Position()
        pos.from_fen("k7/8/8/8/8/5n2/7K/8 w - - 0 1")
        assert pos.result() == ccore.DRAW

    def test_fen_roundtrip_fidelity(self):
        complex_fen = "r3k2r/pp1b1ppp/1qnbpn2/3p4/3P4/1PNBPN2/P4PPP/R1BQK2R w KQkq - 1 10"
        pos = ccore.Position()
        pos.from_fen(complex_fen)
        out_fen = pos.to_fen()

        assert complex_fen.split()[0] == out_fen.split()[0]
        assert complex_fen.split()[2] == out_fen.split()[2]

    def test_perft_depth_1(self):
        pos = ccore.Position()
        assert len(pos.legal_moves()) == 20

        pos.from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        assert len(pos.legal_moves()) == 48

    def test_mate_in_one_mechanics(self):
        pos = ccore.Position()
        pos.from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")

        mate_move = None
        for m in pos.legal_moves():
            if ccore.uci_of_move(m) == "a1a8":
                mate_move = m
                break

        assert mate_move is not None, "az a1a8 lépésnek szabályosnak kell lennie"

        pos.make_move(mate_move)
        assert pos.result() == ccore.WHITE_WIN, (
            f"Az a1a8 után az eredménynek világos győzelme (WHITE_WIN) kellene legyen, " f"de {pos.result()} lett"
        )
