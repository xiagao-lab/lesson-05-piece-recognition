from xiagao_chess.recognition import layout_to_board


def test_layout_to_board_contract():
    layout = "\n".join(
        [
            "3k5",
            ".........",
            ".........",
            ".........",
            "..p......",
            ".........",
            ".........",
            "....B.C..",
            "....A....",
            "..r.K.B..",
        ]
    )
    # FEN 式数字不是本识别布局的合法表示。
    try:
        layout_to_board(layout)
    except ValueError:
        pass
    else:
        raise AssertionError("识别布局必须是每行 9 个显式字符")


def test_explicit_layout_to_board():
    layout = "\n".join(
        [
            "...k.....",
            ".........",
            ".........",
            ".........",
            "..p......",
            ".........",
            ".........",
            "....B.C..",
            "....A....",
            "..r.K.B..",
        ]
    )
    board = layout_to_board(layout)
    assert len(board) == 8
    assert board[0] == {"piece": "blackGeneral", "cord": {"row": 10, "col": 4}}
