import pytest
from sudoku.sudoku import Sudoku, InvalidInput, SudokuCell, UnsolvableSudoku

CLI_DISPLAY = """
. . 3|. 2 .|6 . .
9 . .|3 . 5|. . 1
. . 1|8 . 6|4 . .
-----+-----+-----
. . 8|1 . 2|9 . .
7 . .|. . .|. . 8
. . 6|7 . 8|2 . .
-----+-----+-----
. . 2|6 . 9|5 . .
8 . .|2 . 3|. . 9
. . 5|. 1 .|3 . .
""".strip()


def test_cell_unknown() -> None:
    assert SudokuCell.unknown() == SudokuCell("123456789")


def test_cell_has_only() -> None:
    cell = SudokuCell("12")
    assert not cell.has_only("1")

    cell = SudokuCell("1")
    assert cell.has_only("1")


def test_cell_is_empty() -> None:
    cell = SudokuCell("123456789")
    assert not cell.is_empty()

    cell = SudokuCell("1")
    assert not cell.is_empty()

    cell = SudokuCell("")
    assert cell.is_empty()


def test_cell_drop() -> None:
    cell = SudokuCell("123456789")
    cell.drop("1")
    cell.drop("3")
    assert cell == SudokuCell("2456789")


def test_copy() -> None:
    puzzle = Sudoku.empty()
    grid = puzzle.grid
    grid_copy = puzzle.copy_grid(grid)
    assert grid is not grid_copy
    assert grid == grid_copy
    for key in grid:
        assert grid[key] is not grid_copy[key]


def test_parsing_from_invalid_length_string() -> None:
    input_str = "123456789"
    with pytest.raises(InvalidInput):
        Sudoku.from_string(input_str)


def test_get_units() -> None:
    puzzle = Sudoku.empty()
    units = puzzle.get_units("C2")
    assert units == (
        ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"),
        ("A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2", "I2"),
        ("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"),
    )


def test_empty_sudoku_unsolvability() -> None:
    puzzle = Sudoku.from_string(
        "..............................................................................111"
    )
    with pytest.raises(UnsolvableSudoku):
        puzzle.solve()


def test_get_peers() -> None:
    puzzle = Sudoku.empty()
    peers = puzzle.get_peers("C2")
    assert peers == {
        "A1",
        "A2",
        "A3",
        "B1",
        "B2",
        "B3",
        "C1",
        "C3",
        "D2",
        "E2",
        "F2",
        "G2",
        "H2",
        "I2",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
    }


def test_parse_from_string() -> None:
    input_str = "..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3.."
    puzzle = Sudoku.from_string(input_str)
    assert puzzle.grid["A1"] == SudokuCell("123456789")
    assert puzzle.grid["A3"] == SudokuCell("3")


def test_parsing_and_drawing() -> None:
    input_str = "..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3.."
    puzzle = Sudoku.from_string(input_str)
    displayed = puzzle.cli_display()
    for expected, actual in zip(CLI_DISPLAY.splitlines(), displayed.splitlines()):
        assert expected.strip() == actual.strip()
