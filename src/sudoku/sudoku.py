"""
Glossary:

Digit: A number from 1-9 used to feel-in squares.
Square: A cell in the sudoku grid, that can take a number from 1-9
Unit: A row, column, or 3x3 square, where the constraint of unique numbers applies to each unit.
Peer: A member of the unit
Box: A 3x3 block of cells, separated by thick lines (there are 99 of thems in classic sudoku)

Columns use 1-9 notation
Rows use A-I notation
"""

import pathlib
from typing import Dict, List, Optional, Self, Literal
from functools import cache
import numpy as np
import cv2

Digit = Literal["1", "2", "3", "4", "5", "6", "7", "8", "9"]
DigitSet = str
rows = "ABCDEFGHI"
cols = "123456789"
all_digits = "123456789"
SquarePosition = str


COLS_COUNT = 9
ROWS_COUNT = 9
TOTAL_SQUARES = COLS_COUNT * ROWS_COUNT
VALID_STR_DIGITS = set(map(str, range(1, 10)))
EMPTY_CELL = "."

# ['A1'..'I9']
SQUARE_POSITIONS: List[SquarePosition] = [r + c for r in rows for c in cols]


class SudokuCell:
    __slots__ = ("possible_values", "initially_provided")

    def __init__(self, possible_values: DigitSet, initially_provided=False) -> None:
        self.possible_values = possible_values
        self.initially_provided = initially_provided

    @classmethod
    def unknown(cls) -> Self:
        return cls(possible_values=all_digits[:])

    def copy(self) -> "SudokuCell":
        return SudokuCell(
            possible_values=self.possible_values[:],
            initially_provided=self.initially_provided,
        )

    def is_unknown(self) -> bool:
        return self.possible_values == all_digits

    def is_single(self) -> bool:
        return len(self.possible_values) == 1

    def has_only(self, value: Digit) -> bool:
        return self.is_single() and self.first() == value

    def is_empty(self) -> bool:
        return not bool(self.possible_values)

    def first(self) -> Digit:
        return self.possible_values[0]

    def drop(self, value: Digit) -> None:
        self.possible_values = self.possible_values.replace(str(value), "")

    def can_be(self, value: Digit) -> bool:
        return value in self.possible_values

    def __eq__(self, __value: object) -> bool:
        return (
            isinstance(__value, SudokuCell)
            and self.possible_values == __value.possible_values
        )

    def __repr__(self) -> str:
        return f"SudokuCell({self.possible_values})"


Grid = Dict[SquarePosition, SudokuCell]


class InvalidInput(Exception):
    pass


class UnsolvableSudoku(Exception):
    pass


class CLIRenderer:
    HL_COLOR = "red"
    SPACER = " "
    VERTICAL_DASH = "|"
    HORIZONTAL_DASH = "-"
    CORNER_DELIMITER = "+"

    def __init__(self, grid: Grid) -> None:
        self.grid = grid

    def render_value(self, cell: SudokuCell, highlight: bool = False) -> str:
        result = EMPTY_CELL
        if cell.is_unknown():
            result = EMPTY_CELL
        elif cell.is_single():
            result = str(cell.first())
        else:
            result = "[" + "".join(str(v) for v in sorted(cell.possible_values)) + "]"
        return f"-> {result} <=" if highlight else result

    def render_cell(
        self,
        row: str,
        column: str,
        max_width: int,
        highlighted_positions: tuple[str, ...] = (),
    ) -> str:
        # FIXME: doesn't work
        highlight = (row + column) in highlighted_positions
        value = self.render_value(self.grid[row + column], highlight=highlight)
        # Add a vertical dash after every 3rd and 6th column
        after_value = self.VERTICAL_DASH if column in ("3", "6") else self.SPACER
        return value + (" " * (max_width - len(value) - len(after_value))) + after_value

    def render_horizontal_line(self, max_width: int = 15) -> str:
        # Box underliner
        box_underliner = self.HORIZONTAL_DASH * (3 * max_width + 2)
        return "\n" + "+".join([box_underliner] * 3)

    def render_line(
        self,
        row: str,
        max_width: int,
        highlighted_positions: tuple[str, ...] = (),
    ) -> str:
        cells = [
            self.render_cell(
                row,
                col,
                max_width=max_width,
                highlighted_positions=highlighted_positions,
            )
            for col in cols
        ]
        # Add a horizontal dash after every 3rd row
        after_row = (
            self.render_horizontal_line(max_width=max_width)
            if row in ("C", "F")
            else ""
        )
        return "".join(cells) + after_row

    def render(
        self,
        title: Optional[str] = None,
        highlighted_positions: tuple[str, ...] = (),
    ) -> str:
        """
            Example of the output:\
. . 3 | . 2 . | 6 . .
9 . . | 3 . 5 | . . 1
. . 1 | 8 . 6 | 4 . .
------+-------+------
. . 8 | 1 . 2 | 9 . .
7 . . | . . . | . . 8
. . 6 | 7 . 8 | 2 . .
------+-------+------
. . 2 | 6 . 9 | 5 . .
8 . . | 2 . 3 | . . 9
. . 5 | . 1 . | 3 . .
        """
        max_cell_width = max(
            len(str(self.render_value(cell))) for cell in self.grid.values()
        )
        lines = []
        if title:
            lines.append(title)
        for row in rows:
            lines.append(
                self.render_line(
                    row,
                    highlighted_positions=highlighted_positions,
                    max_width=max_cell_width,
                )
            )

        return "\n".join(lines)


class ImageRenderer:
    FONT_SICKNESS = 2
    FONT_SCALE = 1
    INITIALLY_FILLED_COLOR = (0, 0, 0)
    GUESSED_BY_ALGO_COLOR = (0, 0, 255)

    def __init__(
        self,
        grid: Grid,
        output_path: pathlib.Path,
        image_size: int = 450,
        font=cv2.FONT_HERSHEY_SIMPLEX,
    ) -> None:
        self.grid = grid
        self.output_path = output_path
        self.image_size = image_size
        self.cell_size = image_size // 9
        self.font = font

    def digit_color(self, cell) -> tuple[int, int, int]:
        if cell.initially_provided:
            return self.INITIALLY_FILLED_COLOR
        return self.GUESSED_BY_ALGO_COLOR

    def row_col_index_to_position(self, row: int, col: int) -> SquarePosition:
        """
        e.g. (0, 0) -> "A1"
        """
        return rows[row] + cols[col]

    def render(self) -> None:
        board_img = (
            np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8) + 255
        )  # White background

        self.render_grid_lines(board_img)
        self.render_grid_numbers(board_img)
        self.store_image(board_img)

    def store_image(self, image) -> None:
        cv2.imwrite(str(self.output_path), image)

    def render_grid_lines(self, image) -> None:
        # Draw the grid lines and place the numbers
        for i in range(10):
            thickness = 2 if i % 3 == 0 else 1
            cv2.line(
                image,
                (0, i * self.cell_size),
                (self.image_size, i * self.cell_size),
                (0, 0, 0),
                thickness,
            )
            cv2.line(
                image,
                (i * self.cell_size, 0),
                (i * self.cell_size, self.image_size),
                (0, 0, 0),
                thickness,
            )

    def render_grid_numbers(self, image) -> None:
        for row in range(9):
            for col in range(9):
                cell = self.grid[self.row_col_index_to_position(row, col)]
                if cell.is_single():
                    number = cell.first()
                else:
                    number = None
                if number is None:
                    continue
                text_size = cv2.getTextSize(
                    str(number), self.font, self.FONT_SCALE, self.FONT_SICKNESS
                )[0]
                text_x = (col * self.cell_size) + (self.cell_size - text_size[0]) // 2
                text_y = (row * self.cell_size) + (self.cell_size + text_size[1]) // 2
                cv2.putText(
                    image,
                    str(number),
                    (text_x, text_y),
                    self.font,
                    self.FONT_SCALE,
                    self.digit_color(cell),
                    self.FONT_SICKNESS,
                )


class Sudoku:
    """
    Based entirely upon Peter Norvig's implementation:

    https://github.com/norvig/pytudes/blob/main/ipynb/Sudoku.ipynb
    """

    def __init__(self, start_grid: Grid) -> None:
        self.grid = start_grid

    @classmethod
    def empty(cls) -> Self:
        return cls({position: SudokuCell.unknown() for position in SQUARE_POSITIONS})

    def copy_grid(self, grid: Grid) -> Grid:
        return {position: cell.copy() for position, cell in grid.items()}

    @cache
    def get_units(
        self, position: SquarePosition
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """
        Returns the row, column, and box that the position belongs to
        """
        row, col = position[0], position[1]
        row_unit = tuple(row + c for c in cols)
        col_unit = tuple(r + col for r in rows)
        box_unit = tuple(
            r + c
            for r in rows[rows.index(row) // 3 * 3 : (rows.index(row) // 3 + 1) * 3]
            for c in cols[cols.index(col) // 3 * 3 : (cols.index(col) // 3 + 1) * 3]
        )
        return (row_unit, col_unit, box_unit)

    @cache
    def get_peers(self, position: SquarePosition) -> set[SquarePosition]:
        """
        Returns all the squares that are in the same row, column, or box as the position
        """
        return set().union(*self.get_units(position)) - {position}

    def constrain(self, grid: Grid) -> Grid:
        result = self.empty().grid
        for s in grid:
            if grid[s].is_single():
                self.fill(result, s, grid[s].first())
        # We need to pass the information about initially filled values
        # for the rendering purposes :)
        for s in grid:
            if grid[s].initially_provided:
                result[s].initially_provided = True
        return result

    def search(self, grid: Grid | None) -> Grid | None:
        if grid is None:
            return None
        # Find the cell with the fewest possible values, but more than one
        start_position = min(
            (position for position in grid if not grid[position].is_single()),
            key=lambda position: len(grid[position].possible_values),
            default=None,
        )

        if start_position is None:
            return grid

        for possible_value in grid[start_position].possible_values:
            solution = self.search(
                self.fill(self.copy_grid(grid), start_position, possible_value),
            )
            if solution:
                return solution

        return None

    def is_valid(self, grid: Grid) -> bool:
        for position in SQUARE_POSITIONS:
            cell = grid[position]
            if cell.is_empty():
                return False
            if cell.is_single():
                peer_positions = self.get_peers(position)
                if any(
                    cell.has_only(grid[peer_position].first())
                    for peer_position in peer_positions
                ):
                    return False
        return True

    def assert_valid(self, grid: Grid):
        assert self.is_valid(grid), "Invalid grid"

    def fill(
        self, grid: Grid, position: SquarePosition, value: Digit
    ) -> Optional[Grid]:
        cell: SudokuCell = grid[position]

        # We've already set
        if cell.has_only(value) or all(
            self.eliminate(grid, position, new_value)
            for new_value in cell.possible_values
            if new_value != value
        ):
            return grid
        return None

    def eliminate(
        self, grid: Grid, position: SquarePosition, value: Digit
    ) -> Optional[Grid]:
        cell: SudokuCell = grid[position]

        if not cell.can_be(value):
            return grid

        cell.drop(value)

        if cell.is_empty():
            return None

        if len(cell.possible_values) == 1:
            possible_value = cell.first()
            # Current cell has only one value.
            # It means it can be removed from all of the peers.
            if not all(
                self.eliminate(grid, peer_position, possible_value)
                for peer_position in self.get_peers(position)
            ):
                return None

        for unit in self.get_units(position):
            unit: tuple[SquarePosition, ...]
            # For each unit count how many time the digit appears
            placements = [
                u_pos for u_pos in unit if value in grid[u_pos].possible_values
            ]
            if not placements or (
                len(placements) == 1 and not self.fill(grid, placements[0], value)
            ):
                return None
        return grid

    def solve(self) -> str:
        solved = self.search(self.constrain(self.grid))
        if solved is None:
            raise UnsolvableSudoku("Sudoku is unsolvable")
        return solved

    @classmethod
    def from_string(cls, grid_string: str) -> Self:
        if len(grid_string) != TOTAL_SQUARES:
            raise InvalidInput("Grid string must be 81 characters long")
        grid: Grid = {}
        for char, square_position in zip(grid_string, SQUARE_POSITIONS):
            is_empty = char not in VALID_STR_DIGITS
            if is_empty:
                grid[square_position] = SudokuCell.unknown()
            else:
                grid[square_position] = SudokuCell(char, initially_provided=True)
        return cls(start_grid=grid)

    def cli_display(
        self,
        grid: Optional[Grid] = None,
        title: Optional[str] = None,
        highlighted_positions: tuple[str, ...] = (),
    ) -> str:
        if grid is None:
            grid = self.grid
        return CLIRenderer(grid=grid).render(
            title=title,
            highlighted_positions=highlighted_positions,
        )

    def render_to_image(
        self, image_path: pathlib.Path, grid: Optional[Grid] = None
    ) -> None:
        ImageRenderer(grid=grid or self.grid, output_path=image_path).render()


def two_grids(grid1, grid2):
    r1 = CLIRenderer(grid1)
    r2 = CLIRenderer(grid2)
    res1 = r1.render()
    res2 = r2.render()
    for left, right in zip(res1.splitlines(), res2.splitlines()):
        print(left + "   #   " + right)
