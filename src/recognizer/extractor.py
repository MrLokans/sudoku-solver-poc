import cv2
import numpy as np

import pathlib

from recognizer.recognizers import DigitRecognizer


class SudokuExtractor:
    def __init__(self, image_path: pathlib.Path, digit_recognizer: DigitRecognizer):
        self.image_path = image_path
        self.digit_recognizer = digit_recognizer

    def process(self) -> str:
        sudoku_image = self.process_sudoku_image()
        # FIXME: This is a temporary solution :)))
        cv2.imwrite("sudoku_transformed.jpg", sudoku_image)
        sudoku_cells = self.extract_cells("sudoku_transformed.jpg")
        sudoku_line = self.to_sudoku_line(sudoku_cells)
        return sudoku_line

    def process_sudoku_image(self):
        image = cv2.imread(self.image_path.absolute().as_posix())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200, 255)

        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        puzzleContour = None

        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

            if len(approx) == 4:
                puzzleContour = approx
                break

        if puzzleContour is None:
            raise Exception("Could not find Sudoku puzzle outline.")

        warped = self.__four_point_transform(gray, puzzleContour.reshape(4, 2))

        return warped

    def __order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def __four_point_transform(self, image, pts):
        rect = self.__order_points(pts)

        maxWidth = 1000
        maxHeight = 1000

        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def to_sudoku_line(self, cells):
        sudoku_line = ""
        for cell in cells:
            recognized = self.digit_recognizer.recognize(cell)
            sudoku_line += str(recognized) if recognized != -1 else "."
        return sudoku_line

    def extract_cells(self, squared_image_path: pathlib.Path, cell_size=50):
        # Read the image in grayscale
        gray_image = cv2.imread(squared_image_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image was loaded correctly
        if gray_image is None:
            raise ValueError("Could not load the image, check the path.")

        # Assuming the whole image is a Sudoku grid, find the size of the grid
        grid_size = gray_image.shape[0]  # Assuming the image is a square

        # Find the size of each cell
        cell_height = grid_size // 9
        cell_width = grid_size // 9

        # Initialize an array to hold the output cells
        cells = np.zeros((81, cell_size, cell_size), dtype=np.uint8)

        # Iterate over the grid and extract cells
        for i in range(9):  # For each row
            for j in range(9):  # For each column
                # Calculate the starting point of the current cell
                y = i * cell_height
                x = j * cell_width

                # Extract the cell using slicing
                cell = gray_image[y : y + cell_height, x : x + cell_width]

                # Resize the cell to the desired size (28x28)
                cell_resized = cv2.resize(
                    cell, (cell_size, cell_size), interpolation=cv2.INTER_AREA
                )

                # Store the resized cell in the cells array
                cells[i * 9 + j] = cell_resized

        return cells
