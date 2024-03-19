import pathlib
from telegram import Update
from telegram.ext import (
    CallbackContext,
)
from telegram.ext import (
    CallbackContext,
    MessageHandler,
    filters,
)

from recognizer.extractor import SudokuExtractor
from recognizer.recognizers import KerasRecognizer
from sudoku.sudoku import Sudoku


class PhotoHandler(MessageHandler):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(filters.PHOTO, self.handle, *args, **kwargs)

    async def handle(self, update: Update, context: CallbackContext):
        new_file = await update.message.effective_attachment[-1].get_file()
        image_path = pathlib.Path("bot-sudoku.jpg")
        # FIXME: c'mon, at least use temp files
        await new_file.download_to_drive(image_path)

        extractor = SudokuExtractor(image_path, KerasRecognizer(model=self.model))
        sudoku_line = extractor.process()
        puzzle = Sudoku.from_string(sudoku_line)
        try:
            solution = puzzle.solve()
            puzzle.render_to_image("solved-sudoku.jpg", grid=solution)
            with open("solved-sudoku.jpg", "rb") as f:
                await context.bot.send_photo(chat_id=update.message.chat_id, photo=f)
        except Exception:
            await context.bot.send_message(
                chat_id=update.message.chat_id,
                text="I did not manage to parse or solve your puzzle :(",
            )
