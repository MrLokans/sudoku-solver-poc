import pathlib

from telegram.ext import (
    ApplicationBuilder,
    filters,
)

import tensorflow as tf


from bot.handlers import PhotoHandler


def app_factory(token: str, model_path: pathlib.Path):
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    app = ApplicationBuilder().token(token).build()
    app.add_handler(PhotoHandler(model, filters.PHOTO))
    return app
