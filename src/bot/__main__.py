#!/usr/bin/env python

import click

from bot.app import app_factory


@click.command()
@click.option(
    "--token",
    help="Access Token for you telegram bot, provided by the BotFather.",
    required=True,
)
@click.option("--model-path", help="Path to your Keras persisted model.", required=True)
def run_bot(token, model_path):

    app = app_factory(token, model_path)
    app.run_polling()


if __name__ == "__main__":
    run_bot()
