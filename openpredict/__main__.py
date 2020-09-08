import click
import logging
import sys

from openpredict.openpredict_api import start_api as start_openpredict_api
from openpredict.openpredict_omim_drugbank import train_drug_disease_classifier

@click.command()
@click.option(
    '-p', '--port', default=8808,
    help='Choose the port the API will run on.')
@click.option(
    '--server-url', default='/',
    help='The URL the API will be deployed to.')
@click.option('--debug', is_flag=True, help="Run in development mode with debugger enabled.")
@click.option('--start-spark/--no-spark', default=True, help="Start local Spark cluster (default to yes).")
def start_api(port, server_url, debug, start_spark):
    start_openpredict_api(port, server_url, debug, start_spark)


@click.command()
def train_model():
    train_drug_disease_classifier()


@click.group()
def main(args=None):
    """Command Line Interface to run OpenPredict"""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

main.add_command(start_api)
main.add_command(train_model)

if __name__ == "__main__":
    sys.exit(main())
