import click
import logging
import sys

from openpredict.openpredict_api import start_api as start_openpredict_api
from openpredict.openpredict_omim_drugbank import get_drug_disease_classifier
from openpredict.feature_generation import generate_feature

@click.command()
@click.option(
    '-p', '--port', default=8808,
    help='Choose the port the API will run on.')
@click.option('--debug', is_flag=True, help="Run in development mode with debugger enabled.")
def start_api(port, debug):
    start_openpredict_api(port, debug)


@click.command()
def build_models():
    get_drug_disease_classifier()

@click.command()
def generate_features():
    generate_feature()


@click.group()
def main(args=None):
    """Command Line Interface to run OpenPredict"""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

main.add_command(start_api)
main.add_command(build_models)
main.add_command(generate_features)


if __name__ == "__main__":
    sys.exit(main())
