import click
import logging
import sys

from openpredict.openpredict_api import start_api as start_openpredict_api
from openpredict.openpredict_model import train_model
from openpredict.rdf_utils import add_run_metadata, retrieve_features

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
@click.option('--from-scratch/--no-scratch', default=True, help="Build the features from scratch (default to yes).")
def train_model(from_scratch):
    print ('from_scratch',from_scratch)
    model_features = retrieve_features('All').keys()
    clf, scores, hyper_params = train_model(from_scratch)
    add_run_metadata(scores, model_features, hyper_params)


@click.group()
def main(args=None):
    """Command Line Interface to run OpenPredict"""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

main.add_command(start_api)
main.add_command(train_model)

if __name__ == "__main__":
    sys.exit(main())
