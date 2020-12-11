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
@click.option('--debug', is_flag=True, help="Run in development mode with debugger enabled.")
@click.option('--start-spark/--no-spark', default=True, help="Start local Spark cluster (default to yes).")
def start_api(port, debug, start_spark):
    start_openpredict_api(port, debug, start_spark)

# TODO: update this call to make it "addEmbeddings?"
@click.command()
@click.option('--model', default='openpredict-baseline-omim-drugbank', help="Build the features from scratch (default to yes).")
def train_model(model):
    print ('Using model: ', model)
    model_features = retrieve_features('All').keys()
    clf, scores, hyper_params = train_model(model)
    add_run_metadata(scores, model_features, hyper_params)


@click.group()
def main(args=None):
    """Command Line Interface to run OpenPredict"""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

main.add_command(start_api)
main.add_command(train_model)

if __name__ == "__main__":
    sys.exit(main())
