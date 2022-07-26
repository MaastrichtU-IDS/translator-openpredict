import click
import logging
import sys
import uvicorn

# from openpredict.openpredict_api import start_api as start_openpredict_api
from openpredict.openpredict_model import train_model as train_openpredict_model
from openpredict.rdf_utils import add_run_metadata, retrieve_features
from openpredict.main import app

@click.command()
@click.option(
    '-p', '--port', default=8808,
    help='Choose the port the API will run on.')
@click.option('--debug', is_flag=True, help="Run in development mode with debugger enabled.")
@click.option('--start-spark/--no-spark', default=True, help="Start local Spark cluster (default to yes).")
def start_api(port, debug, start_spark):
    uvicorn.run(app, host="0.0.0.0", port=port, reload=debug)
    # start_openpredict_api(port, debug, start_spark)

# TODO: update this call to make it "addEmbeddings?"
@click.command()
@click.option('--model', default='openpredict-baseline-omim-drugbank', help="Build the features from scratch (default to yes).")
def train_model(model):
    print('Using model: ', model)
    model_features = retrieve_features('All').keys()
    clf, scores, hyper_params, features_df = train_openpredict_model(model)
    # add_run_metadata(scores, model_features, hyper_params)


# @click.command()
# @click.option('-mesh', help="Path to MeSH annotations")
# @click.option('-hpo', help="Path to HPO annotations")
# @click.option('-di', help="Path to drug indications")
# def generate_features(mesh, hpo, di):
#     print('Generating features for: ', mesh, hpo, di)


@click.group()
def main(args=None):
    """Command Line Interface to run OpenPredict"""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

main.add_command(start_api)
main.add_command(train_model)
# main.add_command(generate_features)

if __name__ == "__main__":
    sys.exit(main())
