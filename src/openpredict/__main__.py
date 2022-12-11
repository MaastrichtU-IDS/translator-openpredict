import logging
import sys

import click

from openpredict.rdf_utils import retrieve_features
from openpredict_model.train import train_model as train_openpredict_model


# TODO: update this call to make it "addEmbeddings?"
@click.command()
@click.option('--model', default='openpredict-baseline-omim-drugbank', help="Build the features from scratch (default to yes).")
def train_model(model):
    print(f'Using model: {model}')
    retrieve_features('All').keys()
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


main.add_command(train_model)
# main.add_command(generate_features)

if __name__ == "__main__":
    sys.exit(main())
