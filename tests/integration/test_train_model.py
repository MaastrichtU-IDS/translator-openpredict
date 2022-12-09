import numpy as np
import pandas as pd

from openpredict_model.train import geometricMean, train_model


def test_train_model():
    """Test to train from baseline model to get drug-disease similarities (drugbank-omim)"""
    clf, scores, hyper_params, features_df = train_model()

    assert 0.80 < scores['precision'] < 0.95
    assert 0.60 < scores['recall'] < 0.85
    assert 0.80 < scores['accuracy'] < 0.95
    assert 0.85 < scores['roc_auc'] < 0.95
    assert 0.70 < scores['f1'] < 0.85
    assert 0.75 < scores['average_precision'] < 0.95

# def test_add_embeddings():
#     """Test add embeddings to the model and rebuild it"""
#     embeddings_filepath = str(pathlib.Path(__file__).parent.joinpath("../data/neurodkg_embedding.json"))
#     # JSON embeddings broken since tabular embeddings has been added

#     with open(embeddings_filepath,  encoding="utf8") as embeddings_file:
#         run_id, scores = addEmbedding(embeddings_file, 'test_embedding', 'Both', 'test embedding', 'openpredict-baseline-omim-drugbank')
#         assert path.exists(get_openpredict_dir('models/' + run_id + '.joblib'))
#         assert len(run_id) > 10
#         assert 0.80 < scores['precision'] < 0.95
#         assert 0.60 < scores['recall'] < 0.85
#         assert 0.80 < scores['accuracy'] < 0.95
#         assert 0.85 < scores['roc_auc'] < 0.95
#         assert 0.70 < scores['f1'] < 0.85
#         assert 0.75 < scores['average_precision'] < 0.95

def test_calculate_combined():
    """Test geometric mean, a measure for drug-disease similarities (drugbank-omim)"""
    disease = '206200'
    drug = 'DB00136'

    # di_feat_col ='HPO-SIM'
    # dr_feat_col ='SE-SIM'
    #diseaseDF= disease_df[di_feat_col]
    #drugDF = drug_df[dr_feat_col]

    drugDF= pd.DataFrame.from_dict({'DB00136': {'DB00136': 1.0, 'DB00286': 0.13522012578616352},
    'DB00286': {'DB00136': 0.13522012578616352, 'DB00286': 1.0}})

    data_dis = {'208085': {'208085': 1.0, '206200': 0.3738388048970476, '156000': 0.27540399660290193},
                '206200': {'208085': 0.3738388048970476, '206200': 1.0, '156000': 0.19287170205206816},
                '156000': {'208085': 0.27540399660290193, '206200': 0.19287170205206816,'156000': 1.0}}
    diseaseDF= pd.DataFrame.from_dict(data_dis, orient='index')


    knownDrugDisease = np.array([['DB00136','208085'],['DB00286','206200'],['DB00286','156000']])
    x1 = geometricMean(drug, disease, knownDrugDisease, drugDF, diseaseDF)
    print(x1,np.sqrt(0.373839))
    assert( np.isclose(x1,np.sqrt(0.373839), rtol=1e-05, atol=1e-08, equal_nan=False))

    disease = '206200'
    drug = 'DB00286'
    x2 = geometricMean(drug, disease, knownDrugDisease, drugDF, diseaseDF)
    print(x2, np.sqrt(0.192872))
    assert( np.isclose(x2, np.sqrt(0.192872), rtol=1e-05, atol=1e-08, equal_nan=False))
