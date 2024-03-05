import shutil
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from trapi_predict_kit import load, save

hyper_params = {"n_jobs": 2, "random_state": 42}
data, y = load_iris(return_X_y=True, as_frame=True)
model = RandomForestClassifier(
    n_jobs=hyper_params["n_jobs"],
    random_state=hyper_params["random_state"],
)
model.fit(data, y)
scores = {
    "precision": 0.85,
    "recall": 0.80,
    "accuracy": 0.85,
    "roc_auc": 0.90,
    "f1": 0.75,
    "average_precision": 0.85,
}
tmp_path = "tests/tmp"
model_path = "tests/tmp/model_test"


def test_save_pickle():
    """Test to save and load a basic model with pickle"""
    save(
        model,
        model_path,
        sample_data=data,
        scores=scores,
        hyper_params=hyper_params,
    )
    assert Path(model_path).is_file()
    assert Path(f"{model_path}.ttl").is_file()
    loaded_model = load(model_path)
    assert loaded_model.model is not None
    shutil.rmtree(tmp_path)


def test_save_mlem():
    """Test to save and load a basic model with mlem"""
    save(model, model_path, sample_data=data, scores=scores, hyper_params=hyper_params, method="mlem")
    assert Path(model_path).is_file()
    assert Path(f"{model_path}.mlem").is_file()
    assert Path(f"{model_path}.ttl").is_file()
    loaded_model = load(model_path, method="mlem")
    assert loaded_model.model is not None
    shutil.rmtree(tmp_path)
