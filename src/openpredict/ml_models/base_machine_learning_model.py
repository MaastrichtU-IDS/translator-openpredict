import os
from pathlib import Path
from typing import List

from openpredict.config import settings
from pydantic import BaseModel


class TrapiRelation(BaseModel):
    subject: str
    predicate: str
    object: str


class BaseMachineLearningModel():

    _folder_path: str = settings.OPENPREDICT_DATA_DIR
    
    trapi_relations: List[TrapiRelation] = []

    def __init__(self, train: bool = False):
        print("Check if model already present")
        if os.path.exists(f"{self._folder_path}"):
            print("✅ Model already present")
        else:
            print("Otherwise: try to download pretrained model")
            try:
                self.download()
            except:
                train = True

        if train:
            print("Or train the model (also if download fail/not possible)")


    def download(self):
        print(f"📥️ Download and unzip pretrained model in {self._folder_path}")
        Path(f"{self._folder_path}").mkdir(parents=True, exist_ok=True)


    def train(self):
        print("All the code for training the model from scratch")


    def predict(self):
        print("All the code for getting predictions from the model")
        print("This function will be the main function called by the API")


    def __str__(self):
        json = {
            'folder_path': self._folder_path
        } 
        return str(json)

