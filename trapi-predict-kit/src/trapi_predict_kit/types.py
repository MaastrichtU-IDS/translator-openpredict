from typing import List, Optional

from pydantic import BaseModel


class PredictHit(BaseModel):
    subject: str
    object: str
    score: float
    subject_type: Optional[str]
    object_type: Optional[str]
    label: Optional[str]
    subject_label: Optional[str]
    object_label: Optional[str]

    class Config:
        arbitrary_types_allowed = True


class PredictOutput(BaseModel):
    hits: List[PredictHit]
    count: int
    # input_id: str
    # input_type: str


class PredictOptions(BaseModel):
    model_id: Optional[str] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    n_results: Optional[int] = None
    # types: Optional[List[str]] = None

    class Config:
        arbitrary_types_allowed = True


class PredictInput(BaseModel):
    subjects: List[str] = []
    objects: List[str] = []
    options: PredictOptions = PredictOptions()

    class Config:
        arbitrary_types_allowed = True


class TrainingOutput(BaseModel):
    # All scores are floats between 0 and 1
    precision: float
    recall: float
    accuracy: float
    roc_auc: float
    f1: float
    average_precision: float
    # elapsed_time: datetime

    class Config:
        arbitrary_types_allowed = True
