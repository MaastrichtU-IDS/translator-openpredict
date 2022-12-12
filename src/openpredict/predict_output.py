from typing import List, Optional

from pydantic import BaseModel


class PredictHit(BaseModel):
    id: str
    type: str
    score: float
    label: Optional[str]


class PredictOutput(BaseModel):
    hits: List[PredictHit]
    count: int
    # input_id: str
    # input_type: str


class PredictOptions(BaseModel):
    model_id: Optional[str] = 'openpredict-baseline-omim-drugbank'
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    n_results: Optional[int] = None
    types: Optional[List[str]] = None

    class Config:
        arbitrary_types_allowed = True


class TrapiRelation(BaseModel):
    subject: str
    predicate: str
    object: str


class TrainingOutput(BaseModel):
    # All scores are floats between 0 and 1
    precision: float
    recall: float
    accuracy: float
    roc_auc: float
    f1: float
    average_precision: float
    # elapsed_time: datetime
