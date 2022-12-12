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
    input: str


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


# class TrainingOutput(BaseModel):
#     score: float
