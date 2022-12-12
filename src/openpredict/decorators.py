import functools
from typing import List, Optional

from openpredict.predict_output import PredictOptions, TrapiRelation


def trapi_predict(
    path: str,
    relations: List[TrapiRelation],
    name: Optional[str] =None,
    description: Optional[str] = "",
    default_input: Optional[str] = "DRUGBANK:DB00394",
    default_model: Optional[str] = "openpredict-baseline-omim-drugbank",
):
    """ A decorator to indicate a function is a function to generate prediction that can be integrated to TRAPI.
    The function needs to take an input_id and returns a list of predicted entities related to the input entity
    """
    if not name:
        name = path
    def decorator(func):
        @functools.wraps(func)
        def wrapper(input_id: str, options: PredictOptions):
            options = PredictOptions.parse_obj(options)
            # Add any additional logic or behavior here
            # print(f'Decorator parameter: {relations}')
            return func(input_id, options)
        return wrapper, {
            'relations': relations,
            'path': path,
            'name': name,
            'description': description,
            'default_input': default_input,
            'default_model': default_model,
        }
    return decorator
