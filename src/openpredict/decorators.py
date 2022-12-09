import functools
import inspect
from datetime import datetime
from typing import List

from openpredict.models.predict_output import PredictOptions, TrapiRelation


def trapi_predict(relations):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(input_id: str, options: PredictOptions):
            options = PredictOptions.parse_obj(options)
            # Add any additional logic or behavior here
            # print(f'Decorator parameter: {relations}')
            return func(input_id, options)
        return wrapper, relations
    return decorator



def trapi_predict_fairworkflow(
    relations: List[TrapiRelation] = [],
    **kwargs
):
    """Mark a function that returns predictions for a set of relation (spo)

    Args:
        label (str): Label of the fair step (corresponds to rdfs:label predicate)
        is_pplan_step (str): Denotes whether this step is a pplan:Step
        is_manual_task (str): Denotes whether this step is a bpmn.ManualTask
        is_script_task (str): Denotes whether this step is a bpmn.ScriptTask

    All additional arguments are expected to correspond to input parameters of the decorated
    function, and are used to provide extra semantic types for that parameter. For example,
    consider the following decorated function:
        @is_fairstep(label='Addition', a='http://www.example.org/number', out='http://www.example.org/float')
        def add(a:float, b:float) -> float:
            return a + b
    1. Note that using 'a' as parameter to the decorator allows the user to provide a URI for a semantic type
    that should be associated with the function's input parameter, 'a'. This can be either a string, an
    rdflib.URIRef, or a list of these.
    2. Note that the return parameter is referred to using 'returns', because it does not otherwise have a name.
    In this case, the function only returns one value. However, if e.g. a tuple of 3 values were returned,
    you could use a tuple for 'returns' in the decorator arguments too. For example:
        out=('http://www.example.org/mass', 'http://www.example.org/distance')
    This would set the semantic type of the first return value as some 'mass' URI, and the second
    return value as 'distance'. Lists can also be provided instead of a single URI, if more than one
    semantic type should be associated with a given output. Any element of this tuple can also be
    set to None, if no semantic type is desired for it.
    3. The return parameter name (by default 'returns') can be changed if necessary, by modifying
    the IS_FAIRSTEP_RETURN_VALUE_PARAMETER_NAME constant.
    """

    def _modify_function(func):
        """
        Store FairStep object as _fairstep attribute of the function. Use inspection to get the
        description, inputs, and outputs of the step based on the function specification.

        Returns this function decorated with the noodles schedule decorator.
        """
        # Description of step is the raw function code
        description = inspect.getsource(func)
        inputs = _extract_inputs_from_function(func, kwargs)
        outputs = _extract_outputs_from_function(func, kwargs)

        fairstep = FairStep(uri='http://www.example.org/unpublished-'+func.__name__,
                            label=label,
                            description=description,
                            is_pplan_step=is_pplan_step,
                            is_manual_task=is_manual_task,
                            is_script_task=is_script_task,
                            language=LINGSYS_PYTHON,
                            inputs=inputs,
                            outputs=outputs)

        def _add_logging(func):
            @functools.wraps(func)
            def _wrapper(*func_args, **func_kwargs):

                # Get the arg label/value pairs as a dict (for both args and kwargs)
                func_args_dict = dict(zip(inspect.getfullargspec(func).args, func_args))
                all_args = {**func_args_dict, **func_kwargs}

                # Execute step (with timing)
                t0 = datetime.now()
                if is_manual_task:
                    execution_result = manual_assistant.execute_manual_step(fairstep)
                else:
                    execution_result = func(*func_args, **func_kwargs)
                t1 = datetime.now()

                # Log step execution
                prov_logger.add(StepRetroProv(step=fairstep, step_args=all_args, output=execution_result, time_start=t0, time_end=t1))

                return execution_result

            return _wrapper
        func._fairstep = fairstep
        return noodles.schedule(_add_logging(func))

    return _modify_function
