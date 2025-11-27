"""Module contains methods for serializing and deserializing pipelines."""

from typing import Any

from sklearn.pipeline import Pipeline

from tubular.base import CLASS_REGISTRY


def dump(pipeline: Pipeline) -> dict[str, dict[str, Any]]:
    """Serialize a pipeline into json dictionary.

    Parameters
    ----------
    pipeline: Pipeline
        sequence of transformer objects

    Raises
    ------
    RuntimeError
        If any of the transformer in pipeline is not jsonable it raises RuntimeError.


    Returns
    -------
    dict
        json dictionary representing the pipeline.

    Examples
    --------
    >>> from profiling.pipeline_generator import TubularPipelineGenerator
    >>> from profiling import create_dataset as create
    >>> pipe = TubularPipelineGenerator()
    >>> pipe = pipe.generate_pipeline(["GroupRareLevelsTransformer"])
    >>> df_1 = create.create_standard_pandas_dataset()
    >>> a = pipe.fit(df_1, df_1["AveRooms"])
    >>> pipeline_json = dump(pipe)
    >>> pipeline_json
    {'GroupRareLevelsTransformer': {...}}

    """
    steps = pipeline.steps
    non_jsonable_steps = [step[0] for step in steps if step[1].jsonable is False]
    if non_jsonable_steps:
        msg = f"the following steps are not yet jsonable: {non_jsonable_steps}"
        raise RuntimeError(msg)

    return {step_name: step.to_json() for step_name, step in steps}


def load(pipeline_json: dict[str, dict[str, Any]]) -> Pipeline:
    """Deserialize a pipeline json structure into a pipeline.

    Parameters
    ----------
    pipeline_json: dict
        json dictionary representing the pipeline.

    Returns
    -------
    Pipeline loaded  from json dict

    Examples
    --------
    >>> from profiling.pipeline_generator import TubularPipelineGenerator
    >>> from profiling import create_dataset as create
    >>> pipe = TubularPipelineGenerator()
    >>> pipe = pipe.generate_pipeline(["GroupRareLevelsTransformer"])
    >>> df_1 = create.create_standard_pandas_dataset()
    >>> a = pipe.fit(df_1, df_1["AveRooms"])
    >>> pipeline_json = dump(pipe)
    >>> pipeline = load(pipeline_json)
    >>> pipeline
    Pipeline(steps=[('GroupRareLevelsTransformer',
                     GroupRareLevelsTransformer(columns=['categorical_4']))])

    """
    steps = [
        (step_name, CLASS_REGISTRY[step_name].from_json(json_dict))
        for step_name, json_dict in pipeline_json.items()
    ]

    return Pipeline(steps)
