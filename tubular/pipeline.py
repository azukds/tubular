"""Module contains methods for serializing and deserializing pipelines."""

from typing import Any

from sklearn.pipeline import Pipeline

from tubular.base import CLASS_REGISTRY


def dump_pipeline_to_json(pipeline: Pipeline) -> dict[str, dict[str, Any]]:
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
    ```pycon
    >>> import polars as pl
    >>> from tubular.imputers import MeanImputer, MedianImputer
    >>> from sklearn.pipeline import Pipeline

    >>> df = pl.DataFrame({"a": [1, 5], "b": [10, 20]})
    >>> median_imputer = MedianImputer(columns=["b"])
    >>> mean_imputer = MeanImputer(columns=["b"])
    >>> original_pipeline = Pipeline(
    ...     [("MedianImputer", median_imputer), ("MeanImputer", mean_imputer)]
    ... )
    >>> original_pipeline = original_pipeline.fit(df, df["a"])
    >>> pipeline_json = dump_pipeline_to_json(original_pipeline)
    >>> pipeline_json  # doctest: +NORMALIZE_WHITESPACE
    {'MedianImputer': {'tubular_version':...,
    'classname': 'MedianImputer',
    'init': {'columns': ['b'],
    'copy': False,
    'verbose': False,
    'return_native': True,
    'weights_column': None},
    'fit': {'impute_values_': {'b': 15.0}}},
    'MeanImputer': {'tubular_version':...,
    'classname': 'MeanImputer',
    'init': {'columns': ['b'],
    'copy': False,
    'verbose': False,
    'return_native': True,
    'weights_column': None},
        'fit': {'impute_values_': {'b': 15.0}}}}

    ```

    """
    steps = pipeline.steps
    non_jsonable_steps = [step[0] for step in steps if step[1].jsonable is False]
    if non_jsonable_steps:
        msg = f"the following steps are not yet jsonable: {non_jsonable_steps}"
        raise RuntimeError(msg)

    return {step_name: step.to_json() for step_name, step in steps}


def load_pipeline_from_json(pipeline_json: dict[str, dict[str, Any]]) -> Pipeline:
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
    ```pycon
    >>> import polars as pl
    >>> from tubular.imputers import MeanImputer, MedianImputer
    >>> from sklearn.pipeline import Pipeline
    >>> df = pl.DataFrame({"a": [1, 5], "b": [10, 20]})
    >>> median_imputer = MedianImputer(columns=["b"])
    >>> mean_imputer = MeanImputer(columns=["b"])
    >>> original_pipeline = Pipeline(
    ...     [("MedianImputer", median_imputer), ("MeanImputer", mean_imputer)]
    ... )

    >>> original_pipeline = original_pipeline.fit(df, df["a"])
    >>> pipeline_json = dump_pipeline_to_json(original_pipeline)
    >>> pipeline = load_pipeline_from_json(pipeline_json)
    >>> pipeline
    Pipeline(steps=[('MedianImputer', MedianImputer(columns=['b'])),
                    ('MeanImputer', MeanImputer(columns=['b']))])

    ```

    """
    steps = [
        (step_name, CLASS_REGISTRY[step_name].from_json(json_dict))
        for step_name, json_dict in pipeline_json.items()
    ]

    return Pipeline(steps)
