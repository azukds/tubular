"""Module contains methods for serializing and deserializing pipelines."""

from copy import copy
from typing import Any

from sklearn.pipeline import Pipeline

from tubular.base import CLASS_REGISTRY


def dump_pipeline_to_json(pipeline: Pipeline) -> dict[str, dict[str, Any]]:
    """Serialize a pipeline into json dictionary.

    Parameters
    ----------
    pipeline: Pipeline
        sequence of transformer objects

    Returns
    -------
    dict
        json dictionary representing the pipeline.

    Raises
    ------
    RuntimeError
        If any of the transformer in pipeline is not jsonable it raises RuntimeError.

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
    ...     [("median_imputer", median_imputer), ("mean_imputer", mean_imputer)]
    ... )
    >>> original_pipeline = original_pipeline.fit(df, df["a"])
    >>> pipeline_json = dump_pipeline_to_json(original_pipeline)
    >>> pipeline_json  # doctest: +NORMALIZE_WHITESPACE
    {'median_imputer': {'tubular_version':...,
    'classname': 'MedianImputer',
    'init': {'columns': ['b'],
    'copy': False,
    'verbose': False,
    'return_native': True,
    'weights_column': None},
    'fit': {'is_fitted_': True, 'impute_values_': {'b': 15.0}}},
    'mean_imputer': {'tubular_version':...,
    'classname': 'MeanImputer',
    'init': {'columns': ['b'],
    'copy': False,
    'verbose': False,
    'return_native': True,
    'weights_column': None},
        'fit': {'is_fitted_': True, 'impute_values_': {'b': 15.0}}}}

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
    ...     [("median_imputer", median_imputer), ("mean_imputer", mean_imputer)]
    ... )

    >>> original_pipeline = original_pipeline.fit(df, df["a"])
    >>> pipeline_json = dump_pipeline_to_json(original_pipeline)
    >>> pipeline = load_pipeline_from_json(pipeline_json)
    >>> pipeline
    Pipeline(steps=[('median_imputer', MedianImputer(columns=['b'])),
                    ('mean_imputer', MeanImputer(columns=['b']))])

    ```

    """
    steps = [
        (step_name, CLASS_REGISTRY[json_dict["classname"]].from_json(json_dict))
        for step_name, json_dict in pipeline_json.items()
    ]

    return Pipeline(steps)


def filter_pipeline_for_features(pipeline: Pipeline, features: list[str]) -> Pipeline:
    """Filter down pipeline to just produce specified features.

    Useful to slim down feature selection pipeline to production
    pipeline, without having to rewrite.

    Parameters
    ----------
    pipeline: Pipeline
        pipeline to filter

    features:
        features to filter for

    Returns
    -------
    Filtered pipeline

    Examples
    --------
    ```pycon
    >>> from sklearn.pipeline import Pipeline
    >>> from tubular.numeric import DifferenceTransformer, RatioTransformer
    >>> from tubular.imputers import MeanImputer
    >>> import polars as pl
    >>> difference_transformer = DifferenceTransformer(columns=["a", "b"])
    >>> ratio_transformer = RatioTransformer(columns=["a", "b"])
    >>> imputer = MeanImputer(columns=["a_minus_b", "a_divided_by_b"])
    >>> pipeline = Pipeline(
    ...     [
    ...         ("difference", difference_transformer),
    ...         ("ratio", ratio_transformer),
    ...         ("imputer", imputer),
    ...     ]
    ... )
    >>> df = pl.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]})
    >>> pipeline.fit(df)
    Pipeline(steps=[('difference', DifferenceTransformer(columns=['a', 'b'])),
                    ('ratio', RatioTransformer(columns=['a', 'b'])),
                    ('imputer',
                     MeanImputer(columns=['a_minus_b', 'a_divided_by_b']))])
    >>> filter_pipeline_for_features(pipeline, ["a", "a_minus_b"])
    Pipeline(steps=[DifferenceTransformer(columns=[['a', 'b']]),
                    RatioTransformer(columns=['a', 'b']),
                    MeanImputer(columns=['a_minus_b'])])

    ```

    """
    reversed_pipeline = pipeline.steps[::-1]
    needed_steps = []
    needed_columns = copy(features)

    for _, step in reversed_pipeline:
        step_outputs = step.get_feature_names_out()
        step_lineage = step.get_features_out_lineage()

        # find outputs which overlap with needed columns
        needed_columns_overlap = set(step_outputs).intersection(needed_columns)

        if needed_columns_overlap:
            # find inputs needed for these outputs
            needed_inputs = []
            for output_column in needed_columns_overlap:
                needed_inputs = [*needed_inputs, *step_lineage[output_column]]

            # filter step to just produce needed inputs
            step.select(sorted(needed_columns_overlap))

        # add in new needed columns for next round
        needed_columns.append(set(needed_inputs))
        needed_columns = sorted(set(needed_columns))

        needed_steps.append(step)

    needed_steps.reverse()

    return Pipeline(needed_steps)
