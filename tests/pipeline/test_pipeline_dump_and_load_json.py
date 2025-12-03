import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from tubular.capping import CappingTransformer
from tubular.imputers import MeanImputer, MedianImputer
from tubular.pipeline import dump_pipeline_to_json, load_pipeline_from_json


class TestDumpLoad:
    """Tests for dump_pipeline_to_json() and load_pipeline_from_json()."""

    def test_dump_pipeline_then_load_pipeline(self):
        df = pl.DataFrame({"a": [1, 5], "b": [10, 20]})

        median_imputer = MedianImputer(columns=["b"])
        mean_imputer = MeanImputer(columns=["b"])

        original_pipeline = Pipeline(
            [("MedianImputer", median_imputer), ("MeanImputer", mean_imputer)]
        )

        original_pipeline.fit(df, df["a"])

        pipeline_json = dump_pipeline_to_json(original_pipeline)

        pipeline = load_pipeline_from_json(pipeline_json)

        assert len(original_pipeline.steps) == len(pipeline.steps), (
            f"number of steps in the pipeline does not match with that of original pipeline, expected {len(original_pipeline.steps)} steps but got {len(pipeline.steps)}"
        )
        i = 0
        for x, y in zip(original_pipeline.steps, pipeline.steps):
            i = i + 1
            assert x[0] == y[0], (
                f"loaded pipeline does not match the original pipeline at step {i}, expected step name {x[0]} but got {y[0]}"
            )

            x1 = x[1].__dict__
            x1.pop("built_from_json", None)
            y1 = y[1].__dict__
            y1.pop("built_from_json", None)
            assert x1 == y1, (
                f"loaded pipeline does not match the original pipeline at step {i}, expected step name {x1} but got {y1}"
            )

    def test_dump_pipeline_to_json_output(self):
        df = pl.DataFrame({"a": [1, 5], "b": [10, 20]})

        median_imputer = MedianImputer(columns=["b"])
        mean_imputer = MeanImputer(columns=["b"])

        original_pipeline = Pipeline(
            [("MedianImputer", median_imputer), ("MeanImputer", mean_imputer)]
        )

        original_pipeline.fit(df, df["a"])

        actual_json = dump_pipeline_to_json(original_pipeline)

        expected_json = {
            "MeanImputer": {
                "classname": "MeanImputer",
                "fit": {"impute_values_": {"b": 15.0}},
                "init": {
                    "columns": ["b"],
                    "copy": False,
                    "return_native": True,
                    "verbose": False,
                    "weights_column": None,
                },
                "tubular_version": "...",
            },
            "MedianImputer": {
                "classname": "MedianImputer",
                "fit": {"impute_values_": {"b": 15.0}},
                "init": {
                    "columns": ["b"],
                    "copy": False,
                    "return_native": True,
                    "verbose": False,
                    "weights_column": None,
                },
                "tubular_version": "...",
            },
        }
        transformers = ["MedianImputer", "MeanImputer"]

        i = 0
        for transformer in transformers:
            i = i + 1
            assert (
                actual_json[transformer]["classname"]
                == expected_json[transformer]["classname"]
            ), (
                f"loaded json pipeline does not match the original pipeline at step {i}, expected step name {expected_json[transformer]['classname']} but got {actual_json[transformer]['classname']}"
            )
            assert (
                actual_json[transformer]["init"] == expected_json[transformer]["init"]
            ), (
                f"loaded json pipeline does not match the original pipeline at step {i}, expected step name {expected_json[transformer]['init']} but got {actual_json[transformer]['init']}"
            )
            assert (
                actual_json[transformer]["fit"] == expected_json[transformer]["fit"]
            ), (
                f"loaded json pipeline does not match the original pipeline at step {i}, expected step name {expected_json[transformer]['fit']} but got {actual_json[transformer]['fit']}"
            )

    def test_dump_transformer_not_jsonable(self):
        df = pl.DataFrame({"a": [1, 5], "b": [10, 20]})
        capping_transformer = CappingTransformer(capping_values={"b": [0, 15]})

        original_pipeline = Pipeline([("CappingTransformer", capping_transformer)])

        original_pipeline.fit(df, df["a"])
        with pytest.raises(
            RuntimeError,
        ):
            dump_pipeline_to_json(original_pipeline)
