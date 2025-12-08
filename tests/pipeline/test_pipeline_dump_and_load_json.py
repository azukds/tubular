import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from tubular.capping import CappingTransformer
from tubular.imputers import MeanImputer, MedianImputer
from tubular.pipeline import dump_pipeline_to_json, load_pipeline_from_json


class TestPipelineDumpAndLoadJson:
    """Tests for dump_pipeline_to_json() and load_pipeline_from_json()."""

    def test_dump_pipeline_then_load_pipeline(self):    # noqa: PLR6301
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

        for i, (x, y) in enumerate(zip(original_pipeline.steps, pipeline.steps)):
            assert x[0] == y[0], (
                f"loaded pipeline does not match the original pipeline at step {i}, expected step name {x[0]} but got {y[0]}"
            )

            x1 = x[1].__dict__
            x1.pop("built_from_json", None)
            y1 = y[1].__dict__
            y1.pop("built_from_json", None)
            assert x1 == y1, (
                f"loaded pipeline does not match the original pipeline at step {i}, expected step {x1} but got {y1}"
            )

    def test_dump_pipeline_to_json_output(self):    # noqa: PLR6301
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
                "fit": {"impute_values_": mean_imputer.impute_values_},
                "init": {
                    "columns": mean_imputer.columns,
                    "copy": mean_imputer.copy,
                    "return_native": mean_imputer.return_native,
                    "verbose": mean_imputer.verbose,
                    "weights_column": mean_imputer.weights_column,
                },
                "tubular_version": "...",
            },
            "MedianImputer": {
                "classname": "MedianImputer",
                "fit": {"impute_values_": median_imputer.impute_values_},
                "init": {
                    "columns": median_imputer.columns,
                    "copy": median_imputer.copy,
                    "return_native": median_imputer.return_native,
                    "verbose": median_imputer.verbose,
                    "weights_column": median_imputer.weights_column,
                },
                "tubular_version": "...",
            },
        }
        transformers = ["MedianImputer", "MeanImputer"]

        for i, transformer in enumerate(transformers):
            assert actual_json[transformer] == expected_json[transformer], (
                f"loaded json pipeline does not match the original pipeline at step {i}, expected step {expected_json[transformer]} but got {actual_json[transformer]}"
            )

    def test_dump_transformer_not_jsonable(self):   # noqa: PLR6301
        df = pl.DataFrame({"a": [1, 5], "b": [10, 20]})
        capping_transformer = CappingTransformer(capping_values={"b": [0, 15]})

        original_pipeline = Pipeline([("CappingTransformer", capping_transformer)])

        original_pipeline.fit(df, df["a"])
        with pytest.raises(
            RuntimeError,
        ):
            dump_pipeline_to_json(original_pipeline)
