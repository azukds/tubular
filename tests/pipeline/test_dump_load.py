import pytest

from profiling import create_dataset as create
from profiling.pipeline_generator import TubularPipelineGenerator
from tubular.pipeline import dump, load


class TestDumpLoad:
    """Tests for dump() and load()."""

    def test_dump_then_load(self):
        original_pipeline = TubularPipelineGenerator()
        original_pipeline = original_pipeline.generate_pipeline(
            [
                "ArbitraryImputer",
                "MedianImputer",
                "MeanImputer",
                "ModeImputer",
                "GroupRareLevelsTransformer",
            ]
        )

        df_1 = create.create_standard_pandas_dataset()

        original_pipeline.fit(df_1, df_1["AveRooms"])

        pipeline_json = dump(original_pipeline)

        pipeline = load(pipeline_json)

        assert len(original_pipeline.steps) == len(pipeline.steps)

        for x, y in zip(original_pipeline.steps, pipeline.steps):
            assert x[0] == y[0]

            x1 = x[1].__dict__
            x1.pop("built_from_json", None)
            y1 = y[1].__dict__
            y1.pop("built_from_json", None)
            assert x1 == y1

    def test_dump_output(self):
        original_pipeline = TubularPipelineGenerator()
        transformer = "MedianImputer"
        original_pipeline = original_pipeline.generate_pipeline([transformer])

        df_1 = create.create_standard_pandas_dataset()

        original_pipeline.fit(df_1, df_1["AveRooms"])

        actual_json = dump(original_pipeline)

        expected_json = {
            "MedianImputer": {
                "tubular_version": "dev",
                "classname": "MedianImputer",
                "init": {
                    "columns": ["HouseAge_2", "AveOccup_2", "Population_2"],
                    "copy": False,
                    "verbose": False,
                    "return_native": True,
                    "weights_column": None,
                },
                "fit": {
                    "impute_values_": {
                        "AveOccup_2": 2.818923991761345,
                        "HouseAge_2": 29.0,
                        "Population_2": 1166.0,
                    }
                },
            }
        }

        assert (
            actual_json[transformer]["classname"]
            == expected_json[transformer]["classname"]
        )
        assert actual_json[transformer]["init"] == expected_json[transformer]["init"]
        assert actual_json[transformer]["fit"] == expected_json[transformer]["fit"]

    def test_dump_transformer_not_jsonable(self):
        original_pipeline = TubularPipelineGenerator()
        original_pipeline = original_pipeline.generate_pipeline(["LogTransformer"])

        df_1 = create.create_standard_pandas_dataset()

        original_pipeline.fit(df_1, df_1["AveRooms"])
        with pytest.raises(
            RuntimeError,
        ):
            dump(original_pipeline)
