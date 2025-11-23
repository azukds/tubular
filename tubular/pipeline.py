"""Module contains utility class for serializing and deserializing pipelines."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from tubular.base import CLASS_REGISTRY, BaseTransformer


class PipelineSerializer:
    """Utility class for serializing and deserializing pipelines.

    Provides dump and load methods for pipeline.

    """

    @classmethod
    def dump(cls, steps: list[tuple[str, BaseTransformer]]) -> str | dict[str, dict]:
        """Serialize a sequence of pipeline steps into a json dictionary.

        Parameters
        ----------
        steps : list
            sequence of pipeline steps

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
        >>> pipeline_json = PipelineSerializer.dump(pipe.steps)
        >>> pipeline_json
        {'GroupRareLevelsTransformer': {'tubular_version': 'dev', 'classname': 'GroupRareLevelsTransformer', 'init': {'columns': ['categorical_4'], 'copy': False, 'verbose': False, 'return_native': True, 'cut_off_percent': 0.01, 'weights_column': None, 'rare_level_name': 'rare', 'record_rare_levels': True, 'unseen_levels_to_rare': True}, 'fit': {'non_rare_levels': {'categorical_4': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']}}}}


        """
        for transformer in steps:
            if not transformer[1].jsonable:
                return f"{transformer[0]} is not jsonable"

        return {step_name: step.to_json() for step_name, step in steps}

    @classmethod
    def load(cls, pipeline_json: dict) -> Pipeline:
        """Deserialize a pipeline json structure into a pipeline.

        Parameters
        ----------
        pipeline_json: dict
            json dictionary representing the pipeline.

        Returns
        -------
        Pipeline

        Examples
        --------
        >>> from profiling.pipeline_generator import TubularPipelineGenerator
        >>> from profiling import create_dataset as create
        >>> pipe = TubularPipelineGenerator()
        >>> pipe = pipe.generate_pipeline(["GroupRareLevelsTransformer"])
        >>> df_1 = create.create_standard_pandas_dataset()
        >>> a = pipe.fit(df_1, df_1["AveRooms"])
        >>> pipeline_json = PipelineSerializer.dump(pipe.steps)
        >>> pipeline = PipelineSerializer.load(pipeline_json)
        >>> pipeline
        Pipeline(steps=[('GroupRareLevelsTransformer',
                         GroupRareLevelsTransformer(columns=['categorical_4']))])

        """
        steps = [
            (step_name, CLASS_REGISTRY[step_name].from_json(json_dict))
            for step_name, json_dict in pipeline_json.items()
        ]

        return Pipeline(steps)
