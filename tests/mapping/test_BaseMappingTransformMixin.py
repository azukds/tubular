import copy

import narwhals as nw
import pytest

import tests.test_data as d
from tests.base_tests import (
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.utils import (
    _check_if_skip_test,
    _collect_frame,
    _convert_to_lazy,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)
from tubular.mapping import BaseMappingTransformer, BaseMappingTransformMixin

# Note there are no tests that need inheriting from this file as the only difference is an expected transform output


@pytest.fixture
def mapping():
    return {
        "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h", 9: None},
        "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, None: 9},
    }


class TestInit(ColumnStrListInitTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformMixin"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformMixin"


class TestTransform(GenericTransformTests):
    """
    Tests for BaseMappingTransformMixin.transform().

    Because this is a Mixin transformer it is not always appropriate to inherit the generic transform tests. A number of the tests below overwrite the tests in GenericTransformTests.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformMixin"

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_for_str_and_int(self, mapping, library, lazy):
        """Test outputs for str/int type inputs."""

        df = d.create_df_1(library=library)

        expected_dict = {
            "a": ["a", "b", "c", "d", "e", "f"],
            "b": [1, 2, 3, 4, 5, 6],
        }

        expected = dataframe_init_dispatch(
            dataframe_dict=expected_dict,
            library=library,
        )

        transformer = BaseMappingTransformMixin(columns=["a", "b"])

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        return_dtypes = {"a": "String", "b": "Int64"}

        base_mapping_transformer = BaseMappingTransformer(
            mappings=mapping,
            return_dtypes=return_dtypes,
        )

        transformer.mappings = base_mapping_transformer.mappings
        transformer.return_dtypes = base_mapping_transformer.return_dtypes
        transformer.mappings_from_null = base_mapping_transformer.mappings_from_null

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(expected, _collect_frame(df_transformed, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_for_cat(self, library, lazy):
        """Test that output for cat input"""

        df = d.create_df_2(library=library)

        df = nw.from_native(df).clone().to_native()

        mapping = {
            "c": {
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
                "e": 5,
                "f": 6,
                None: -1,
            },
        }

        expected_dict = {
            "a": [1, 2, 3, 4, 5, 6, None],
            "b": ["a", "b", "c", "d", "e", "f", None],
            "c": [1, 2, 3, 4, 5, 6, -1],
        }

        expected = dataframe_init_dispatch(
            dataframe_dict=expected_dict,
            library=library,
        )

        transformer = BaseMappingTransformMixin(columns=["c"])

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        return_dtypes = {"c": "Int64"}

        base_mapping_transformer = BaseMappingTransformer(
            mappings=mapping,
            return_dtypes=return_dtypes,
        )

        transformer.mappings = base_mapping_transformer.mappings
        transformer.return_dtypes = base_mapping_transformer.return_dtypes
        transformer.mappings_from_null = base_mapping_transformer.mappings_from_null

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(expected, _collect_frame(df_transformed, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_expected_output_boolean_with_nulls(self, library, lazy):
        """Test that output is as expected for tricky bool cases:
        e.g. mapping {True:1, False:0, None: 0}, potential causes of failure:
            - None being cast to False when these values are inserted into bool series
            - None mapping failing, as mapping logic relies on merging and None->None values
            will not merge

        Example failure 1:
        df=pd.DataFrame({'a': [True, False, None]})
        mappings={True:1, False:0, None:0}
        return_dtypes={'a': 'Int8'}
        mapping_transformer=MappingTransformer(mappings, return_dtypes)

        mapping_transformer.transform(df)->
        pd.DataFrame(
            {
            'a': [
                1,
                0,
                None # mapping merge has failed on None,
                # resulting in None instead of 0
            ]
            }
        )

        ---------
        Example Failure 2
        df=pd.DataFrame({'a': [1, 0, -1]})
        mappings={1:True, 0:False, -1:None}
        return_dtypes={'a': 'Int8'}
        mapping_transformer=MappingTransformer(mappings, return_dtypes)

        mapping_transformer.transform(df)->
        pd.DataFrame(
            {
            'a': [
                True,
                False,
                # when the mapping values are put into bool series
                # the none value is converted to False, instead of None
                False,

            ]
            }
        )

        """

        df_dict = {
            "a": [None, 0, 1, None, 0],
            "b": [True, False, None, True, False],
            "c": [None, 0, 0, None, 1],
            "d": [True, None, None, True, False],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        mapping = {
            "a": {0: False, 1: True},
            "b": {False: 0, True: 1},
            "c": {0: False, None: False, 1: True},
            "d": {False: 1, True: 0, None: 1},
        }

        return_dtypes = {
            "a": "Boolean",
            "b": "Float64",
            "c": "Boolean",
            "d": "Int64",
        }

        expected_dict = {
            "a": [None, False, True, None, False],
            "b": [1, 0, None, 1, 0],
            "c": [False, False, False, False, True],
            "d": [0, 1, 1, 0, 1],
        }

        expected = dataframe_init_dispatch(
            dataframe_dict=expected_dict,
            library=library,
        )

        base_mapping_transformer = BaseMappingTransformer(
            mappings=mapping,
            return_dtypes=return_dtypes,
        )

        transformer = BaseMappingTransformMixin(columns=["c"])

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        transformer.mappings = base_mapping_transformer.mappings
        transformer.return_dtypes = base_mapping_transformer.return_dtypes
        transformer.mappings_from_null = base_mapping_transformer.mappings_from_null

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        # convert bool type to pyarrow
        if library == "pandas":
            expected = nw.from_native(expected)
            expected = expected.with_columns(nw.maybe_convert_dtypes(expected["c"]))
            expected = expected.with_columns(nw.maybe_convert_dtypes(expected["a"]))
            expected = expected.to_native()

        assert_frame_equal_dispatch(expected, _collect_frame(df_transformed, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_can_map_values_to_none(self, library, lazy):
        """replace_strict  does  not support mappings values to None, so additional
        logic has been added. Explicitly test a case of mapping to None here.

        """

        df_dict = {
            "a": [1, 2, 3],
        }

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        mapping = {
            "a": {1: 2, 2: 3, 3: None},
        }

        return_dtypes = {
            "a": "Float64",
        }

        expected_dict = {
            "a": [2.0, 3.0, None],
        }

        expected = dataframe_init_dispatch(
            dataframe_dict=expected_dict,
            library=library,
        )

        base_mapping_transformer = BaseMappingTransformer(
            mappings=mapping,
            return_dtypes=return_dtypes,
        )

        transformer = BaseMappingTransformMixin(columns=["a"])

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        transformer.mappings = base_mapping_transformer.mappings
        transformer.return_dtypes = base_mapping_transformer.return_dtypes
        transformer.mappings_from_null = base_mapping_transformer.mappings_from_null

        df_transformed = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(expected, _collect_frame(df_transformed, lazy=lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_mappings_unchanged(self, mapping, library, lazy):
        """Test that mappings is unchanged in transform."""
        df = d.create_df_1(library=library)

        transformer = BaseMappingTransformMixin(columns=["a", "b"])

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        return_dtypes = {
            "a": "String",
            "b": "Int64",
        }

        base_mapping_transformer = BaseMappingTransformer(
            mappings=mapping,
            return_dtypes=return_dtypes,
        )

        transformer.mappings = base_mapping_transformer.mappings
        transformer.return_dtypes = base_mapping_transformer.return_dtypes
        transformer.mappings_from_null = base_mapping_transformer.mappings_from_null

        transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert mapping == transformer.mappings, (
            f"BaseMappingTransformer.transform has changed self.mappings unexpectedly, expected {mapping} but got {transformer.mappings}"
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_original_df_not_updated(self, mapping, library, lazy):
        """Test that the original dataframe is not transformed when transform method used."""

        df = d.create_df_10(library=library)

        transformer = BaseMappingTransformMixin(columns=["a"])

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        return_dtypes = {"a": "String", "b": "Int64"}

        base_mapping_transformer = BaseMappingTransformer(
            mappings=mapping,
            return_dtypes=return_dtypes,
        )

        transformer.mappings = base_mapping_transformer.mappings
        transformer.return_dtypes = base_mapping_transformer.return_dtypes
        transformer.mappings_from_null = base_mapping_transformer.mappings_from_null
        transformer.copy = True

        df = _convert_to_lazy(df, lazy=lazy)

        _ = transformer.transform(df)

        assert_frame_equal_dispatch(
            _collect_frame(df, lazy=lazy), d.create_df_10(library=library)
        )

    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas"],
        indirect=True,
    )
    def test_pandas_index_not_updated(
        self,
        initialized_transformers,
        minimal_dataframe_lookup,
        mapping,
    ):
        """Test that the original (pandas) dataframe index is not transformed when transform method used."""

        df = minimal_dataframe_lookup[self.transformer_name]

        df = nw.from_native(copy.deepcopy(df))
        df = nw.maybe_convert_dtypes(df).to_native()

        transformer = initialized_transformers[self.transformer_name]

        return_dtypes = {"a": "String", "b": "String"}

        base_mapping_transformer = BaseMappingTransformer(
            mappings=mapping,
            return_dtypes=return_dtypes,
        )

        transformer.mappings = base_mapping_transformer.mappings
        transformer.return_dtypes = base_mapping_transformer.return_dtypes
        transformer.mappings_from_null = base_mapping_transformer.mappings_from_null

        # update to abnormal index
        df.index = [2 * i for i in df.index]

        original_df = copy.deepcopy(df)

        transformer = transformer.fit(df, df["a"])

        _ = transformer.transform(df)

        assert_frame_equal_dispatch(df, original_df)

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize(
        "minimal_dataframe_lookup",
        ["pandas", "polars"],
        indirect=True,
    )
    def test_empty_in_empty_out(
        self,
        initialized_transformers,
        minimal_dataframe_lookup,
        mapping,
        lazy,
    ):
        """Test transforming empty frame returns empty frame"""

        df = minimal_dataframe_lookup[self.transformer_name]
        transformer = initialized_transformers[self.transformer_name]

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        return_dtypes = {"a": "String", "b": "String"}

        base_mapping_transformer = BaseMappingTransformer(
            mappings=mapping,
            return_dtypes=return_dtypes,
        )

        transformer.mappings = base_mapping_transformer.mappings
        transformer.return_dtypes = base_mapping_transformer.return_dtypes
        transformer.mappings_from_null = base_mapping_transformer.mappings_from_null

        df = nw.from_native(df)
        # take 0 rows from df
        df = df.head(0).to_native()

        output = transformer.transform(
            _convert_to_lazy(df, lazy=lazy),
        )

        output = nw.from_native(_collect_frame(output, lazy=lazy))

        assert output.shape[0] == 0, (
            "expected empty frame transform to return empty frame"
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_can_handle_lots_of_mappings(self, library, lazy):
        """older implementations had issues erroring for too many mappings, include
        this stress test
        """
        df_dict = {"a": range(1000)}

        df = dataframe_init_dispatch(dataframe_dict=df_dict, library=library)

        mappings = {"a": {i: i + 1 for i in range(1000)}}

        expected_df_dict = {"a": range(1, 1001)}

        expected_df = dataframe_init_dispatch(
            dataframe_dict=expected_df_dict, library=library
        )

        transformer = BaseMappingTransformMixin(columns="a")

        if _check_if_skip_test(transformer, df, lazy=lazy, from_json=False):
            return

        base_mapping_transformer = BaseMappingTransformer(
            mappings=mappings,
        )

        transformer.mappings = base_mapping_transformer.mappings
        transformer.return_dtypes = base_mapping_transformer.return_dtypes
        transformer.mappings_from_null = base_mapping_transformer.mappings_from_null

        output = transformer.transform(_convert_to_lazy(df, lazy=lazy))

        assert_frame_equal_dispatch(_collect_frame(output, lazy=lazy), expected_df)


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwrite specific tests in this class if the tested transformer modifies this behaviour.
    """

    # overload test as class needs special  handling to run
    @pytest.mark.parametrize(
        "library",
        ["pandas", "polars"],
    )
    def test_get_feature_names_out_matches_new_features(
        self,
        library,
        initialized_transformers,
    ):
        """Test that the expected newly created features (if any) are indeed contained
        in the output df"""

        df = d.create_df_1(library=library)

        x = initialized_transformers[self.transformer_name]

        x.mappings = {"b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}}
        x.return_dtypes = {"b": "Int8"}
        x.mappings_from_null = {"b": 1}

        output = x.transform(df)

        output_columns = set(output.columns)

        expected_new_columns = set(x.get_feature_names_out())

        # are expected columns in the data
        assert expected_new_columns.intersection(output_columns), (
            f"{x.classname()}: get_feature_names_out does not agree with output of .transform, expected {expected_new_columns} but got {output_columns}"
        )

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformMixin"
