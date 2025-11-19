import numpy as np
import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tests.base_tests import (
    GenericFitTests,
    GenericInitTests,
    GenericTransformTests,
    OtherBaseBehaviourTests,
)
from tests.utils import _check_if_skip_test, _handle_from_json
from tubular.mapping import BaseMappingTransformer


# The first part of this file builds out the tests for BaseMappingTransformer so that they can be
# imported into other test files (by not starting the class name with Test)
# The second part actually calls these tests (along with all other require tests) for the BaseMappingTransformer
class BaseMappingTransformerInitTests(GenericInitTests):
    """
    Tests for BaseMappingTransformer.init().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    def test_no_keys_dict_error(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
    ):
        """Test that an exception is raised if mappings is a dict but with no keys."""

        kwargs = minimal_attribute_dict[self.transformer_name]
        kwargs["mappings"] = {}

        with pytest.raises(
            ValueError,
            match=f"{self.transformer_name}: mappings has no values",
        ):
            uninitialized_transformers[self.transformer_name](**kwargs)

    def test_inferred_return_dtypes(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
    ):
        "test that return_dtypes are inferred correctly if not provided"

        kwargs = minimal_attribute_dict[self.transformer_name]
        kwargs["mappings"] = {
            "a": {"a": 1, "b": 2},
            "b": {"c": True, "d": False},
            "c": {"d": 1.0, "e": 2.0},
        }
        kwargs["return_dtypes"] = None

        transformer = uninitialized_transformers[self.transformer_name](
            **kwargs,
        )

        expected = {
            "a": "Int64",
            "b": "Boolean",
            "c": "Float64",
        }

        actual = transformer.return_dtypes

        assert actual == expected, (
            f"return_dtypes attr not inferred as expected, expected {expected} but got {actual}"
        )

    @pytest.mark.parametrize(
        "mappings",
        [
            {"a": {np.nan: 1, None: 2}},
            {"a": {np.nan: 1, pd.NA: 2}},
            {"a": {None: 1, pd.NA: 2}},
        ],
    )
    def test_multiple_null_mappings_error(
        self,
        uninitialized_transformers,
        minimal_attribute_dict,
        mappings,
    ):
        "verify error thrown if multiple mappings for null are provided"

        return_dtypes = {
            "b": "Int64",
        }

        kwargs = minimal_attribute_dict[self.transformer_name]

        kwargs["mappings"] = mappings
        kwargs["return_dtypes"] = return_dtypes

        with pytest.raises(
            ValueError,
            match="Multiple mappings have been provided for null values in column a, transformer is set up to handle nan/None/NA as one",
        ):
            uninitialized_transformers[self.transformer_name](**kwargs)


class BaseMappingTransformerTransformTests(GenericTransformTests):
    """
    Tests for the transform method on MappingTransformer.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    @pytest.mark.parametrize("from_json", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_mappings_unchanged(
        self,
        minimal_attribute_dict,
        uninitialized_transformers,
        library,
        from_json,
    ):
        """Test that mappings is unchanged in transform."""
        df = d.create_df_3(library=library)

        mapping = {
            "b": {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7},
        }
        return_dtypes = {"b": "Float32"}

        args = minimal_attribute_dict[self.transformer_name].copy()
        args["mappings"] = mapping
        args["return_dtypes"] = return_dtypes

        transformer = uninitialized_transformers[self.transformer_name](**args)

        if _check_if_skip_test(transformer, df, lazy=False, from_json=from_json):
            return

        transformer = _handle_from_json(transformer, from_json)

        transformer.transform(df)

        assert mapping == transformer.mappings, (
            f"{self.transformer_name}.transform has changed self.mappings unexpectedly, expected {mapping} but got {transformer.mappings}"
        )


class TestInit(BaseMappingTransformerInitTests):
    """Tests for BaseMappingTransformer.init()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformer"


class TestFit(GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformer"


class TestTransform(BaseMappingTransformerTransformTests):
    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformer"

    @pytest.mark.parametrize(
        ("df", "expected"),
        ta.pandas.adjusted_dataframe_params(d.create_df_1(), d.create_df_1()),
    )
    def test_X_returned(self, df, expected, uninitialized_transformers):
        """Test that X is returned from transform."""
        mapping = {
            "a": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"},
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        }

        x = BaseMappingTransformer(mappings=mapping)

        df_transformed = x.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=expected,
            actual=df_transformed,
            msg="Check X returned from transform",
        )


class TestOtherBaseBehaviour(OtherBaseBehaviourTests):
    """
    Class to run tests for BaseTransformerBehaviour outside the three standard methods.

    May need to overwite specific tests in this class if the tested transformer modifies this behaviour.
    """

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseMappingTransformer"
