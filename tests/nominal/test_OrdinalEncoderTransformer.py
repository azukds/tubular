import pytest
import test_aide as ta
import tests.test_data as d
import pandas as pd

import tubular
from tubular.nominal import OrdinalEncoderTransformer


class TestInit(object):
    """Tests for OrdinalEncoderTransformer.init()."""

    def test_arguments(self):
        """Test that init has expected arguments."""

        ta.functions.test_function_arguments(
            func=OrdinalEncoderTransformer.__init__,
            expected_arguments=["self", "response_column", "columns", "weights_column"],
            expected_default_values=(None, None),
        )

    def test_class_methods(self):
        """Test that OrdinalEncoderTransformer has fit and transform methods."""

        x = OrdinalEncoderTransformer(response_column="a")

        ta.classes.test_object_method(obj=x, expected_method="fit", msg="fit")

        ta.classes.test_object_method(
            obj=x, expected_method="transform", msg="transform"
        )

    def test_inheritance(self):
        """Test that NominalToIntegerTransformer inherits from BaseNominalTransformer."""

        x = OrdinalEncoderTransformer(response_column="a")

        ta.classes.assert_inheritance(x, tubular.nominal.BaseNominalTransformer)
        ta.classes.assert_inheritance(x, tubular.mapping.BaseMappingTransformMixin)

    def test_super_init_called(self, mocker):
        """Test that init calls BaseNominalTransformer.__init__."""

        spy = mocker.spy(tubular.nominal.BaseNominalTransformer, "__init__")

        x = OrdinalEncoderTransformer(
            response_column="a", columns=None, verbose=True, copy=True
        )

        assert (
            spy.call_count == 1
        ), "unexpected number of calls to BaseTransformer.__init__"

        call_args = spy.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        expected_kwargs = {"columns": None, "verbose": True, "copy": True}

        assert (
            call_kwargs == expected_kwargs
        ), "unexpected kwargs in BaseTransformer.__init__ call"

        expected_pos_args = (x,)

        assert (
            len(call_pos_args) == 1
        ), "unexpected # positional args in BaseTransformer.__init__ call"

        assert (
            expected_pos_args == call_pos_args
        ), "unexpected positional args in BaseTransformer.__init__ call"

    def test_response_column_not_str_error(self):
        """Test that an exception is raised if response_column is not a str."""

        with pytest.raises(TypeError, match="response_column should be a str"):

            OrdinalEncoderTransformer(response_column=1)

    def test_weights_column_not_str_error(self):
        """Test that an exception is raised if weights_column is not a str."""

        with pytest.raises(TypeError, match="weights_column should be a str"):

            OrdinalEncoderTransformer(response_column="a", weights_column=1)

    def test_response_weights_column_equal_error(self):
        """Test that an exception is raised if weights_column and response_column are equal."""

        with pytest.raises(
            ValueError, match="weights_column and response_column are the same column"
        ):

            OrdinalEncoderTransformer(response_column="a", weights_column="a")

    def test_values_passed_in_init_set_to_attribute(self):
        """Test that the values passed in init are saved in an attribute of the same name."""

        x = OrdinalEncoderTransformer(response_column="aaa")

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={"response_column": "aaa"},
            msg="Attributes for OrdinalEncoderTransformer set in init",
        )


class TestFit(object):
    """Tests for OrdinalEncoderTransformer.fit()"""

    def test_arguments(self):
        """Test that init fit expected arguments."""

        ta.functions.test_function_arguments(
            func=OrdinalEncoderTransformer.fit,
            expected_arguments=["self", "X", "y"],
            expected_default_values=(None,),
        )

    def test_super_fit_called(self, mocker):
        """Test that fit calls BaseNominalTransformer.fit."""

        df = d.create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(response_column="a", columns="b")

        spy = mocker.spy(tubular.nominal.BaseNominalTransformer, "fit")

        x.fit(df)

        assert spy.call_count == 1, "unexpected number of calls to BaseTransformer.fit"

        call_args = spy.call_args_list[0]
        call_pos_args = call_args[0]
        call_kwargs = call_args[1]

        expected_kwargs = {}

        assert (
            call_kwargs == expected_kwargs
        ), "unexpected kwargs in BaseTransformer.fit call"

        expected_pos_args = (
            x,
            d.create_OrdinalEncoderTransformer_test_df(),
            None,
        )

        assert len(expected_pos_args) == len(
            call_pos_args
        ), "unexpected # positional args in BaseTransformer.fit call"

        assert (
            expected_pos_args[0] == call_pos_args[0]
        ), "unexpected 1st positional arg in BaseTransformer.fit call"

        ta.equality.assert_equal_dispatch(
            expected_pos_args[1:3],
            call_pos_args[1:3],
            "unexpected 2nd, 3rd positional arg in BaseTransformer.fit call",
        )

    def test_fit_returns_self(self):
        """Test fit returns self?"""

        df = d.create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(response_column="a", columns="b")

        x_fitted = x.fit(df)

        assert (
            x_fitted is x
        ), "Returned value from create_OrdinalEncoderTransformer_test_df.fit not as expected."

    def test_fit_not_changing_data(self):
        """Test fit does not change X."""

        df = d.create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(response_column="a", columns="b")

        x.fit(df)

        ta.equality.assert_equal_dispatch(
            expected=d.create_OrdinalEncoderTransformer_test_df(),
            actual=df,
            msg="Check X not changing during fit",
        )

    def test_learnt_values(self):
        """Test that the ordinal encoder values learnt during fit are expected."""

        df = d.create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(response_column="a", columns=["b", "d", "f"])

        x.fit(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "mappings": {
                    "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
                    "d": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
                    "f": {False: 1, True: 2},
                }
            },
            msg="mappings attribute",
        )

    def test_learnt_values_weight(self):
        """Test that the ordinal encoder values learnt during fit are expected if a weights column is specified."""

        df = d.create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(
            response_column="a", weights_column="e", columns=["b", "d", "f"]
        )

        x.fit(df)

        ta.classes.test_object_attributes(
            obj=x,
            expected_attributes={
                "mappings": {
                    "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
                    "d": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
                    "f": {False: 1, True: 2},
                }
            },
            msg="mappings attribute",
        )

    def test_response_column_missing_error(self):
        """Test that an exception is raised if response_column is not present in data."""

        df = d.create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(response_column="z", columns=["b", "d", "f"])

        with pytest.raises(ValueError, match="response z not in X"):

            x.fit(df)

    def test_weights_column_missing_error(self):
        """Test that an exception is raised if weights_column is specified but not present in data for fit."""

        df = d.create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(
            response_column="a", weights_column="z", columns=["b", "d", "f"]
        )

        with pytest.raises(ValueError, match="weights column z not in X"):

            x.fit(df)

    def test_response_column_nulls_error(self):
        """Test that an exception is raised if nulls are present in response_column."""

        df = d.create_df_4()

        x = OrdinalEncoderTransformer(response_column="a", columns=["b"])

        with pytest.raises(ValueError):

            x.fit(df)


class TestTransform(object):
    """Tests for OrdinalEncoderTransformer.transform()."""

    def expected_df_1():
        """Expected output for ."""

        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 2, 3, 4, 5, 6],
                "c": ["a", "b", "c", "d", "e", "f"],
                "d": [1, 2, 3, 4, 5, 6],
                "e": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "f": [1, 1, 1, 2, 2, 2],
            }
        )

        df["c"] = df["c"].astype("category")

        return df

    def test_arguments(self):
        """Test that transform has expected arguments."""

        ta.functions.test_function_arguments(
            func=OrdinalEncoderTransformer.transform, expected_arguments=["self", "X"]
        )

    def test_check_is_fitted_called(self, mocker):
        """Test that BaseTransformer check_mappable_rows called."""

        df = d.create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(response_column="a", columns="b")

        x.fit(df)

        expected_call_args = {0: {"args": (df,), "kwargs": {}}}

        with ta.functions.assert_function_call(
            mocker,
            tubular.nominal.BaseNominalTransformer,
            "check_mappable_rows",
            expected_call_args,
        ):

            x.transform(df)

    def test_super_transform_called(self, mocker):
        """Test that BaseMappingTransformMixin.transform called."""

        df = d.create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(response_column="a", columns="b")

        x.fit(df)

        expected_call_args = {
            0: {
                "args": (
                    x,
                    d.create_OrdinalEncoderTransformer_test_df(),
                ),
                "kwargs": {},
            }
        }

        with ta.functions.assert_function_call(
            mocker,
            tubular.mapping.BaseMappingTransformMixin,
            "transform",
            expected_call_args,
            return_value=d.create_OrdinalEncoderTransformer_test_df(),
        ):

            x.transform(df)

    def test_learnt_values_not_modified(self):
        """Test that the mappings from fit are not changed in transform."""

        df = d.create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(response_column="a", columns="b")

        x.fit(df)

        x2 = OrdinalEncoderTransformer(response_column="a", columns="b")

        x2.fit(df)

        x2.transform(df)

        ta.equality.assert_equal_dispatch(
            expected=x.mappings,
            actual=x2.mappings,
            msg="Mean response values not changed in transform",
        )

    @pytest.mark.parametrize(
        "df, expected",
        ta.pandas.adjusted_dataframe_params(
            d.create_OrdinalEncoderTransformer_test_df(), expected_df_1()
        ),
    )
    def test_expected_output(self, df, expected):
        """Test that the output is expected from transform."""

        x = OrdinalEncoderTransformer(response_column="a", columns=["b", "d", "f"])

        # set the impute values dict directly rather than fitting x on df so test works with helpers
        x.mappings = {
            "b": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
            "d": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
            "f": {False: 1, True: 2},
        }

        df_transformed = x.transform(df)

        ta.equality.assert_frame_equal_msg(
            actual=df_transformed,
            expected=expected,
            msg_tag="Unexpected values in OrdinalEncoderTransformer.transform",
        )

    def test_nulls_introduced_in_transform_error(self):
        """Test that transform will raise an error if nulls are introduced."""

        df = d.create_OrdinalEncoderTransformer_test_df()

        x = OrdinalEncoderTransformer(response_column="a", columns=["b", "d", "f"])

        x.fit(df)

        df["b"] = "z"

        with pytest.raises(
            ValueError,
            match="nulls would be introduced into column b from levels not present in mapping",
        ):

            x.transform(df)
