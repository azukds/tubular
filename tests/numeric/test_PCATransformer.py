import pandas as pd
import pytest
import test_aide as ta

import tests.test_data as d
from tests.base_tests import (
    CheckNumericFitMixinTests,
    ColumnStrListInitTests,
    GenericFitTests,
    GenericTransformTests,
)
from tubular.numeric import PCATransformer


class TestInit(
    ColumnStrListInitTests,
):
    """Generic tests for transformer.init()."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "PCATransformer"

    def test_to_random_state_type_error(self):
        """Test that an exception is raised if random_state is not a int or None."""
        with pytest.raises(
            TypeError,
            match=r"""PCATransformer:unexpected type <class 'str'> for random_state, must be int or None.""",
        ):
            PCATransformer(columns="b", random_state="2")

    def test_to_svd_solver_type_error(self):
        """Test that an exception is raised if svd_solver is not a str."""
        with pytest.raises(
            TypeError,
            match=r"""PCATransformer:unexpected type <class 'int'> for svd_solver, must be str""",
        ):
            PCATransformer(columns="b", svd_solver=2)

    def test_to_n_components_type_error(self):
        """Test that an exception is raised if n_components is not a int or 'mle'."""
        with pytest.raises(
            TypeError,
            match=r"""PCATransformer:unexpected type <class 'str'> for n_components, must be int, float \(0-1\) or equal to 'mle'.""",
        ):
            PCATransformer(columns="b", n_components="3")

    def test_to_pca_prefix_type_error(self):
        """Test that an exception is raised if pca_column_prefix is not str."""
        with pytest.raises(
            TypeError,
            match=r"""PCATransformer:unexpected type <class 'int'> for pca_column_prefix, must be str""",
        ):
            PCATransformer(columns="b", n_components=2, pca_column_prefix=3)

    def test_to_svd_solver_value_error(self):
        """Test that an exception is raised if svd_solver is not one of the allowed values."""
        with pytest.raises(
            ValueError,
            match=r"""PCATransformer:svd_solver zzz is unknown. Please select among 'auto', 'full', 'arpack', 'randomized'.""",
        ):
            PCATransformer(columns="b", svd_solver="zzz")

    def test_to_n_components_value_error(self):
        """Test that an exception is raised if n_components is not one of the allowed values."""
        with pytest.raises(
            ValueError,
            match=r"""PCATransformer:n_components must be strictly positive got -1""",
        ):
            PCATransformer(columns="b", n_components=-1)

    def test_to_n_components_float_value_error(self):
        """Test that an exception is raised if n_components is not one of the allowed float values."""
        with pytest.raises(
            ValueError,
            match=r"""PCATransformer:n_components must be strictly positive and must be of type int when greater than or equal to 1. Got 1.4""",
        ):
            PCATransformer(columns="b", n_components=1.4)

    def test_to_arpack_mle_value_error(self):
        """Test that an exception is raised if svd solver is arpack and n_components is "mle"."""
        with pytest.raises(
            ValueError,
            match=r"""PCATransformer: n_components='mle' cannot be a string with svd_solver='arpack'""",
        ):
            PCATransformer(columns="b", n_components="mle", svd_solver="arpack")

    def test_to_arpack_randomized_float_type_error(self):
        """Test that an exception is raised if svd solver is arpack or randomized and n_components is float ."""
        with pytest.raises(
            TypeError,
            match=r"""PCATransformer: n_components 0.3 cannot be a float with svd_solver='arpack'""",
        ):
            PCATransformer(columns="b", n_components=0.3, svd_solver="arpack")


class TestFit(CheckNumericFitMixinTests, GenericFitTests):
    """Generic tests for transformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "PCATransformer"

    def test_to_arpack_n_compontes_value_error(self):
        """Test that an exception is raised if svd solver is arpack and n_components greater than nb samples or features."""
        with pytest.raises(
            ValueError,
            match=r"""PCATransformer: n_components 10 must be between 1 and min\(n_samples 10, n_features 2\) is 2 with svd_solver 'arpack'""",
        ):
            # must be between 1 and min(n_samples 10, n_features 2) is 2 with svd_solver arpack
            df = d.create_numeric_df_1()

            x = PCATransformer(columns=["a", "b"], n_components=10, svd_solver="arpack")

            x.fit(df)


class TestTransform(GenericTransformTests):
    """Tests for transformer.transform."""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "PCATransformer"

    def create_svd_sovler_output(self):
        svd_sovler_output = {}
        svd_sovler_output["full"] = pd.DataFrame(
            {
                "a": [34.48, 21.71, 32.83, 1.08, 32.93, 4.74, 2.76, 75.7, 14.08, 61.31],
                "b": [12.03, 20.32, 24.12, 24.18, 68.99, 0.0, 0.0, 59.46, 11.02, 60.68],
                "c": [
                    17.06,
                    12.25,
                    19.15,
                    29.73,
                    1.98,
                    8.23,
                    15.22,
                    20.59,
                    3.82,
                    39.73,
                ],
                "d": [
                    25.94,
                    70.22,
                    72.94,
                    64.55,
                    0.41,
                    13.62,
                    30.22,
                    4.6,
                    67.13,
                    10.38,
                ],
                "e": [94.3, 4.18, 51.7, 16.63, 2.6, 16.57, 3.51, 30.79, 66.19, 25.44],
                "pca_0": [
                    -7.0285210087721985,
                    -10.570772171093276,
                    0.7141476951788178,
                    -19.755517377029697,
                    30.46293987797488,
                    -37.27200224865943,
                    -37.718068808834694,
                    55.636246999483866,
                    -23.564287941836838,
                    49.095834983588574,
                ],
                "pca_1": [
                    -14.719057085223534,
                    0.6588448890236053,
                    -6.504809368610448,
                    8.411936495027216,
                    30.75596190514493,
                    -0.8912674725933973,
                    -2.647964525208776,
                    -9.600190936709105,
                    2.6606364975891146,
                    -8.124090398439629,
                ],
            },
        )

        svd_sovler_output["randomized"] = pd.DataFrame(
            {
                "a": [34.48, 21.71, 32.83, 1.08, 32.93, 4.74, 2.76, 75.7, 14.08, 61.31],
                "b": [12.03, 20.32, 24.12, 24.18, 68.99, 0.0, 0.0, 59.46, 11.02, 60.68],
                "c": [
                    17.06,
                    12.25,
                    19.15,
                    29.73,
                    1.98,
                    8.23,
                    15.22,
                    20.59,
                    3.82,
                    39.73,
                ],
                "d": [
                    25.94,
                    70.22,
                    72.94,
                    64.55,
                    0.41,
                    13.62,
                    30.22,
                    4.6,
                    67.13,
                    10.38,
                ],
                "e": [94.3, 4.18, 51.7, 16.63, 2.6, 16.57, 3.51, 30.79, 66.19, 25.44],
                "pca_0": [
                    -7.028521008772197,
                    -10.570772171093276,
                    0.7141476951788183,
                    -19.755517377029697,
                    30.46293987797488,
                    -37.27200224865943,
                    -37.718068808834694,
                    55.636246999483866,
                    -23.564287941836838,
                    49.09583498358857,
                ],
                "pca_1": [
                    -14.71905708522353,
                    0.6588448890236093,
                    -6.504809368610448,
                    8.411936495027184,
                    30.755961905144947,
                    -0.8912674725933926,
                    -2.647964525208781,
                    -9.600190936709092,
                    2.660636497589127,
                    -8.12409039843965,
                ],
            },
        )

        svd_sovler_output["arpack"] = pd.DataFrame(
            {
                "a": [34.48, 21.71, 32.83, 1.08, 32.93, 4.74, 2.76, 75.7, 14.08, 61.31],
                "b": [12.03, 20.32, 24.12, 24.18, 68.99, 0.0, 0.0, 59.46, 11.02, 60.68],
                "c": [
                    17.06,
                    12.25,
                    19.15,
                    29.73,
                    1.98,
                    8.23,
                    15.22,
                    20.59,
                    3.82,
                    39.73,
                ],
                "d": [
                    25.94,
                    70.22,
                    72.94,
                    64.55,
                    0.41,
                    13.62,
                    30.22,
                    4.6,
                    67.13,
                    10.38,
                ],
                "e": [94.3, 4.18, 51.7, 16.63, 2.6, 16.57, 3.51, 30.79, 66.19, 25.44],
                "pca_0": [
                    -7.0285210087722,
                    -10.570772171093276,
                    0.7141476951788169,
                    -19.75551737702969,
                    30.46293987797488,
                    -37.272002248659426,
                    -37.718068808834694,
                    55.63624699948385,
                    -23.564287941836838,
                    49.09583498358856,
                ],
                "pca_1": [
                    -14.71905708522354,
                    0.6588448890236054,
                    -6.5048093686104504,
                    8.411936495027229,
                    30.755961905144936,
                    -0.8912674725933969,
                    -2.647964525208771,
                    -9.600190936709119,
                    2.660636497589114,
                    -8.124090398439632,
                ],
            },
        )
        return svd_sovler_output

    @pytest.mark.parametrize(
        ("svd_solver", "svd_solver_output_str"),
        [("full", "full"), ("arpack", "arpack"), ("randomized", "randomized")],
    )
    def test_output_from_pca_transform_set_to_columns(
        self,
        mocker,
        svd_solver,
        svd_solver_output_str,
    ):
        """Test that the call to the pca.transform method returns expected outputs."""
        df = d.create_numeric_df_1()

        x = PCATransformer(
            columns=["a", "b", "c"],
            n_components=2,
            svd_solver=svd_solver,
            random_state=32,
        )
        x.fit(df)
        df_transformed = x.transform(df)

        pca_transform_output = self.create_svd_sovler_output()

        mocker.patch(
            "sklearn.decomposition.PCA.transform",
            return_value=pca_transform_output[svd_solver_output_str],
        )

        ta.equality.assert_equal_dispatch(
            expected=pca_transform_output[svd_solver_output_str],
            actual=df_transformed,
            msg=f"output from {svd_solver_output_str} doesn't match",
        )

    @pytest.mark.parametrize("columns", [("b"), ("c"), (["b", "c"])])
    def test_return_type(self, columns):
        """Test that transform returns a pd.DataFrame."""
        df = d.create_numeric_df_1()

        x = PCATransformer(columns=columns, n_components=1)

        x.fit(df)

        df_transformed = x.transform(df)

        assert type(df_transformed) is pd.DataFrame, (
            "unexpected output type from transform"
        )
