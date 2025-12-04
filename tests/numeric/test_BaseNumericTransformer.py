import re

import narwhals as nw
import pytest

import tests.test_data as d
from tests.base_tests import GenericFitTests, GenericInitTests, GenericTransformTests
from tests.utils import (
    _check_if_skip_test,
    _convert_to_lazy,
    dataframe_init_dispatch,
)


class BaseNumericTransformerInitTests(GenericInitTests):
    """
    Tests for BaseNumericTransformer.init().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """


class BaseNumericTransformerFitTests(GenericFitTests):
    """
    Tests for BaseNumericTransformer.fit().
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("df_generator", "bad_cols"),
        [
            (d.create_df_2, ["b"]),  # str
            (d.create_is_between_dates_df_1, ["a"]),  # datetime
            (d.create_bool_and_float_df, ["b"]),  # bool
            (d.create_df_with_none_and_nan_cols, ["b"]),  # None
        ],
    )
    def test_non_numeric_exception_raised(
        self,
        initialized_transformers,
        df_generator,
        bad_cols,
        library,
        lazy,
    ):
        """Test an exception is raised if self.columns are non-numeric in X."""
        df = df_generator(library=library)

        x = initialized_transformers[self.transformer_name]

        if _check_if_skip_test(x, df, lazy):
            return

        x.columns = bad_cols

        # add in 'target column' for fit
        df = nw.from_native(df)
        native_backend = nw.get_native_namespace(df).__name__
        df = df.with_columns(
            nw.new_series(
                name="c",
                values=[1] * len(df),
                backend=native_backend,
            ),
        ).to_native()

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"{self.transformer_name}: The following columns are not numeric in X; {bad_cols}",
            ),
        ):
            x.fit(_convert_to_lazy(df, lazy), df["c"])

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("df_generator", "cols"),
        [
            (d.create_df_2, ["a"]),  # int
            (d.create_bool_and_float_df, ["a"]),  # float
            (d.create_df_with_none_and_nan_cols, ["a"]),  # nan
        ],
    )
    def test_numeric_passes(
        self,
        initialized_transformers,
        df_generator,
        cols,
        library,
        lazy,
    ):
        """Test check passes if self.columns numeric in X."""
        df = df_generator(library=library)

        x = initialized_transformers[self.transformer_name]
        x.columns = cols

        if _check_if_skip_test(x, df, lazy):
            return

        # add in 'target column' for fit
        df = nw.from_native(df)
        native_backend = nw.get_native_namespace(df).__name__

        # Add this as OneDKmeansTransformer does not accept missing values:
        if self.transformer_name == "OneDKmeansTransformer":
            if native_backend == "polars":
                df = df.with_columns(
                    nw.when(
                        nw.col(cols[0]).is_nan(),
                    )
                    .then(
                        0,
                    )
                    .otherwise(
                        cols[0],
                    )
                    .alias(cols[0]),
                )
            df = df.with_columns(nw.col(cols[0]).fill_null(0))
            # Add this as samples are less than the 8 default clusters
            x.n_clusters = 2

        df = df.with_columns(
            nw.new_series(
                name="c",
                values=[1] * len(df),
                backend=native_backend,
            ),
        ).to_native()

        x.fit(_convert_to_lazy(df, lazy), df["c"])


class BaseNumericTransformerTransformTests(
    GenericTransformTests,
):
    """
    Tests for the transform method on BaseNumericTransformer.
    Note this deliberately avoids starting with "Tests" so that the tests are not run on import.
    """

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("df_generator", "bad_cols"),
        [
            (d.create_df_2, ["b"]),  # str
            (d.create_is_between_dates_df_1, ["a"]),  # datetime
            (d.create_bool_and_float_df, ["b"]),  # bool
            (d.create_df_with_none_and_nan_cols, ["b"]),  # None
        ],
    )
    def test_non_numeric_exception_raised(
        self,
        initialized_transformers,
        df_generator,
        bad_cols,
        library,
        lazy,
    ):
        """Test an exception is raised if self.columns are non-numeric in X."""
        df = df_generator(library=library)

        x = initialized_transformers[self.transformer_name]
        x.columns = bad_cols

        if _check_if_skip_test(x, df, lazy):
            return

        # add in 'target column' for and additional numeric column fit
        df = nw.from_native(df)
        native_backend = nw.get_native_namespace(df).__name__

        # Add this as samples dont have enough volume for 8 default clusters
        if self.transformer_name == "OneDKmeansTransformer":
            # Add this as samples are less than the 8 default clusters
            x.n_clusters = 2

        df = df.with_columns(
            nw.new_series(
                name="c",
                values=[1] * len(df),
                backend=native_backend,
            ),
        ).to_native()

        # if the transformer fits, run a working fit before transform
        if x.FITS:
            # create numeric df to fit on
            df_dict = dict.fromkeys([*x.columns, "c"], df["c"])
            numeric_df = dataframe_init_dispatch(
                dataframe_dict=df_dict,
                library=library,
            )
            x.fit(numeric_df, numeric_df["c"])

        with pytest.raises(
            TypeError,
            match=re.escape(
                rf"{self.transformer_name}: The following columns are not numeric in X; {x.columns}",
            ),
        ):
            x.transform(_convert_to_lazy(df, lazy))

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize(
        ("df_generator"),
        [
            d.create_df_2,  # int
            d.create_bool_and_float_df,  # float
            d.create_df_with_none_and_nan_cols,  # nan
        ],
    )
    def test_numeric_passes(
        self,
        initialized_transformers,
        df_generator,
        library,
        lazy,
    ):
        """Test check passes if self.columns numeric in X."""
        df = df_generator(library=library)

        x = initialized_transformers[self.transformer_name]
        x.columns = ["a", "b"]

        if _check_if_skip_test(x, df, lazy):
            return

        # add in 'target column' for and additional numeric column fit
        df = nw.from_native(df)
        native_backend = nw.get_native_namespace(df).__name__

        # Add this as OneDKmeansTransformer does not accept missing values:
        if self.transformer_name == "OneDKmeansTransformer":
            if native_backend == "polars":
                df = df.with_columns(
                    nw.when(
                        nw.col("a").is_nan(),
                    )
                    .then(
                        0,
                    )
                    .otherwise(
                        nw.col("a"),
                    )
                    .alias("a"),
                )
            df = df.with_columns(nw.col("a").fill_null(0))
            # Add this as samples are less than the 8 default clusters
            x.n_clusters = 2

        df = df.with_columns(
            nw.new_series(
                name="c",
                values=[1] * len(df),
                backend=native_backend,
            ),
            nw.new_series(
                name="b",
                values=[1] * len(df),
                backend=native_backend,
            ),
        ).to_native()

        if x.FITS:
            # create numeric df to fit on
            df_dict = dict.fromkeys([*x.columns, "c"], df["c"])
            numeric_df = dataframe_init_dispatch(
                dataframe_dict=df_dict,
                library=library,
            )
            x.fit(numeric_df, numeric_df["c"])

        x.transform(_convert_to_lazy(df, lazy))


class TestInit(BaseNumericTransformerInitTests):
    """Tests for BaseNumericTransformer.init()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNumericTransformer"


class TestFit(BaseNumericTransformerFitTests):
    """Tests for BaseNumericTransformer.fit()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNumericTransformer"


class TestTransform(BaseNumericTransformerTransformTests):
    """Tests for BaseNumericTransformer.transform()"""

    @classmethod
    def setup_class(cls):
        cls.transformer_name = "BaseNumericTransformer"
