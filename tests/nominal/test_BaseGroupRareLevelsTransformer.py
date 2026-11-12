import pytest

import tests.test_data as d
from tubular.nominal import BaseGroupRareLevelsTransformer


class TestGetTransformExprs:
    """Tests for BaseGroupRareLevelsTransformer.get_transform_exprs."""

    def test_not_implemented_error(self):
        """Test that the base class requires type specific subclasses to define transform exprs."""
        transformer = BaseGroupRareLevelsTransformer(columns=["b"])

        with pytest.raises(
            NotImplementedError,
            match="BaseGroupRareLevelsTransformer: get_transform_exprs is not implemented",
        ):
            transformer.get_transform_exprs()

    def test_no_allowed_types_error_on_fit(self):
        """Test that base class cannot be fit, as it has no allowed types."""
        transformer = BaseGroupRareLevelsTransformer(columns=["b"])

        with pytest.raises(TypeError, match="transformer can only handle"):
            transformer.fit(d.create_df_8("polars"))
