import pytest
from torchapp import params


def test_assert_distribution():
    with pytest.raises(NotImplementedError):
        params.Param(annotation=float, tune_min=1.0, tune_max=100.0, distribution="uniform")
