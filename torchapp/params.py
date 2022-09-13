import math
from typer.models import OptionInfo
from typing import NamedTuple, List
from numbers import Number


class Param(OptionInfo):
    def __init__(
        self,
        default=None,
        tune=False,
        tune_min=None,
        tune_max=None,
        tune_choices=None,
        log=False,
        distribution=None,
        annotation=None,
        **kwargs,
    ):
        super().__init__(default=default, **kwargs)
        self.tune = tune
        self.log = log
        self.tune_min = tune_min if tune_min is not None else self.min
        self.tune_max = tune_max if tune_max is not None else self.max
        self.annotation = annotation
        self.distribution = distribution
        self.tune_choices = tune_choices
        
        if distribution:
            raise NotImplementedError("Distribution for parameters not implemented yet")
