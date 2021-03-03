import numpy as np
from lantern import FunctionalBase
from lantern.functional import star
from typing import Callable, Any, Optional, Dict, List, Union
from pydantic import BaseModel, Extra


class MapMetric(FunctionalBase):
    map_fn: Optional[Callable[..., Any]]
    state: List[Any]

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = True
        extra = Extra.forbid

    def __init__(self, map_fn=None, state=list()):
        super().__init__(
            map_fn=map_fn,
            state=state,
        )

    def map(self, fn):
        if self.map_fn is None:
            return self.replace(map_fn=fn)
        else:
            return self.replace(
                map_fn=lambda *args, **kwargs: fn(self.map_fn(*args, **kwargs))
            )

    def starmap(self, fn):
        return self.map(star(fn))

    def reduce(self, fn):
        if self.map_fn is None:
            return ReduceMetric(
                map_fn=lambda *args: args,
                reduce_fn=lambda state, args: fn(state, *args),
                state=self.state,  # TODO: apply function on state...
            )
        else:
            return ReduceMetric(
                map_fn=self.map_fn,
                reduce_fn=fn,
                state=self.state,
            )

    def aggregate(self, fn):
        return AggregateMetric(metric=self, aggregate_fn=fn)

    def staraggregate(self, fn):
        return self.aggregate(star(fn))

    def update_(self, *args, **kwargs):
        if self.map_fn is None:
            self.state.append(args)
        else:
            self.state.append(self.map_fn(*args, **kwargs))
        return self

    def update(self, *args, **kwargs):
        if self.map_fn is None:
            return self.replace(
                state=self.state + ([args[0]] if len(args) == 1 else [args])
            )
        else:
            return self.replace(state=self.state + [self.map_fn(*args, **kwargs)])

    def compute(self):
        return self.state

    def log(self, tensorboard_logger, tag, metric_name, step=None):
        tensorboard_logger.add_scalar(
            f"{tag}/{metric_name}",
            self.compute(),
            step,
        )
        return self

    def log_dict(self, tensorboard_logger, tag, step=None):
        for name, value in self.compute().items():
            tensorboard_logger.add_scalar(
                f"{tag}/{name}",
                value,
                step,
            )
        return self


Metric = MapMetric


class ReduceMetric(FunctionalBase):
    map_fn: Callable[..., Any]
    reduce_fn: Callable[..., Any]
    state: Any

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = True

    def replace(self, **kwargs):
        new_dict = self.dict()
        new_dict.update(**kwargs)
        return type(self)(**new_dict)

    def update_(self, *args, **kwargs):
        self.state = self.reduce_fn(self.state, self.map_fn(*args, **kwargs))
        return self

    def update(self, *args, **kwargs):
        return self.replace(
            state=self.reduce_fn(self.state, self.map_fn(*args, **kwargs))
        )

    def compute(self):
        return self.state

    def log(self, tensorboard_logger, tag, metric_name, step=None):
        tensorboard_logger.add_scalar(
            f"{tag}/{metric_name}",
            self.compute(),
            step,
        )
        return self

    def log_dict(self, tensorboard_logger, tag, step=None):
        for name, value in self.compute().items():
            tensorboard_logger.add_scalar(
                f"{tag}/{name}",
                value,
                step,
            )
        return self


class AggregateMetric(FunctionalBase):
    metric: Union[MapMetric, ReduceMetric]
    aggregate_fn: Callable

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = True

    def map(self, fn):
        return self.replace(aggregate_fn=lambda state: fn(self.aggregate_fn(state)))

    def starmap(self, fn):
        return self.map(star(fn))

    def update_(self, *args, **kwargs):
        self.metric = self.metric.update(*args, **kwargs)
        return self

    def update(self, *args, **kwargs):
        return self.replace(metric=self.metric.update(*args, **kwargs))

    def compute(self):
        return self.aggregate_fn(self.metric.compute())

    def log(self, tensorboard_logger, tag, metric_name, step=None):
        tensorboard_logger.add_scalar(
            f"{tag}/{metric_name}",
            self.compute(),
            step,
        )
        return self

    def log_dict(self, tensorboard_logger, tag, step=None):
        for name, value in self.compute().items():
            tensorboard_logger.add_scalar(
                f"{tag}/{name}",
                value,
                step,
            )
        return self


def test_metric():
    pass
