import numpy as np
import functools
from lantern import FunctionalBase, star
from typing import Callable, Any, Optional, List, Union


class MapMetric(FunctionalBase):
    map_fn: Optional[Callable[..., Any]]
    state: List[Any]

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = True

    def __init__(self, state=list(), map_fn=None):
        super().__init__(
            state=state,
            map_fn=map_fn,
        )

    def map(self, fn):
        if self.map_fn is None:
            map_fn = fn
        else:

            def map_fn(*args, **kwargs):
                return fn(self.map_fn(*args, **kwargs))

        return self.replace(
            map_fn=map_fn,
            state=list(map(fn, self.state)),
        )

    def starmap(self, fn):
        return self.map(star(fn))

    def reduce(self, fn, initial=None):
        if self.map_fn is None:

            def reduce_fn(state, *args):
                return fn(state, *args)

        else:

            def reduce_fn(state, args):
                return fn(state, self.map_fn(args))

        return ReduceMetric(
            reduce_fn=reduce_fn,
            state=functools.reduce(reduce_fn, self.state, initial),
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

    def __call__(self):
        return self.compute()

    def __iter__(self):
        return iter(self.compute())


Metric = MapMetric


class ReduceMetric(FunctionalBase):
    reduce_fn: Callable[..., Any]
    state: Any

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = True

    def update_(self, *args, **kwargs):
        self.state = self.reduce_fn(self.state, *args, **kwargs)
        return self

    def update(self, *args, **kwargs):
        return self.replace(state=self.reduce_fn(self.state, *args, **kwargs))

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


def test_map_update():
    assert Metric().map(lambda x: x * 2).update(2).compute() == [4]


def test_map_after_update():
    assert Metric().update(2).map(lambda x: x * 2).compute() == [4]


def test_reduce():
    assert Metric([2, 3]).reduce(lambda state, x: state + x, initial=0).compute() == 5


def test_update_after_reduce():
    assert (
        Metric([2, 3]).reduce(lambda state, x: state + x, initial=0).update(2).compute()
        == 7
    )


def test_aggregate():
    assert Metric([2, 3, 4]).aggregate(lambda xs: np.mean(xs)).compute() == 3


def test_map_after_aggregate():
    assert (
        Metric([2, 3, 4])
        .aggregate(lambda xs: np.mean(xs))
        .map(lambda x: x**2)
        .compute()
        == 9
    )


def test_update_last():
    assert (
        Metric()
        .aggregate(lambda xs: np.mean(xs))
        .map(lambda x: x**2)
        .update(2)
        .update(3)
        .compute()
        == 2.5**2
    )
