import numpy as np
from abc import ABC, abstractmethod
from lantern import FunctionalBase
from typing import Callable, Any, Optional


class Metric(ABC):
    @abstractmethod
    def update(self):
        ...

    @abstractmethod
    def update_(self):
        ...

    @abstractmethod
    def compute(self):
        ...


class ReduceMetric(FunctionalBase, Metric):
    reduce_fn: Callable
    compute_fn: Callable
    state: Optional[Any]

    class Config:
        allow_mutation = True

    def __init__(self, reduce_fn, compute_fn=None, initial_state=None):
        super().__init__(
            reduce_fn=reduce_fn,
            compute_fn=((lambda x: x) if compute_fn is None else compute_fn),
            state=initial_state,
        )

    def update(self, *args, **kwargs):
        return self.replace(state=self.reduce_fn(self.state, *args, **kwargs))

    def update_(self, *args, **kwargs):
        self.state = self.reduce_fn(self.state, *args, **kwargs)
        return self

    def compute(self):
        return self.compute_fn(self.state)

    def log(self, tensorboard_logger, tag, name, step=1):
        tensorboard_logger.add_scalar(
            f"{tag}/{name}",
            self.compute(),
            step,
        )
        return self


def MapMetric(map_fn, compute_fn=np.mean):
    """Metric version of `compute_fn(map(map_fn, input))`"""
    return ReduceMetric(
        reduce_fn=lambda state, *args, **kwargs: state + [map_fn(*args, **kwargs)],
        compute_fn=compute_fn,
        initial_state=list(),
    )
