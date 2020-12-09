import numpy as np


class ReduceMetric:
    def __init__(self, reduce_fn, compute_fn=None, initial_state=None):
        self.reduce_fn = reduce_fn
        if compute_fn is None:
            self.compute_fn = lambda x: x
        else:
            self.compute_fn = compute_fn
        self.state = initial_state

    def reduce(self, *args, **kwargs):
        return ReduceMetric(
            reduce_fn=self.reduce_fn,
            compute_fn=self.compute_fn,
            initial_state=self.reduce_fn(self.state, *args, **kwargs),
        )

    def compute(self):
        return self.compute_fn(self.state)


def MapMetric(map_fn, compute_fn=np.mean):
    return ReduceMetric(
        reduce_fn=lambda state, *args, **kwargs: state + [map_fn(*args, **kwargs)],
        compute_fn=compute_fn,
        initial_state=list(),
    )
