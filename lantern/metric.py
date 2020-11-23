import numpy as np


class ReduceMetric:
    def __init__(self, reduce, compute, initial_state=None):
        self._reduce = reduce
        self._compute = compute
        self.state = initial_state

    def reduce(self, *args, **kwargs):
        return ReduceMetric(
            reduce=self._reduce,
            compute=self._compute,
            initial_state=self._reduce(self.state, *args, **kwargs),
        )

    def compute(self):
        return self._compute(self.state)


def MapMetric(map, compute=np.mean):
    return ReduceMetric(
        reduce=lambda state, *args, **kwargs: state + [map(*args, **kwargs)],
        compute=compute,
        initial_state=list(),
    )
