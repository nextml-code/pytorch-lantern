import numpy as np
from functools import wraps


class NumpySeed:
    """
    Function decorator that sets a temporary numpy seed during execution.
    Can be used as a decorator or context manager
    """

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.random_state = np.random.get_state()
        np.random.seed(self.seed)
        return self.random_state

    def __exit__(self, type, value, traceback):
        np.random.set_state(self.random_state)

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with NumpySeed(self.seed):
                return fn(*args, **kwargs)

        return wrapper


numpy_seed = NumpySeed


def test_unchanged_random_state():
    random_state = np.random.get_state()
    with numpy_seed(1):
        np.random.random()
    assert np.all(random_state[1] == np.random.get_state()[1])


def test_same_result():
    with numpy_seed(1):
        result = np.random.random()

    assert result == numpy_seed(1)(np.random.random)()


def test_different_result():
    with numpy_seed(1):
        result = np.random.random()

    assert result != numpy_seed(None)(np.random.random)()
