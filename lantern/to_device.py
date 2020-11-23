import torch
from functools import wraps


def to_device(x, device):
    if callable(x):
        return to_device_decorator(device)(x)
    elif type(x) == tuple:
        return tuple(to_device(value, device) for value in x)
    elif type(x) == list:
        return [to_device(value, device) for value in x]
    elif type(x) == dict:
        return {key: to_device(value, device) for key, value in x.items()}
    elif type(x) == torch.Tensor:
        return x.to(device)
    else:
        return x


def to_device_decorator(device):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            fn(
                *[to_device(value, device) for value in args],
                **{key: to_device(value, device) for key, value in kwargs.items},
            )

        return wrapper

    return decorator
