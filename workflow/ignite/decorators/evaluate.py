from functools import wraps

from workflow.torch import model_device
from workflow.ignite.decorators import (
    to_device, no_grad
)


def evaluate(model):
    device = model_device(model)

    def decorator(process_batch):

        @wraps(process_batch)
        @to_device(device)
        @no_grad
        def _process_batch(*args, **kwargs):
            model.eval()
            return process_batch(*args, **kwargs)
        return _process_batch
        
    return decorator