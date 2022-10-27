from .numpy_seed import numpy_seed
from .star import star
from .functional_base import FunctionalBase
from .tensor import Tensor
from .numpy import Numpy
from .epochs import Epochs
from .metric import Metric
from .metric_table import MetricTable
from .lambda_module import Lambda
from .module_device import module_device
from .module_train import module_train, module_eval
from .requires_grad import requires_grad, requires_nograd
from .set_learning_rate import set_learning_rate
from .set_seeds import set_seeds
from .worker_init_fn import worker_init_fn
from .progress_bar import ProgressBar

try:
    from .early_stopping import EarlyStopping
except ImportError:
    pass
from .git_info import git_info

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("pytorch-lantern").version
except DistributionNotFound:
    __version__ = "dev"
