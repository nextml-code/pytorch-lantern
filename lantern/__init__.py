from lantern import functional

from lantern.numpy_from_matplotlib_figure import numpy_from_matplotlib_figure
from lantern.numpy_seed import numpy_seed

from lantern.functional_base import FunctionalBase
from lantern.tensor import Tensor
from lantern.numpy import Numpy
from lantern.epochs import Epochs
from lantern.metric import Metric
from lantern.metric_table import MetricTable
from lantern.module_device import module_device
from lantern.module_compose import ModuleCompose
from lantern.to_device import to_device
from lantern.to_shapes import to_shapes
from lantern.module_train import module_train, module_eval
from lantern.requires_grad import requires_grad, requires_nograd
from lantern.set_learning_rate import set_learning_rate
from lantern.set_seeds import set_seeds
from lantern.worker_init_fn import worker_init_fn
from lantern.evaluate import evaluate
from lantern.step import step
from lantern.train import train
from lantern.update_cpu_model import update_cpu_model
from lantern.progress_bar import ProgressBar
from lantern.early_stopping import EarlyStopping

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("pytorch-lantern").version
except DistributionNotFound:
    pass
