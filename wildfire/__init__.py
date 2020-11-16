from wildfire import functional

from wildfire.numpy.figure_to_numpy import figure_to_numpy
from wildfire.numpy.numpy_seed import numpy_seed

from wildfire.module_device import module_device
from wildfire.module_compose import ModuleCompose
from wildfire.to_device import to_device
from wildfire.to_shapes import to_shapes
from wildfire.module_train import module_train, module_eval
from wildfire.requires_grad import requires_grad, requires_nograd
from wildfire.set_learning_rate import set_learning_rate
from wildfire.set_seeds import set_seeds
from wildfire.evaluate import evaluate
from wildfire.step import step
from wildfire.train import train
from wildfire.update_cpu_model import update_cpu_model
from wildfire.progress_bar import ProgressBar
from wildfire.early_stopping import EarlyStopping

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution('pytorch-wildfire').version
except DistributionNotFound:
    pass
