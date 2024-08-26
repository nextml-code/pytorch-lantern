Welcome to pytorch-lantern's documentation!
===========================================

Lantern contains our process of bringing a project to fruition as
efficiently as possible. This is subject to change as we iterate and improve.
This package implements tools and missing features to help bridge the gap
between frameworks and libraries that we utilize.

The main packages and tools that we build around are:

- `pytorch <https://pytorch.org>`_
- `pytorch-datastream <https://github.com/nextml-code/pytorch-datastream>`_
- `guild <https://guild.ai>`_

.. autoclass:: lantern.EarlyStopping
   :members:
   :undoc-members:
   :member-order: bysource
.. autoclass:: lantern.ProgressBar
   :members:
   :undoc-members:
   :member-order: bysource
.. autoclass:: lantern.MetricTable
   :members:
   :undoc-members:
   :member-order: bysource
.. autoclass:: lantern.MapMetric
   :members:
   :undoc-members:
   :member-order: bysource
.. autoclass:: lantern.ReduceMetric
   :members:
   :undoc-members:
   :member-order: bysource
.. autoclass:: lantern.FunctionalBase
   :members:
   :undoc-members:
   :member-order: bysource
.. autoclass:: lantern.Tensor
   :members:
   :undoc-members:
   :member-order: bysource
.. autoclass:: lantern.Numpy
   :members:
   :undoc-members:
   :member-order: bysource

.. autofunction:: lantern.Epochs
.. autofunction:: lantern.module_device
.. autofunction:: lantern.set_seeds
.. autofunction:: lantern.module_train
.. autofunction:: lantern.module_eval
.. autofunction:: lantern.worker_init_fn
.. autofunction:: lantern.set_learning_rate
.. autofunction:: lantern.star
.. autofunction:: lantern.numpy_from_matplotlib_figure
.. autofunction:: lantern.numpy_seed
.. autofunction:: lantern.git_info
