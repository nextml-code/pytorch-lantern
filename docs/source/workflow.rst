wildfire package
================

Setup project
-------------
Use ``python -m wildfire.setup_project`` to set up a new project as shown
below:

.. code-block::

    mkdir new-project
    cd new-project
    virtualenv venv -p python3.8
    source venv/bin/activate
    pip install pytorch-wildfire
    python -m wildfire.setup_project

    pip install -r requirements.txt
    pip install -r dev_requirements.txt
    pip freeze > dev_requirements.txt

    # reactivate environment to find guild
    deactivate
    source venv/bin/activate

Subpackages
-----------

.. toctree::
   :maxdepth: 3

   wildfire.functional
   wildfire.ignite
   wildfire.torch

.. autofunction:: wildfire.figure_to_numpy
.. autofunction:: wildfire.numpy_seed
