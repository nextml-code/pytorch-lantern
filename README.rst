================
Pytorch Wildfire
================

.. image:: https://badge.fury.io/py/pytorch-wildfire.svg
       :target: https://badge.fury.io/py/pytorch-wildfire

Wildfire contains our process of bringing a project to fruition as
efficiently as possible. This is subject to change as we iterate and improve.
This package implements tools and missing features to help bridge the gap
between frameworks and libraries that we utilize.

The main packages and tools that we build around are:

- pytorch
- pytorch-datastream
- guild

See the `documentation <https://pytorch-wildfire.readthedocs.io/en/latest/>`_
for more information.

Create new project with template
================================

.. code-block::

    cookiecutter https://github.com/aiwizo/pytorch-wildfire-template.git
    cd <new-project>
    poetry install

You can train a model and inspect the training with:

.. code-block::

    guild run prepare
    guild run train
    guild tensorboard

Use wildfire without templates
==============================

.. code-block::

    pip install pytorch-wildfire
