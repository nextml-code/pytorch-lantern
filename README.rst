================
Pytorch Lantern
================

.. image:: https://badge.fury.io/py/pytorch-lantern.svg
       :target: https://badge.fury.io/py/pytorch-lantern

Lantern contains our process of bringing a project to fruition as
efficiently as possible. This is subject to change as we iterate and improve.
This package implements tools and missing features to help bridge the gap
between frameworks and libraries that we utilize.

The main packages and tools that we build around are:

- pytorch
- pytorch-datastream
- guild

See the `documentation <https://pytorch-lantern.readthedocs.io/en/latest/>`_
for more information.

Create new project with template
================================

Install `cookiecutter <https://github.com/cookiecutter/cookiecutter>`_
and `poetry <https://github.com/python-poetry/poetry>`_:

.. code-block::

    pip install cookiecutter
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

Setup project:

.. code-block::

    cookiecutter https://github.com/aiwizo/pytorch-lantern-template.git
    cd <new-project>
    poetry install

You can now train the placeholder model and inspect the results:

.. code-block::

    guild run prepare
    guild run train
    guild tensorboard

Use lantern without templates
==============================

Install lantern from pypi using pip or poetry:

.. code-block::

    poetry add pytorch-lantern
    # pip install pytorch-lantern
