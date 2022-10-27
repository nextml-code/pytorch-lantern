================
Pytorch Lantern
================

.. image:: https://badge.fury.io/py/pytorch-lantern.svg
       :target: https://badge.fury.io/py/pytorch-lantern

.. image:: https://img.shields.io/pypi/pyversions/pytorch-lantern.svg
       :target: https://pypi.python.org/pypi/pytorch-lantern

.. image:: https://readthedocs.org/projects/pytorch-lantern/badge/?version=latest
       :target: https://pytorch-lantern.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/pypi/l/pytorch-lantern.svg
       :target: https://pypi.python.org/pypi/pytorch-lantern

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

Lantern contains our process of bringing a project to fruition as
efficiently as possible. This is subject to change as we iterate and improve.
This package implements tools and missing features to help bridge the gap
between frameworks and libraries that we utilize.

The main packages and tools that we build around are:

- `pytorch <https://pytorch.org>`_
- `pytorch-datastream <https://github.com/Aiwizo/pytorch-datastream>`_
- `guild <https://guild.ai>`_


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

    poetry add pytorch-lantern[training]

.. code-block::

    pip install pytorch-lantern[training]
