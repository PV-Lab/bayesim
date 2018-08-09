.. OpenBTE documentation master file, created by
   sphinx-quickstart on Mon Dec  4 16:00:38 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting Started
===============

.. toctree::
   :hidden:

Prerequisites
-------------

``bayesim`` has been most thoroughly tested in Python 3.6.

Download
--------

To install bayesim simply type

.. code-block:: shell

  pip install bayesim

Usage
-----

Python / Jupyter
^^^^^^^^^^^^^^^^

The easiest way to learn how ``bayesim`` works is by stepping through one of the :doc:`Examples </examples>`, which have well-commented `Jupyter notebooks <http://jupyter.org>`_ associated with them and can be run in `binder <https://mybinder.org>`_ with no need to install Python, ``bayesim``, or any of its dependencies locally.

If you are comfortable coding in Python, these examples will also make it clear how to script using ``bayesim``.

Command line
^^^^^^^^^^^^

We have also developed a command line interface...


.. In case you are familiar with Python, you can setup a simulation with a script, i.e.

 .. code-block:: python

   from openbte.material import *
   from openbte.geometry import *
   from openbte.solver import *
   from openbte.plot import *

   mat = Material(matfile='Si-300K',n_mfp=10,n_theta=6,n_phi=32)

   geo = Geometry(type='porous/aligned',lx=10,ly=10,
                 porosity = 0.25,
                 step = 1.0,
                 shape = 'square')

   sol = Solver()

   Plot(variable='map/flux_bte/magnitude')


 .. _ShengBTE: http://www.shengbte.com
