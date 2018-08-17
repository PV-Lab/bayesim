Examples
========

.. toctree::
   :hidden:

This page is a (growing) list of example applications of ``bayesim``, attempting to showcase the variety of applicability as well as ways to run the code.

Ideal diode solar cell
----------------------
The most self-contained example. Available to run in two different ways:

Jupyter notebook
^^^^^^^^^^^^^^^^
To run in Jupyter:

* Download the `Github repository <https://github.com/PV-Lab/bayesim>`_ and run from your local machine using `Jupyter <http://jupyter.org>`_

OR

* Run in the cloud using Binder! |binderbutton|

.. |binderbutton| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/pv-lab/bayesim/master?filepath=examples%2Fdiode%2Fideal_diode.ipynb

Command line
^^^^^^^^^^^^
To run from the command line:

1. Download the `Github repository <https://github.com/PV-Lab/bayesim>`_ (e.g. using ``git clone``) to your local machine.

2. Navigate to the folder ``bayesim/examples/command_line/``.

3. ...to be continued...(this example is not completely working yet)

Kinematics
----------
Probably the simplest example, explained on the :doc:`bayesics` page. It exists in the `Github repo <https://github.com/PV-Lab/bayesim>`_ under ``examples/kinematics/kinematics.py`` and you can run it all in one go from a terminal (``python kinematics.py``) or step-by-step in an IDE like Spyder.

Tin Sulfide (SnS) solar cell
----------------------------
An example using an actual numerical model -- in this case, reproducing the fit (from `this paper <https://www.sciencedirect.com/science/article/pii/S254243511730096X>`_) of four material and interface properties in a tin sulfide solar cell. Also in Jupyter notebook form:

* Download the `Github repository <https://github.com/PV-Lab/bayesim>`_ and run from your local machine using `Jupyter <http://jupyter.org>`_

OR

* Run in the cloud using Binder! |binderbutton2|

.. |binderbutton2| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/pv-lab/bayesim/master?filepath=examples%2FSnS%2FSnS_fitting.ipynb
